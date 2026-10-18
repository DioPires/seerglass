# SPDX-License-Identifier: Apache-2.0
"""Run a detector model on a frame: letterbox, one model run, decode."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cardvision.core.onnx import DEFAULT_PROVIDERS, create_session
from cardvision.detect.decode import MAX_DETECTIONS, NMS_IOU, SCORE_THRESHOLD, Detection, best_score, decode
from cardvision.detect.letterbox import INPUT_HEIGHT, INPUT_WIDTH, PAD_VALUE, letterbox_bgr, to_input_tensor

if TYPE_CHECKING:
    import onnxruntime as ort


def model_input_size(session: ort.InferenceSession) -> tuple[int, int]:
    """``(width, height)`` of the model's input, read from its declared shape.

    Falls back to 640 x 384 on an axis the model declares as dynamic.
    """
    shape = session.get_inputs()[0].shape
    height = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else INPUT_HEIGHT
    width = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else INPUT_WIDTH
    return width, height


class Detector:
    """A detector model and its operating point.

    The model takes one ``(1, 3, height, width)`` float32 tensor (see
    ``cardvision.detect.letterbox``) and returns ``(1, anchors, 5 + classes)`` decoded rows
    (see ``cardvision.detect.decode``).
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        threshold: float = SCORE_THRESHOLD,
        nms_iou: float = NMS_IOU,
        max_detections: int = MAX_DETECTIONS,
        pad: int = PAD_VALUE,
    ) -> None:
        self.session = session
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.pad = pad
        self.input_name = session.get_inputs()[0].name
        self.input_width, self.input_height = model_input_size(session)

    @classmethod
    def open(cls, model_path: str | Path, *, providers: Sequence[str] = DEFAULT_PROVIDERS, **settings) -> Detector:
        """Open a detector ONNX file. ``settings`` are the keyword arguments of the constructor."""
        return cls(create_session(model_path, providers=providers), **settings)

    def raw(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """The model's rows for one RGB frame, ``(anchors, 5 + classes)``, and the letterbox ratio."""
        canvas, ratio = letterbox_bgr(rgb, self.input_width, self.input_height, self.pad)
        rows = self.session.run(None, {self.input_name: to_input_tensor(canvas)})[0][0]
        return rows, ratio

    def look(self, rgb: np.ndarray, threshold: float | None = None) -> tuple[list[Detection], float]:
        """``detect`` and, from the same model run, the highest score of any row before the threshold.

        The detections are exactly those of ``detect``. The score tells a frame with nothing in
        it from one where the detector came close (``cardvision.detect.decode.best_score``).
        """
        rows, ratio = self.raw(rgb)
        found = decode(rows, ratio, self.threshold if threshold is None else threshold, self.nms_iou, self.max_detections)
        return found, best_score(rows)

    def detect(self, rgb: np.ndarray, threshold: float | None = None) -> list[Detection]:
        """Detections for one ``(H, W, 3)`` uint8 RGB frame, in that frame's pixels, best first.

        ``threshold`` overrides the detector's own for this call. Boxes are not clipped to the
        frame and none is dropped for being small; the caller decides what is worth cropping.
        """
        rows, ratio = self.raw(rgb)
        return decode(rows, ratio, self.threshold if threshold is None else threshold, self.nms_iou, self.max_detections)
