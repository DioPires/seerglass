# SPDX-License-Identifier: Apache-2.0
"""The card detector: its input contract, its output decode and a runner.

``letterbox`` turns a frame into the model's input, ``decode`` turns the model's rows into
boxes, and ``Detector`` does both around one model run. Only NumPy, Pillow and ONNX Runtime are
needed.
"""

from cardvision.detect.decode import (
    CLASS_NAMES,
    MAX_DETECTIONS,
    NMS_IOU,
    SCORE_THRESHOLD,
    Detection,
    best_score,
    class_name,
    corner_boxes,
    decode,
    iou_one_to_many,
    nms,
    scores_and_classes,
)
from cardvision.detect.detector import Detector, model_input_size
from cardvision.detect.letterbox import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    PAD_VALUE,
    letterbox_bgr,
    letterbox_ratio,
    letterbox_size,
    to_input_tensor,
)

__all__ = [
    "CLASS_NAMES",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "MAX_DETECTIONS",
    "NMS_IOU",
    "PAD_VALUE",
    "SCORE_THRESHOLD",
    "Detection",
    "Detector",
    "best_score",
    "class_name",
    "corner_boxes",
    "decode",
    "iou_one_to_many",
    "letterbox_bgr",
    "letterbox_ratio",
    "letterbox_size",
    "model_input_size",
    "nms",
    "scores_and_classes",
    "to_input_tensor",
]
