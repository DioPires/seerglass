# SPDX-License-Identifier: Apache-2.0
"""Reference inference: from an image or a video to boxes and names.

``Pipeline`` runs the detector and the recognizer on a frame. Around it are the frame readers
(``frames``), the pictures (``render``, ``contact_sheet``), the prefill writer for the labeling
tool (``prefill``) and the command line (``cli``).
"""

from cardvision.infer.contact_sheet import ArtLibrary, SheetRow, contact_sheet, sheet_paths, sheet_rows, write_contact_sheet
from cardvision.infer.frames import (
    MAX_ANALYSIS_WIDTH,
    MIN_ANALYSIS_WIDTH,
    Frame,
    VideoSupportMissing,
    analysis_frame,
    analysis_size,
    expand_inputs,
    is_video,
    read_frames,
    read_video,
)
from cardvision.infer.pipeline import Card, FrameResult, Pipeline, card_json, frame_box, reported_box
from cardvision.infer.prefill import prefill_box, prefill_flag, prefill_record, write_prefill
from cardvision.infer.render import annotate, write_annotated, write_crops, write_frame

__all__ = [
    "MAX_ANALYSIS_WIDTH",
    "MIN_ANALYSIS_WIDTH",
    "ArtLibrary",
    "Card",
    "Frame",
    "FrameResult",
    "Pipeline",
    "SheetRow",
    "VideoSupportMissing",
    "analysis_frame",
    "analysis_size",
    "annotate",
    "card_json",
    "contact_sheet",
    "expand_inputs",
    "frame_box",
    "is_video",
    "prefill_box",
    "prefill_flag",
    "prefill_record",
    "read_frames",
    "read_video",
    "reported_box",
    "sheet_paths",
    "sheet_rows",
    "write_annotated",
    "write_contact_sheet",
    "write_crops",
    "write_frame",
]
