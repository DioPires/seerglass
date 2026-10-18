# SPDX-License-Identifier: Apache-2.0
"""Constants of the detector-to-recognizer hand-off, shared by every tool that reads boxes."""

# The detector predicts two classes. A face is cropped and passed to the recognizer; a back is
# reported as a card but never named.
DETECTOR_FACE_CLASS = 0
DETECTOR_BACK_CLASS = 1

# A box whose short side is below this many pixels, measured in the frame the detector
# analyzed, is not a valid proposal and is dropped before recognition.
DETECTOR_MIN_CROP_PX = 24

# The lowest detector score threshold the export's box parity check covers: it bounds the boxes
# of the anchors any threshold of 0.01 or more keeps, so a release may not use a lower one.
DETECTOR_PARITY_SCORE_FLOOR = 0.01
