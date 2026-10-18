# SPDX-License-Identifier: Apache-2.0
"""The optional geometric stage: verify the recognizer's candidates with local features.

The recognizer compares one embedding of a crop with one row per card. This stage looks again
at a few candidate cards, by matching local keypoints of the crop with keypoints taken from each
card's image and fitting a geometric model to the matches (``stage``). The inliers of that model
are fused with the cosine, and the fused scores can re-rank the candidates or lift a correct
best card over the accept threshold, by settings (``config``). The keypoints of every card are
computed once into a descriptor index (``index``, ``cardvision build-features``).

The stage is optional at every level. It needs OpenCV (the ``geometric`` extra), which is
imported only when the stage is used. A recognizer without a stage, and every command run
without ``--geometric``, behave exactly as they did before the stage existed. The browser
runtime has no geometric stage.
"""

from cardvision.geometric.config import (
    DEFAULTS,
    FEATURE_PARAMS,
    FEATURE_TYPES,
    GeometricConfig,
    GeometricConfigError,
    load_geometric,
    parse_geometric,
)

__all__ = [
    "DEFAULTS",
    "FEATURE_PARAMS",
    "FEATURE_TYPES",
    "GeometricConfig",
    "GeometricConfigError",
    "load_geometric",
    "parse_geometric",
]
