# SPDX-License-Identifier: Apache-2.0
"""Label manifests: the format, its validator, role assignment, frame fetching and readers.

A label manifest is a text file that says, for frames of public videos, where the cards are
and which cards they are. It holds no pixels: a row names a video and a time, and the frame is
obtained from the video under the extraction contract.

* ``schema``: the format (``Row``, ``Box``, ``Point``, ``ROW_SCHEMA``).
* ``manifest``: read, write and filter manifests.
* ``geometry``: fractions of the frame, pixels and card outlines.
* ``validate``: the check run on manifests before they are accepted.
* ``roles``: the seeded rule that gives each recording group a role.
* ``sources``: a game's source types and its hand-row setting.
* ``dhash``: the fingerprint of a frame, to tell whether two images show the same frame.
* ``extract``: the extraction contract and the frame folder.
* ``fetch``: frames from local video files, or by downloading when asked.
* ``reachable``: a manifest's reachable fields, set from the index of a fetch run.
* ``targets``: rows to detector targets and recognizer crops.

Only the standard library and NumPy are needed to read and check manifests; reading video and
downloading are optional extras.
"""

from cardvision.data.extract import (
    FORMAT_SELECTOR,
    RATIO_TOLERANCE,
    FrameMissing,
    FrameStore,
    frame_check,
    frame_file_name,
    frame_index,
    frame_time_at_or_after,
    mid_frame_time,
    on_screen_time,
    read_frame_at,
    read_index,
    sampled_frame_time,
)
from cardvision.data.geometry import (
    bounding_box,
    clockwise,
    quad_from_box,
    quad_from_rotated_box,
    quarter_turns,
    side_aspect,
    to_fractions,
    to_pixels,
)
from cardvision.data.manifest import iter_rows, read_manifest, rows_by_group, rows_by_role, write_manifest
from cardvision.data.roles import (
    assign_by_quota,
    assign_by_share,
    assign_manifest_roles,
    pseudonymous_group_id,
    split_in_order,
    tie_groups,
)
from cardvision.data.schema import (
    COMPLETENESS,
    IDENTITIES,
    KINDS,
    LOCAL_PLATFORM,
    MANIFEST_VERSION,
    PLATFORMS,
    READABLE,
    ROLES,
    ROW_SCHEMA,
    SHAPES,
    SOURCE_TYPES,
    Box,
    ManifestError,
    Point,
    Row,
    parse_row,
    row_problems,
)
from cardvision.data.sources import SourceConfig, load_source_config
from cardvision.data.targets import (
    Crop,
    DetectorTarget,
    TargetRules,
    classify_box,
    crop_exclusion,
    detector_labels,
    detector_targets,
    margin_crop,
    negative_crops,
    recognizer_crops,
    write_crops,
)
from cardvision.data.validate import Problem, Report, validate_manifests

__all__ = [
    "COMPLETENESS",
    "FORMAT_SELECTOR",
    "IDENTITIES",
    "KINDS",
    "LOCAL_PLATFORM",
    "MANIFEST_VERSION",
    "PLATFORMS",
    "RATIO_TOLERANCE",
    "READABLE",
    "ROLES",
    "ROW_SCHEMA",
    "SHAPES",
    "SOURCE_TYPES",
    "Box",
    "Crop",
    "DetectorTarget",
    "FrameMissing",
    "FrameStore",
    "ManifestError",
    "Point",
    "Problem",
    "Report",
    "Row",
    "SourceConfig",
    "TargetRules",
    "assign_by_quota",
    "assign_by_share",
    "assign_manifest_roles",
    "bounding_box",
    "classify_box",
    "clockwise",
    "crop_exclusion",
    "detector_labels",
    "detector_targets",
    "frame_check",
    "frame_file_name",
    "frame_index",
    "frame_time_at_or_after",
    "iter_rows",
    "load_source_config",
    "margin_crop",
    "mid_frame_time",
    "on_screen_time",
    "negative_crops",
    "parse_row",
    "pseudonymous_group_id",
    "quad_from_box",
    "quad_from_rotated_box",
    "quarter_turns",
    "read_frame_at",
    "read_index",
    "read_manifest",
    "recognizer_crops",
    "row_problems",
    "rows_by_group",
    "rows_by_role",
    "sampled_frame_time",
    "side_aspect",
    "split_in_order",
    "tie_groups",
    "to_fractions",
    "to_pixels",
    "validate_manifests",
    "write_crops",
    "write_manifest",
]
