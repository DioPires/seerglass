# SPDX-License-Identifier: Apache-2.0
"""Offline evaluation: score one configuration on labeled frames.

A configuration is a detector, a recognizer, a bank, a catalog and the rules around them
(``config``). The frames are the ones ``cardvision fetch`` wrote for a label manifest, and the
labels are the manifest's rows. Four measures:

* ``hover`` and ``events``: for each labeled position, what a pointer resting there would
  show, and whether that is right, wrong or nothing (``scoring.run_hover``);
* ``crops``: the recognizer alone, on crops cut from labeled face boxes;
* ``detector``: the detector alone, on exhaustively boxed frames: recall at false-positive
  caps, and AP@0.5;
* ``scoring.run_compare``: two configurations on the same hover events, as a transition table
  with paired differences.

``scoring.analyze_hover`` analyzes each frame once and scores the same hover events under any
number of accept rules; ``cardvision.calibrate`` builds on it.

``stats`` holds the two intervals every rate carries, ``rows`` the account of which rows were
scored, and ``report`` the JSON result and the Markdown report.

``browser`` reads hover outcomes that the browser runtime's scoring page recorded on video
files, and scores them with the same events, rates and intervals as ``hover``.

``scenes`` reads a folder of scenes that ``cardvision make-composites`` generated as rows and
frames, so that ``hover``, ``crops`` and ``detector`` score it by the generator's exact labels.
Generated scenes are not footage, and the report says so.
"""

from cardvision.evaluate.browser import (
    BrowserConfiguration,
    BrowserEventsError,
    BrowserRun,
    check_correspondence,
    manifest_warnings,
    open_browser_configuration,
    read_browser_events,
    run_browser_hover,
)
from cardvision.evaluate.config import Configuration, Spec, file_sha256, open_configuration, spec_from_paths, spec_from_release
from cardvision.evaluate.crops import CropResult, crop_rates, crop_results, run_crops
from cardvision.evaluate.detector import (
    FrameRecord,
    average_precision,
    cap_thresholds,
    detect_frame,
    detector_results,
    frame_truth,
    greedy_match,
    polygon_iou,
    run_detector,
    score_frame,
    walk_down,
)
from cardvision.evaluate.events import (
    Event,
    Scored,
    classify,
    is_adjacent,
    named_face_centers,
    rates,
    row_events,
    score_event,
    score_row,
    shown_identity,
    tally,
    truth_identity,
)
from cardvision.evaluate.hover import (
    ACCEPTED_NMS_IOU,
    SELECTIONS,
    HoverLayer,
    Proposal,
    Shown,
    hit,
    hitbox_quad,
    hover_order,
    kept_after_nms,
    point_in_convex_quad,
    proposals_of,
    resolve,
)
from cardvision.evaluate.report import build_result, format_rate, to_markdown, write_result
from cardvision.evaluate.rows import Account, read_rows, scorable_rows
from cardvision.evaluate.scenes import SceneFrames, ScenesError, read_scenes
from cardvision.evaluate.scoring import (
    REGRESSIONS,
    AnalyzedRow,
    HoverAnalysis,
    analyze_hover,
    compare_results,
    hover_results,
    paired_differences,
    run_compare,
    run_hover,
    transition,
    transitions,
)
from cardvision.evaluate.stats import GroupBootstrap, binomial_cdf, bootstrap_draws, exact_lower, exact_upper, rate

__all__ = [
    "ACCEPTED_NMS_IOU",
    "REGRESSIONS",
    "SELECTIONS",
    "Account",
    "AnalyzedRow",
    "BrowserConfiguration",
    "BrowserEventsError",
    "BrowserRun",
    "Configuration",
    "CropResult",
    "Event",
    "FrameRecord",
    "GroupBootstrap",
    "HoverAnalysis",
    "HoverLayer",
    "Proposal",
    "SceneFrames",
    "ScenesError",
    "Scored",
    "Shown",
    "Spec",
    "analyze_hover",
    "average_precision",
    "binomial_cdf",
    "bootstrap_draws",
    "build_result",
    "cap_thresholds",
    "check_correspondence",
    "classify",
    "compare_results",
    "crop_rates",
    "crop_results",
    "detect_frame",
    "detector_results",
    "exact_lower",
    "exact_upper",
    "file_sha256",
    "format_rate",
    "frame_truth",
    "greedy_match",
    "hit",
    "hitbox_quad",
    "hover_order",
    "hover_results",
    "is_adjacent",
    "kept_after_nms",
    "manifest_warnings",
    "named_face_centers",
    "open_browser_configuration",
    "open_configuration",
    "paired_differences",
    "point_in_convex_quad",
    "polygon_iou",
    "proposals_of",
    "rate",
    "rates",
    "read_browser_events",
    "read_rows",
    "read_scenes",
    "resolve",
    "row_events",
    "run_browser_hover",
    "run_compare",
    "run_crops",
    "run_detector",
    "run_hover",
    "scorable_rows",
    "score_event",
    "score_frame",
    "score_row",
    "shown_identity",
    "spec_from_paths",
    "spec_from_release",
    "tally",
    "to_markdown",
    "transition",
    "transitions",
    "truth_identity",
    "walk_down",
    "write_result",
]
