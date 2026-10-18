# SPDX-License-Identifier: Apache-2.0
"""The ``cardvision infer`` subcommand: images or video in, boxes and names out.

One JSON object per frame is printed to standard output; everything else (progress, warnings)
goes to standard error, so the output can be piped. The optional outputs are files:
annotated frames, the crops, a contact sheet (in pages when it is long), a prefill file for the
labeling tool, and the unmarked frames that the prefill file refers to.

The models come from a release document (``--release``), which pins every file and setting, or
from the separate files (``--detector``, ``--recognizer``, ``--bank``, ``--catalog`` and
optionally ``--accept``). The two forms cannot be mixed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

SUMMARY = "Detect and name the cards in images or video."

# At most this many frames with no box are listed one by one; the rest are counted.
EMPTY_FRAMES_LISTED = 5


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the subcommand's arguments."""
    parser.add_argument("inputs", nargs="+", metavar="IMAGE_OR_VIDEO",
                        help="an image, a video, or a folder of images; several may be given")
    parser.add_argument("--release", metavar="R.json", help="a release document; it pins every file and setting")
    parser.add_argument("--detector", metavar="X.onnx", help="the detector model (without --release)")
    parser.add_argument("--recognizer", metavar="Y.onnx", help="the recognizer model (without --release)")
    parser.add_argument("--bank", metavar="B.json", help="the embedding bank built with that recognizer (without --release)")
    parser.add_argument("--catalog", metavar="C.json", help="the game's catalog (without --release)")
    parser.add_argument("--accept", metavar="accept.json",
                        help='the accept rule, {"tau": ..., "delta": ..., "rotationRetry": ...}; '
                             "the published model's values, with the rotation retry on, when omitted")
    parser.add_argument("--geometric", metavar="G.json",
                        help="the geometric stage's settings file, which pins its descriptor index: the recognizer's "
                             "candidates are verified with local features (needs the geometric extra). It may be "
                             "given with --release; without it nothing changes")
    parser.add_argument("--art", metavar="DIR_OR_URL",
                        help="folder or URL base that the catalog's art paths are resolved against (contact sheet only)")
    parser.add_argument("--annotate", metavar="DIR", help="write each frame with its boxes and names drawn")
    parser.add_argument("--crops", metavar="DIR", help="write each face crop as the recognizer saw it")
    parser.add_argument("--contact-sheet", metavar="FILE",
                        help="write an image of each crop beside its three best matches. A sheet too long for one "
                             "image is written as pages: FILE-1, FILE-2 and so on, with FILE's suffix")
    parser.add_argument("--contact-sheet-rows", type=int, default=None, metavar="N",
                        help="most crops on the contact sheet, over all its pages (default 60)")
    parser.add_argument("--contact-sheet-page-rows", type=int, default=None, metavar="N",
                        help="most crops on one image of the contact sheet (default 20)")
    parser.add_argument("--prefill-out", metavar="FILE", help="write a prefill file for the labeling tool")
    parser.add_argument("--frames", metavar="DIR",
                        help="write each analyzed frame unmarked, under the name the prefill file uses")
    parser.add_argument("--every", type=float, default=1.0, metavar="SECONDS",
                        help="for video, the time between analyzed frames (default 1.0)")
    parser.add_argument("--frame-format", choices=("jpg", "png"), default="jpg",
                        help="for video, the image format the frames are named for and --frames writes them in "
                             "(default jpg). png is lossless; the labeling tool can fingerprint PNG frames, so "
                             "that a manifest row made from one is found again by its content")


def _note(message: str) -> None:
    print(f"cardvision infer: {message}", file=sys.stderr)


def _score_text(score: float) -> str:
    """A score to two decimals, cut and not rounded, so that a score under the threshold never
    reads as equal to it."""
    return f"{math.floor(score * 100 + 1e-6) / 100:.2f}"


def _threshold_text(threshold: float) -> str:
    return f"{threshold:.2f}" if round(threshold, 2) == threshold else f"{threshold:g}"


def _explain_empty_frames(empty: list, frames: int, detector, min_crop_px: int) -> None:
    """Say, on standard error, why frames with no box have none: the run's count and threshold,
    then for each of the first few frames the highest score the detector gave anything. The
    frames after those are counted, the ones whose boxes were all too small apart from the
    ones in which nothing reached the threshold, and the highest score is that of the latter."""
    threshold = _threshold_text(detector.threshold)
    _note(f"no box in {len(empty)} of {frames} frame(s); the detector keeps a box at a score of {threshold} or more")
    for result in empty[:EMPTY_FRAMES_LISTED]:
        if result.too_small:
            _note(f"{result.name}: {result.too_small} box(es) found, but each is under {min_crop_px} pixels "
                  "wide or high, too small to read")
        elif result.best_score is not None:
            _note(f"{result.name}: highest score {_score_text(result.best_score)}, threshold {threshold}")
    left_out = empty[EMPTY_FRAMES_LISTED:]
    if not left_out:
        return
    small = sum(1 for result in left_out if result.too_small)
    scores = [result.best_score for result in left_out if result.best_score is not None and not result.too_small]
    if not small:
        detail = f"; the highest score in them is {_score_text(max(scores))}" if scores else ""
    else:
        detail = f": {small} with only boxes too small to read"
        if scores:
            detail += f", and the highest score in the other {len(left_out) - small} is {_score_text(max(scores))}"
    _note(f"and {len(left_out)} more frame(s) with no box{detail}")


def _spread(counts: list[int]) -> str:
    """The smallest, the median and the largest of some counts, as ``5 / 6 / 6``."""
    return f"{min(counts)} / {statistics.median(counts):g} / {max(counts)}"


def _run(args: argparse.Namespace) -> int:
    from cardvision.core.catalog import IDENTITY_NAME
    from cardvision.core.constants import DETECTOR_BACK_CLASS
    from cardvision.evaluate.config import open_configuration, spec_from_arguments
    from cardvision.infer.contact_sheet import (
        DEFAULT_MAX_ROWS,
        DEFAULT_PAGE_ROWS,
        ArtLibrary,
        SheetRow,
        sheet_paths,
        sheet_rows,
        write_contact_sheet,
    )
    from cardvision.infer.frames import expand_inputs, read_frames
    from cardvision.infer.prefill import prefill_record
    from cardvision.infer.render import write_annotated, write_crops, write_frame

    if not args.release:
        for label, value in (("detector", args.detector), ("recognizer", args.recognizer),
                             ("bank", args.bank), ("catalog", args.catalog), ("accept rule", args.accept)):
            if value is not None and not Path(value).is_file():
                raise FileNotFoundError(f"{value}: no such {label} file")
    spec = spec_from_arguments(args.release, args.detector, args.recognizer, args.bank, args.catalog, args.accept)
    if args.geometric:
        spec = spec.replacing(geometric=args.geometric)
    expand_inputs(args.inputs)  # fail on a missing input before any model is loaded
    max_rows = DEFAULT_MAX_ROWS if args.contact_sheet_rows is None else args.contact_sheet_rows
    if max_rows < 1:
        raise ValueError("--contact-sheet-rows must be at least 1")
    page_rows = DEFAULT_PAGE_ROWS if args.contact_sheet_page_rows is None else args.contact_sheet_page_rows
    if page_rows < 1:
        raise ValueError("--contact-sheet-page-rows must be at least 1")

    configuration = open_configuration(spec)
    catalog, pipeline = configuration.catalog, configuration.pipeline
    stage = pipeline.recognizer.geometric
    if stage is not None:
        if stage.opencv_note():
            _note(f"warning: {stage.opencv_note()}")
        if stage.config["gate"]["sources"] is not None:
            _note("the geometric stage's gate.sources is set and infer knows no source type, so the stage will "
                  "not run on these inputs")
    # Where different cards can share a name, a drawn name is followed by the card id.
    with_ids = catalog.identity != IDENTITY_NAME

    rows: list[SheetRow] = []
    skipped_rows = 0
    prefill = None
    if args.prefill_out:
        Path(args.prefill_out).parent.mkdir(parents=True, exist_ok=True)
        prefill = open(args.prefill_out, "w", encoding="utf-8")
    frames = faces = backs = named = 0
    faces_per_frame: list[int] = []
    backs_per_frame: list[int] = []
    empty = []
    try:
        for frame in read_frames(args.inputs, args.every, args.frame_format):
            result = pipeline.analyze_frame(frame)
            print(json.dumps(result.to_json()), flush=True)
            frames += 1
            faces_per_frame.append(len(result.faces))
            backs_per_frame.append(sum(1 for card in result.cards if card.cls == DETECTOR_BACK_CLASS))
            faces += faces_per_frame[-1]
            backs += backs_per_frame[-1]
            if not result.cards:
                empty.append(result)
            named += sum(1 for card in result.faces if card.recognition.accepted)
            if args.frames:
                write_frame(args.frames, frame.rgb, frame.name)
            if args.annotate:
                write_annotated(args.annotate, frame.rgb, result, with_ids)
            if args.crops:
                write_crops(args.crops, result)
            if prefill is not None:
                prefill.write(json.dumps(prefill_record(result)) + "\n")
            if args.contact_sheet:
                new_rows = sheet_rows(result)
                room = max(0, max_rows - len(rows))
                rows.extend(new_rows[:room])
                skipped_rows += len(new_rows) - min(room, len(new_rows))
    finally:
        if prefill is not None:
            prefill.close()

    if args.contact_sheet:
        art = ArtLibrary(catalog, args.art)
        write_contact_sheet(args.contact_sheet, rows, art, max_rows, page_rows)
        files = sheet_paths(args.contact_sheet, len(rows), page_rows)
        if len(files) == 1:
            _note(f"the contact sheet is {files[0]}")
        else:
            _note(f"the contact sheet has {len(files)} pages of at most {page_rows} crops: {files[0]} to {files[-1]}")
        if skipped_rows:
            _note(f"the contact sheet shows the first {len(rows)} crops; {skipped_rows} more were left out "
                  "(raise --contact-sheet-rows to include them)")
        if art.certificate_problem is not None:
            _note(f"no card image was downloaded: {art.certificate_problem}")
        if art.missing:
            _note(f"no card image could be loaded for {len(art.missing)} card(s); "
                  "the contact sheet shows their names and scores only"
                  + ("" if args.art else " (pass --art with the folder or URL base of the card images)"))
    if empty:
        _explain_empty_frames(empty, frames, pipeline.detector, pipeline.min_crop_px)
    _note(f"{frames} frame(s), {faces} face(s), {backs} back(s), {named} named")
    if frames > 1:
        # A card missed in every frame does not show in the totals; a count per frame, set
        # against what is on the table, does.
        _note(f"boxes per frame, fewest / median / most: faces {_spread(faces_per_frame)}, backs {_spread(backs_per_frame)}")
    return 0


def run(args: argparse.Namespace) -> int:
    """Run the subcommand. Returns the process exit code: 0, or 2 after a reported error."""
    from cardvision.bank.release import ReleaseError
    from cardvision.core.catalog import CatalogError
    from cardvision.infer.frames import VideoSupportMissing

    try:
        return _run(args)
    except ReleaseError as error:
        _note("error: the release cannot be used: " + "; ".join(error.errors))
    except KeyError as error:
        _note(f"error: {error.args[0] if error.args else error}")
    except (VideoSupportMissing, CatalogError, FileNotFoundError, ValueError, ImportError) as error:
        _note(f"error: {error}")
    return 2
