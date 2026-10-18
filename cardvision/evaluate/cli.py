# SPDX-License-Identifier: Apache-2.0
"""The ``cardvision evaluate`` subcommand: score a configuration on labeled frames, offline.

::

    cardvision evaluate hover|crops|detector|compare
        --manifest FILE [FILE ...] --frames DIR
        (--release R.json | --detector X --recognizer Y --bank B --catalog C [--accept A])
        [--geometric G.json] [--role evaluation] [--out DIR] [--bootstrap N] [--seed S]

``hover`` scores what a pointer would show at each labeled position; ``crops`` the recognizer
alone on labeled face crops; ``detector`` the detector alone on exhaustively boxed frames;
``compare`` two configurations on the same hover events. For ``compare`` the arguments above
describe the base, and the candidate is either another release (``--other-release``) or the
base with some files replaced (``--other-detector``, ``--other-recognizer``, ``--other-bank``,
``--other-catalog``, ``--other-accept``, ``--other-geometric``). ``--geometric`` adds the
optional geometric stage (``cardvision.geometric``) to the configuration, a release included;
``compare --other-geometric G.json`` scores the base against the base with the stage.

The Markdown report is printed to standard output. With ``--out`` the JSON result, the report
and the per-item listings are also written to that folder. ``hover --evidence FILE`` also writes
what every face of every frame came to, one JSON line per frame (``cardvision.evaluate.evidence``),
for comparing two runs face by face; it changes no figure. The frames are the folder that
``cardvision fetch`` wrote. Nothing is downloaded.

``hover`` has a second form, for outcomes that were recorded in a web browser by the scoring
page of ``browser/`` (``cardvision.evaluate.browser``)::

    cardvision evaluate hover --browser-events FILE --manifest FILE [FILE ...]
        (--catalog C.json | --release R.json) [--role evaluation] [--out DIR] [--bootstrap N] [--seed S]

No model runs and no frame is read in this form: the events file holds what was shown at each
labeled position, and the same scorer and bootstrap turn it into the same result, report and
listing. The catalog gives the labeled cards their names. With ``--release`` the catalog
is the release's, and the release id recorded in the events file must be that document's.

``hover``, ``crops`` and ``detector`` have a third form, for scenes that ``cardvision
make-composites`` generated from card images (``cardvision.evaluate.scenes``)::

    cardvision evaluate hover|crops|detector --scenes DIR
        (--release R.json | --detector X --recognizer Y --bank B --catalog C [--accept A])
        [--out DIR] [--bootstrap N] [--seed S]

The folder stands in the place of ``--manifest`` and ``--frames``: its images are the frames
and the generator's exact labels are the rows. The scoring is the same code. The report opens
by saying that the rows are generated scenes and not footage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUMMARY = "Score a detector, a recognizer and a bank on labeled frames."
MEASURES = ("hover", "crops", "detector", "compare")
_OTHER_PARTS = ("detector", "recognizer", "bank", "catalog", "accept", "geometric")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the subcommand's arguments."""
    from cardvision.data.schema import ROLES
    from cardvision.evaluate.detector import DEFAULT_CAPS
    from cardvision.evaluate.rows import DEFAULT_ROLE
    from cardvision.evaluate.stats import DEFAULT_REPLICATES, DEFAULT_SEED

    parser.add_argument("measure", choices=MEASURES, help="what to measure")
    parser.add_argument("--manifest", nargs="+", metavar="FILE",
                        help="one or more label manifests (required, except with --scenes)")
    parser.add_argument("--frames", metavar="DIR",
                        help="the folder of frames that cardvision fetch wrote (required, except with --browser-events "
                             "and with --scenes)")
    parser.add_argument("--scenes", metavar="DIR",
                        help="hover, crops and detector: score the scenes in this folder, which cardvision "
                             "make-composites generated, by their exact labels; takes the place of --manifest and "
                             "--frames. Generated scenes are not footage, and the report says so")
    parser.add_argument("--browser-events", metavar="FILE",
                        help="hover only: score the outcomes in this events file, which the scoring page of browser/ "
                             "recorded in a web browser, in place of running the models; takes --manifest and "
                             "--catalog (or --release, to check the release id), and no --frames or model files")
    parser.add_argument("--release", metavar="R.json", help="a release document; it pins every file and setting")
    parser.add_argument("--detector", metavar="X.onnx", help="the detector model (without --release)")
    parser.add_argument("--recognizer", metavar="Y.onnx", help="the recognizer model (without --release)")
    parser.add_argument("--bank", metavar="B.json", help="the embedding bank built with that recognizer (without --release)")
    parser.add_argument("--catalog", metavar="C.json", help="the game's catalog (without --release)")
    parser.add_argument("--accept", metavar="A.json",
                        help='the accept rule, {"tau": ..., "delta": ..., "rotationRetry": ...}; '
                             "the published model's values, with the rotation retry on, when omitted")
    parser.add_argument("--geometric", metavar="G.json",
                        help="the geometric stage's settings file, which pins its descriptor index: the recognizer's "
                             "candidates are verified with local features (needs the geometric extra). It may be "
                             "given with --release; without it nothing changes")
    parser.add_argument("--role", choices=ROLES,
                        help=f"the role of the rows to score (default {DEFAULT_ROLE}; generated scenes have none)")
    parser.add_argument("--out", metavar="DIR", help="write the JSON result, the report and the listings here")
    parser.add_argument("--evidence", metavar="FILE",
                        help="hover only: also write per-face evidence, one JSON line per frame "
                             "(cardvision.evaluate.evidence); with a geometric stage it records the stage's debug "
                             "account of each crop. It changes nothing that is scored")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_REPLICATES, metavar="N",
                        help=f"bootstrap replicates (default {DEFAULT_REPLICATES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="S",
                        help=f"seed of the bootstrap (default {DEFAULT_SEED})")
    other = parser.add_argument_group("the candidate of compare")
    other.add_argument("--other-release", metavar="R.json", help="the candidate, as a release document")
    for part in _OTHER_PARTS:
        help_text = (f"the candidate's {part} file, replacing the base's" if part != "geometric" else
                     "the candidate's geometric stage settings file, or none for no stage; required when the base has "
                     "--geometric and the candidate has another release or bank")
        other.add_argument(f"--other-{part}", metavar="FILE", help=help_text)
    detector = parser.add_argument_group("detector only")
    detector.add_argument("--sources", metavar="S.json",
                          help="the game's sources file, for its hand-row rule when boxes become detector targets")
    detector.add_argument("--fp-caps", default=",".join(f"{cap:g}" for cap in DEFAULT_CAPS), metavar="LIST",
                          help="false positives per frame at which recall is read, comma-separated "
                               f"(default {','.join(f'{cap:g}' for cap in DEFAULT_CAPS)})")


def _note(message: str) -> None:
    print(f"cardvision evaluate: {message}", file=sys.stderr)


def _base_spec(args: argparse.Namespace):
    from cardvision.evaluate.config import spec_from_arguments

    spec = spec_from_arguments(args.release, args.detector, args.recognizer, args.bank, args.catalog, args.accept)
    return spec.replacing(geometric=args.geometric) if args.geometric else spec


def _candidate_spec(args: argparse.Namespace, base):
    """The candidate of ``compare``.

    The geometric stage follows the base only while the base's bank does: when the base has
    ``--geometric`` and the candidate is another release or has another bank, the candidate's
    stage must be said with ``--other-geometric`` (a settings file, or ``none`` for no stage),
    since the base's index was built for the base's bank.
    """
    import dataclasses

    from cardvision.evaluate.config import spec_from_release

    replaced = {part: getattr(args, f"other_{part}") for part in _OTHER_PARTS if getattr(args, f"other_{part}")}
    geometric = replaced.pop("geometric", None)
    if args.other_release:
        if replaced:
            raise ValueError("give --other-release, or --other-* files, not both (--other-geometric may be added)")
    elif not replaced and geometric is None:
        raise ValueError("compare needs a candidate: --other-release, or at least one of "
                         + ", ".join(f"--other-{part}" for part in _OTHER_PARTS))
    if base.geometric is not None and geometric is None and (args.other_release or "bank" in replaced):
        raise ValueError("the base has --geometric, whose index was built for the base's bank: say the candidate's "
                         "stage with --other-geometric G.json, or --other-geometric none to score it without one")
    candidate = spec_from_release(args.other_release) if args.other_release else base.replacing(**replaced)
    if geometric is not None:
        candidate = dataclasses.replace(candidate, geometric=None if geometric == "none" else Path(geometric))
    return candidate


def _caps(text: str) -> tuple[float, ...]:
    try:
        caps = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise ValueError(f"--fp-caps must be numbers separated by commas, got {text!r}") from None
    if not caps or any(cap < 0 for cap in caps):
        raise ValueError(f"--fp-caps must list one or more numbers of 0 or more, got {text!r}")
    return caps


def _check_scenes_arguments(args: argparse.Namespace) -> None:
    """Refuse what has no meaning beside ``--scenes``."""
    from pathlib import Path

    if args.measure == "compare":
        raise ValueError("--scenes is read by hover, crops and detector, not by compare")
    given = [f"--{name.replace('_', '-')}" for name in ("manifest", "frames", "browser_events", "role")
             if getattr(args, name)]
    if given:
        raise ValueError("--scenes gives the rows and the frames, and generated scenes have no role; do not also give "
                         + ", ".join(given))
    if not Path(args.scenes).is_dir():
        raise FileNotFoundError(f"{args.scenes}: no such folder of generated scenes")


def _run_browser(args: argparse.Namespace) -> int:
    """The second form of ``hover``: score the outcomes a browser recorded."""
    from cardvision.evaluate import report, scoring
    from cardvision.evaluate.browser import manifest_warnings, open_browser_configuration, run_browser_hover
    from cardvision.evaluate.rows import DEFAULT_ROLE, read_rows

    if args.measure != "hover":
        raise ValueError(f"--browser-events holds hover outcomes and has no meaning for {args.measure}")
    given = [f"--{name}" for name in ("frames", "detector", "recognizer", "bank", "accept", "geometric")
             if getattr(args, name)]
    if given:
        raise ValueError("--browser-events holds outcomes that were computed in the browser; do not also give "
                         + ", ".join(given))
    if args.release and args.catalog:
        raise ValueError("with --browser-events give --catalog or --release, not both")
    config = open_browser_configuration(args.browser_events, catalog=args.catalog, release=args.release)
    rows, account = read_rows(args.manifest, args.role or DEFAULT_ROLE)
    for warning in manifest_warnings(config.run, account.manifests):
        _note(f"warning: {warning}")
    scored = run_browser_hover(rows, config.run, config.catalog.names(), account, config.catalog.identities())
    results = scoring.hover_results(scored, args.seed, args.bootstrap)
    result = report.build_result("hover", {"configuration": config}, account, results, seed=args.seed,
                                 replicates=args.bootstrap)
    print(report.to_markdown(result), end="")
    if args.out:
        for path in report.write_result(args.out, result, {"events": [item.to_json() for item in scored]}):
            _note(f"wrote {path}")
    unusable = sum(account.unusable.values())
    _note(f"{account.used} row(s) scored from the browser's outcomes, {unusable} not scored"
          + (" (listed in the report)" if unusable else ""))
    return 0


def _run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from cardvision.core.catalog import IDENTITY_NAME
    from cardvision.data.extract import FrameStore
    from cardvision.evaluate import crops, detector, report, scoring
    from cardvision.evaluate.config import open_configuration
    from cardvision.evaluate.rows import DEFAULT_ROLE, read_rows
    from cardvision.evaluate.scenes import read_scenes

    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be at least 1")
    if args.measure != "compare" and (args.other_release or any(getattr(args, f"other_{part}") for part in _OTHER_PARTS)):
        raise ValueError("--other-* arguments describe the candidate of compare and have no meaning for " + args.measure)
    if args.scenes:
        _check_scenes_arguments(args)
    elif not args.manifest:
        raise ValueError("give --manifest, one or more label manifests, or --scenes, a folder of generated scenes")
    for path in args.manifest or ():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path}: no such manifest file")
    if args.evidence and args.measure != "hover":
        raise ValueError(f"--evidence is written by hover, not by {args.measure}")
    if args.browser_events:
        if args.evidence:
            raise ValueError("--evidence is written while frames are analyzed, not from --browser-events")
        return _run_browser(args)
    if not args.scenes:
        if not args.frames:
            raise ValueError("give --frames, the folder of frames that cardvision fetch wrote")
        if not Path(args.frames).is_dir():
            raise FileNotFoundError(f"{args.frames}: no such folder of frames")

    base_spec = _base_spec(args)
    candidate_spec = _candidate_spec(args, base_spec) if args.measure == "compare" else None
    if args.scenes:
        # The scenes' labels are checked against the catalog being scored, so it is opened first.
        base = open_configuration(base_spec)
        rows, account, store = read_scenes(args.scenes, base.catalog)
    else:
        rows, account = read_rows(args.manifest, args.role or DEFAULT_ROLE)
        store = FrameStore(args.frames)
        base = open_configuration(base_spec)
    configurations = {"configuration": base}
    stage = base.pipeline.recognizer.geometric
    if stage is not None and stage.opencv_note():
        _note(f"warning: {stage.opencv_note()}")
    extra: dict[str, list[dict]] = {}
    seed, replicates = args.seed, args.bootstrap
    # Where different cards can share a name, the listings give each card's id beside its name.
    with_ids = base.catalog.identity != IDENTITY_NAME

    if args.measure == "hover":
        writer = None
        if args.evidence:
            from cardvision.evaluate.evidence import EvidenceWriter, enable_debug

            enable_debug(base)
            writer = EvidenceWriter(args.evidence, base)
        try:
            scored = scoring.run_hover(rows, store, base, account, observe=writer)
        finally:
            if writer is not None:
                writer.close()
                _note(f"wrote {writer.path} ({writer.rows} frame(s))")
        results = scoring.hover_results(scored, seed, replicates)
        extra["events"] = [item.to_json() for item in scored]
    elif args.measure == "crops":
        found = crops.run_crops(rows, store, base, account)
        results = crops.crop_results(found, seed, replicates)
        extra["crops"] = [item.to_json(with_ids) for item in found]
    elif args.measure == "detector":
        from cardvision.data.sources import load_source_config
        from cardvision.data.targets import TargetRules

        rules = TargetRules.for_sources(load_source_config(args.sources)) if args.sources else TargetRules()
        records = detector.run_detector(rows, store, base, account, rules=rules)
        results = detector.detector_results(records, base.spec.threshold, caps=_caps(args.fp_caps), seed=seed,
                                            replicates=replicates)
    else:
        candidate = open_configuration(candidate_spec)
        stage = candidate.pipeline.recognizer.geometric
        if stage is not None and stage.opencv_note():
            _note(f"warning: (candidate) {stage.opencv_note()}")
        configurations = {"base": base, "candidate": candidate}
        pairs = scoring.run_compare(rows, store, base, candidate, account)
        results = scoring.compare_results(pairs, seed, replicates, with_ids)
        extra["events"] = [{**before.to_json(), "candidate": {"outcome": after.outcome, "shown": after.shown,
                                                             "shown_id": after.shown_id,
                                                             "box": None if after.box is None else list(after.box)}}
                           for before, after in pairs]

    result = report.build_result(args.measure, configurations, account, results, seed=seed, replicates=replicates)
    print(report.to_markdown(result), end="")
    if args.out:
        for path in report.write_result(args.out, result, extra):
            _note(f"wrote {path}")
    unusable = sum(account.unusable.values())
    _note(f"{account.used} row(s) scored, {unusable} not scored" + (" (listed in the report)" if unusable else ""))
    return 0


def run(args: argparse.Namespace) -> int:
    """Run the subcommand. Returns the process exit code: 0, or 2 after a reported error."""
    from cardvision.bank.release import ReleaseError
    from cardvision.core.catalog import CatalogError
    from cardvision.data.schema import ManifestError

    try:
        return _run(args)
    except ReleaseError as error:
        _note("error: the release cannot be used: " + "; ".join(error.errors))
    except KeyError as error:
        _note(f"error: {error.args[0] if error.args else error}")
    except (ManifestError, CatalogError, FileNotFoundError, ValueError, ImportError) as error:
        _note(f"error: {error}")
    return 2
