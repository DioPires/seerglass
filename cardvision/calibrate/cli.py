# SPDX-License-Identifier: Apache-2.0
"""The ``cardvision calibrate`` subcommand: choose ``tau`` and ``delta`` on labeled frames.

::

    cardvision calibrate --manifest FILE [FILE ...] --frames DIR
        (--release R.json | --detector X --recognizer Y --bank B --catalog C [--accept A])
        [--role train] [--tau LIST] [--delta LIST] [--max-wrong RATE] [--out DIR]
        [--bootstrap N] [--seed S]

The inputs are those of ``cardvision evaluate hover``. Each frame is analyzed once, and every
pair of the grid is then scored on the same looks (``cardvision.calibrate.grid``). The Markdown
report is printed to standard output; progress and warnings go to standard error.

**The rows.** The default role is ``train``, not ``evaluation``: a rule chosen on the rows a
figure is reported on makes that figure optimistic. ``--role evaluation`` is refused unless
``--allow-evaluation-rows`` is also given, and the report then carries a warning.

**The recommendation.** There is no default limit. Without ``--max-wrong`` the report lists
the pairs worth choosing between and recommends none: what share of wrong names is acceptable
is the user's decision, and it is easier to make after seeing what the footage allows. With
``--max-wrong RATE`` one pair is recommended, and ``--out`` also writes it as ``accept.json``.

**Exit codes.** 0; 1 when ``--max-wrong`` was given and no pair is at or under it; 2 after a
reported error.
"""

from __future__ import annotations

import argparse
import sys

SUMMARY = "Choose the accept rule's tau and delta on labeled frames."
EVALUATION_ROLE = "evaluation"
PROGRESS_EVERY = 25


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the subcommand's arguments."""
    from cardvision.calibrate.grid import DEFAULT_DELTA, DEFAULT_TAU
    from cardvision.data.schema import ROLES
    from cardvision.evaluate.stats import DEFAULT_REPLICATES, DEFAULT_SEED

    parser.add_argument("--manifest", required=True, nargs="+", metavar="FILE", help="one or more label manifests")
    parser.add_argument("--frames", required=True, metavar="DIR", help="the folder of frames that cardvision fetch wrote")
    parser.add_argument("--release", metavar="R.json", help="a release document; it pins every file and setting")
    parser.add_argument("--detector", metavar="X.onnx", help="the detector model (without --release)")
    parser.add_argument("--recognizer", metavar="Y.onnx", help="the recognizer model (without --release)")
    parser.add_argument("--bank", metavar="B.json", help="the embedding bank built with that recognizer (without --release)")
    parser.add_argument("--catalog", metavar="C.json", help="the game's catalog (without --release)")
    parser.add_argument("--accept", metavar="A.json",
                        help="the accept rule to start from (without --release): its rotationRetry is kept, and "
                             "its tau and delta are added to the grid; the published model's rule when omitted")
    parser.add_argument("--role", default="train", choices=ROLES,
                        help="the role of the rows to calibrate on (default train). Never the rows you report "
                             "figures on")
    parser.add_argument("--allow-evaluation-rows", action="store_true",
                        help="go on with --role evaluation, although figures scored on those rows afterwards "
                             "are optimistic")
    parser.add_argument("--tau", default=DEFAULT_TAU, metavar="LIST",
                        help="the values of tau to try: numbers separated by commas, each a number or a range "
                             f"START:STOP:STEP that includes STOP (default {DEFAULT_TAU})")
    parser.add_argument("--delta", default=DEFAULT_DELTA, metavar="LIST",
                        help=f"the values of delta to try, written the same way (default {DEFAULT_DELTA})")
    parser.add_argument("--max-wrong", type=float, metavar="RATE",
                        help="recommend the pair that names the most faces right among those whose wrong of "
                             "shown is at or under this fraction (0.02 is 2 %%). There is no default: without "
                             "it the report lists the trade-off and recommends nothing")
    parser.add_argument("--out", metavar="DIR",
                        help="write the report, the JSON result and the grid as CSV here, and accept.json for "
                             "the recommended pair. The folder must not hold an accept.json yet")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_REPLICATES, metavar="N",
                        help=f"bootstrap replicates (default {DEFAULT_REPLICATES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="S",
                        help=f"seed of the bootstrap (default {DEFAULT_SEED})")


def _note(message: str) -> None:
    print(f"cardvision calibrate: {message}", file=sys.stderr, flush=True)


def _run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from cardvision.calibrate import grid, report
    from cardvision.data.extract import FrameStore
    from cardvision.evaluate.config import open_configuration, spec_from_arguments
    from cardvision.evaluate.events import rates
    from cardvision.evaluate.rows import read_rows
    from cardvision.evaluate.scoring import analyze_hover, hover_results

    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be at least 1")
    if args.max_wrong is not None and not 0 <= args.max_wrong <= 1:
        raise ValueError(f"--max-wrong is a fraction from 0 to 1 (0.02 is 2 %), got {args.max_wrong:g}")
    taus, deltas = grid.parse_values(args.tau, "--tau"), grid.parse_values(args.delta, "--delta")
    if args.role == EVALUATION_ROLE and not args.allow_evaluation_rows:
        raise ValueError(
            "--role evaluation names the rows that figures are reported on. A rule chosen on them fits them, so "
            "any figure scored on those rows afterwards is optimistic. Calibrate on --role train, or on a set of "
            "recordings kept for the purpose. To go on all the same, add --allow-evaluation-rows")
    for path in args.manifest:
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path}: no such manifest file")
    if not Path(args.frames).is_dir():
        raise FileNotFoundError(f"{args.frames}: no such folder of frames")
    if args.out:
        report.refuse_existing_accept(args.out)

    spec = spec_from_arguments(args.release, args.detector, args.recognizer, args.bank, args.catalog, args.accept)
    rules = grid.grid_rules(taus, deltas, spec.rule)
    rows, account = read_rows(args.manifest, args.role)
    config = open_configuration(spec)

    def progress(done: int) -> None:
        if done % PROGRESS_EVERY == 0:
            _note(f"{done} frame(s) analyzed")

    _note(f"analyzing the frames of {len(rows)} row(s) of role {args.role}, each once")
    analysis = analyze_hover(rows, FrameStore(args.frames), config, account, progress)
    unusable = sum(account.unusable.values())
    if not analysis.rows:
        raise ValueError(f"no row of role {args.role} could be scored: {account.read} row(s) read, {len(rows)} of "
                         f"that role, {unusable} without a usable frame. Check --role and --frames")
    _note(f"{account.used} frame(s) analyzed, {unusable} row(s) not scored"
          + (" (listed in the report)" if unusable else "") + f"; scoring {len(rules)} pairs on them")

    points = grid.score_grid(analysis, rules, args.seed, args.bootstrap)
    chosen = None if args.max_wrong is None else grid.recommend(points, args.max_wrong)
    chosen_results = None if chosen is None else hover_results(analysis.score(chosen.rule), args.seed, args.bootstrap)
    warnings = [report.EVALUATION_ROLE_WARNING] if args.role == EVALUATION_ROLE else []
    few = report.few_negatives_warning(grid.event_counts(points[0])["negatives"])
    if few:
        warnings.append(few)
    started = rates(analysis.score(spec.rule), args.seed, args.bootstrap)
    result = report.build_calibration(
        config, account, points, grid.trade_off(points), started, taus=taus, deltas=deltas, max_wrong=args.max_wrong,
        recommended=chosen, recommended_results=chosen_results, warnings=warnings, seed=args.seed,
        replicates=args.bootstrap)

    print(report.to_markdown(result), end="")
    for warning in warnings:
        _note(f"warning: {warning}")
    if args.out:
        for path in report.write_calibration(args.out, result):
            _note(f"wrote {path}")
    if args.max_wrong is None:
        _note("no pair recommended: no --max-wrong was given")
    elif chosen is None:
        _note(f"no pair recommended: none keeps wrong of shown at or under {args.max_wrong:g}")
        return 1
    else:
        _note(f"recommended: tau {chosen.rule.tau:g}, delta {chosen.rule.delta:g}"
              + ("" if args.out else "; pass --out DIR to have it written as accept.json"))
    return 0


def run(args: argparse.Namespace) -> int:
    """Run the subcommand. Returns the process exit code (see the module's description)."""
    from cardvision.bank.release import ReleaseError
    from cardvision.core.catalog import CatalogError
    from cardvision.data.schema import ManifestError

    try:
        return _run(args)
    except ReleaseError as error:
        _note("error: the release cannot be used: " + "; ".join(error.errors))
    except KeyError as error:
        _note(f"error: {error.args[0] if error.args else error}")
    except (ManifestError, CatalogError, FileNotFoundError, FileExistsError, ValueError) as error:
        _note(f"error: {error}")
    return 2
