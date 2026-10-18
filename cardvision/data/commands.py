# SPDX-License-Identifier: Apache-2.0
"""The label-manifest subcommands of the ``cardvision`` command line.

This module only describes the commands; the work is imported when a command runs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path


def _fail(command: str, messages: list[str] | str) -> int:
    lines = [messages] if isinstance(messages, str) else messages
    print(f"cardvision {command}: failed", file=sys.stderr)
    for line in lines:
        print(f"  - {line}", file=sys.stderr)
    return 1


# ------------------------------------------------------------------------ validate-manifest
def _validate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifests", nargs="+", type=Path, metavar="FILE",
                        help="a manifest (.jsonl), or a folder whose manifests are all checked; several may "
                             "be given and are checked together")
    parser.add_argument("--catalog", type=Path, metavar="C.json",
                        help="the game's catalog: every card id in the manifest must be in it")
    parser.add_argument("--sources", type=Path, metavar="S.json",
                        help="the game's sources file, which lists the source types it adds to the built-in ones")
    parser.add_argument("--contribution", action="store_true",
                        help='check the rows as offered labels: role "extra", trained_on false and no '
                             "reachability check, since those are set when labels are accepted, and no row "
                             'of a recording of your own (platform "local"), which nobody else can obtain')


def _validate(args: argparse.Namespace) -> int:
    from cardvision.core.catalog import CatalogError, load_catalog
    from cardvision.data.schema import SOURCE_TYPES
    from cardvision.data.sources import SourceConfigError, load_source_config
    from cardvision.data.validate import validate_manifests

    try:
        catalog_ids = load_catalog(args.catalog).ids() if args.catalog else None
        source_types = load_source_config(args.sources).source_types if args.sources else SOURCE_TYPES
        report = validate_manifests(args.manifests, catalog_ids=catalog_ids, source_types=source_types,
                                    contribution=args.contribution)
    except (CatalogError, SourceConfigError, OSError, ValueError) as error:
        return _fail("validate-manifest", str(error))
    if not report.ok:
        return _fail("validate-manifest", [str(problem) for problem in report.problems]
                     + [f"{len(report.problems)} problem(s) in {len(report.files)} file(s)"])
    checked = "card ids checked against the catalog" if catalog_ids is not None else "card ids not checked (no catalog given)"
    print(f"valid: {report.rows} row(s) in {len(report.files)} file(s); {checked}")
    return 0


# ----------------------------------------------------------------------------- assign-roles
def _share(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"a share lies between 0 and 1, got {text}")
    return value


def _assign_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path, metavar="MANIFEST", help="the rows whose recording groups need a role")
    parser.add_argument("--seed", type=int, required=True, help="the seed of the shuffle; write it down with the result")
    parser.add_argument("--evaluation-share", type=_share, required=True, metavar="SHARE",
                        help="the share of each source type's recording groups that goes to evaluation (0 to 1)")
    parser.add_argument("--share", action="append", default=[], metavar="SOURCE_TYPE=SHARE",
                        help="a different evaluation share for one source type; may be repeated")
    parser.add_argument("--published", action="append", default=[], type=Path, metavar="FILE",
                        help="a manifest whose groups already have roles: a group found there keeps its role, "
                             "and its groups count towards the shares; may be repeated")
    parser.add_argument("--out", type=Path, required=True, metavar="FILE", help="where to write the rows with their roles")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")


def _assign(args: argparse.Namespace) -> int:
    from cardvision.data.manifest import read_manifest, write_manifest
    from cardvision.data.roles import assign_manifest_roles
    from cardvision.data.schema import ManifestError

    try:
        shares = {}
        for entry in args.share:
            name, separator, value = entry.partition("=")
            if not separator:
                raise ValueError(f"{entry!r}: expected SOURCE_TYPE=SHARE")
            shares[name] = _share(value)
        if args.out.exists() and not args.overwrite:
            raise ValueError(f"{args.out} exists; pass --overwrite to replace it")
        rows = read_manifest(args.manifest)
        published = [row for path in args.published for row in read_manifest(path)]
        result = assign_manifest_roles(rows, args.seed, args.evaluation_share, shares=shares, published=published)
        # Nothing is created until the rows have their roles, so a refused run leaves no folder.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(args.out, result.rows)
    except (ManifestError, OSError, ValueError, argparse.ArgumentTypeError) as error:
        return _fail("assign-roles", str(error))
    print(f"seed {args.seed}; evaluation share {args.evaluation_share}"
          + "".join(f", {name} {value}" for name, value in sorted(shares.items())))
    for group, (role, how) in sorted(result.groups.items()):
        print(f"  {group}  {role}  ({how})")
    for stratum, tallies in sorted(result.tallies.items()):
        counts = ", ".join(f"{role} {count}" for role, count in sorted(tallies.items()))
        print(f"{stratum}: {counts}")
    print(f"wrote {len(result.rows)} row(s) to {args.out}")
    return 0


# ------------------------------------------------------------------------------------ fetch
# The exit code of fetch when the run stopped early and rows were not tried: the platform
# refused a download, or several downloads in a row failed. It differs from 1, which means
# that the run went to its end and some row was left without a frame.
EXIT_STOPPED = 3
EXIT_REFUSED = EXIT_STOPPED


def _fetch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path, metavar="MANIFEST", help="the manifest whose frames are wanted")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR",
                        help="where to write the frames (DIR/frames) and their index (DIR/index.jsonl); "
                             "it must lie outside the manifest's folder")
    parser.add_argument("--rows", metavar="ROWS", help="only these rows, by number from 1, for example 1-20,35")
    parser.add_argument("--role", action="append", default=[], choices=("train", "evaluation", "extra"),
                        help="only rows with this role; may be repeated")
    parser.add_argument("--video-file", action="append", default=[], metavar="ID=PATH",
                        help="a video file you already have, for the video with this id (the whole video, "
                             'from its start); may be repeated. A recording of your own (platform "local") is '
                             "given this way, under the name the manifest uses for it")
    parser.add_argument("--download", action="store_true",
                        help="download the videos that have no --video-file. This is your own act and is "
                             "subject to the platform's terms; needs the optional downloader. It never applies to "
                             'a recording of your own (platform "local")')
    parser.add_argument("--whole-video", action="store_true",
                        help="with --download, fetch each video whole rather than the few seconds around each frame")
    parser.add_argument("--max-whole-minutes", type=float, default=None, metavar="N",
                        help="with --download and a whole video (--whole-video, or when sections cannot be cut), "
                             "do not download a video the downloader reports as longer than N minutes; its rows "
                             "fail with the cause error. No limit unless given")
    parser.add_argument("--keep-video", type=Path, metavar="DIR",
                        help="a folder of kept video files. With --download, each downloaded file is kept in DIR "
                             "instead of being deleted, named <platform>-<video id>+<start seconds>.<ext> (no "
                             "+<start> for a whole video), with an index of them (DIR/video-files.json). With or "
                             "without --download, a row that a file already in DIR covers is read from it and "
                             "not downloaded. DIR must lie outside the manifest's folder")
    parser.add_argument("--pause", type=float, default=None, metavar="SECONDS",
                        help="with --download, the wait between two downloads (default 5)")
    parser.add_argument("--retries", type=int, default=None, metavar="N",
                        help="with --download, how many more times a part of a video is asked for when its "
                             "download fails (default 1). A refusal by the platform is never tried again, and "
                             "neither is a video the platform says is unavailable or restricted")
    parser.add_argument("--stop-after-failures", type=int, default=None, metavar="N",
                        help="with --download, stop the run when N parts in a row have failed, each after its "
                             "retries, for no stated reason: that may be a rate limit or a network problem "
                             "(default 3; 0 never stops)")
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg", help="the image format of the frames (default jpg)")
    parser.add_argument("--exact-seek", action="store_true",
                        help="read each file from its start, frame by frame, instead of seeking: slow, for files "
                             "that seek badly. The frames are the same either way")


def _fetch(args: argparse.Namespace) -> int:
    from cardvision.data import fetch as fetching
    from cardvision.data.extract import VideoSupportMissing
    from cardvision.data.manifest import iter_rows, read_manifest
    from cardvision.data.schema import LOCAL_PLATFORM, ManifestError

    try:
        rows = read_manifest(args.manifest)
        if args.rows:
            rows = [rows[index] for index in fetching.parse_row_selection(args.rows, len(rows))]
        if args.role:
            rows = list(iter_rows(rows, role=args.role))
        video_files = fetching.parse_video_files(args.video_file)
        folder = args.manifest.resolve().parent
        for given in (args.out, args.keep_video):
            if given is not None and (given.resolve() == folder or folder in given.resolve().parents):
                raise ValueError(f"{given} lies inside the manifest's folder, which holds text only; choose another folder")
        if args.keep_video is not None:
            fetching.read_kept_videos(args.keep_video)   # refuse a folder whose index is not ours before any download
        if args.keep_video is not None and not args.download and not fetching.has_kept_videos(args.keep_video):
            raise ValueError(f"--keep-video without --download reads the video files kept in {args.keep_video}, "
                             "which holds none; pass --download as well to fetch and keep them, or leave "
                             "--keep-video out")
        if args.retries is not None and args.retries < 0:
            raise ValueError("--retries is 0 or more")
        if args.stop_after_failures is not None and args.stop_after_failures < 0:
            raise ValueError("--stop-after-failures is 0 or more")
        if args.max_whole_minutes is not None and not args.max_whole_minutes > 0:
            raise ValueError("--max-whole-minutes is more than 0")
        if args.download and not fetching.downloader_installed():
            raise fetching.DownloadSupportMissing(
                'downloading needs an optional dependency: python -m pip install -e ".[download]" in the repository folder')
    except (ManifestError, OSError, ValueError, fetching.DownloadSupportMissing) as error:
        return _fail("fetch", str(error))
    if not rows:
        return _fail("fetch", "no row is selected")

    if args.download:
        print(fetching.DOWNLOAD_NOTICE, file=sys.stderr)
        if not args.whole_video and not fetching.sections_supported():
            print("ffmpeg was not found, so each video is downloaded whole.", file=sys.stderr)
    pause = fetching.DEFAULT_PAUSE_SECONDS if args.pause is None else max(0.0, args.pause)
    retries = fetching.DEFAULT_RETRIES if args.retries is None else args.retries
    stop_after = fetching.DEFAULT_STOP_AFTER_FAILURES if args.stop_after_failures is None else args.stop_after_failures
    try:
        summary = fetching.fetch_frames(
            rows, args.out, video_files=video_files, download=args.download, whole_video=args.whole_video,
            keep_video=args.keep_video, image_format=args.format, exact_seek=args.exact_seek, pause=pause,
            retries=retries, stop_after_failures=stop_after, max_whole_minutes=args.max_whole_minutes,
            say=lambda line: print(line, file=sys.stderr),
        )
    except (VideoSupportMissing, fetching.DownloadSupportMissing, fetching.CertificatesMissing, OSError) as error:
        return _fail("fetch", str(error))

    print(f"{summary.count('ok')} frame(s) written, {summary.skipped} already present, "
          f"{summary.count('unusable')} unusable, {summary.count('failed')} failed, "
          f"{summary.count('no-video')} without a video; index: {args.out / 'index.jsonl'}")
    if args.keep_video is not None:
        used = f"; {len(summary.kept_used)} already there were used" if summary.kept_used else ""
        print(f"{len(summary.kept)} new video file(s) kept in {args.keep_video}{used}; "
              f"index: {args.keep_video / fetching.KEPT_INDEX_NAME}")
    other_sizes = summary.other_sizes()
    if other_sizes:
        print(f"{sum(other_sizes.values())} of the frame(s) written have another size than their row records "
              "(other_size in the index). The shape is the same, so the labels fit:")
        for (width, height, row_width, row_height), count in sorted(other_sizes.items()):
            print(f"  {count} frame(s) of {width} x {height}, labeled at {row_width} x {row_height}")
    for entry in summary.entries:
        if entry.status in ("unusable", "failed"):
            reason = entry.reason
            if reason == "fingerprint":
                reason = (f"no frame within a second of t looks like the labeled one "
                          f"(nearest differs by {entry.distance} bits)")
            print(f"  {entry.status}: {entry.platform} {entry.video_id} t={entry.t}: {reason}", file=sys.stderr)
    without = [entry for entry in summary.entries if entry.status == "no-video"]
    if any(entry.platform != LOCAL_PLATFORM for entry in without):
        print("Rows without a video: give the file with --video-file ID=PATH, or pass --download.", file=sys.stderr)
    if any(entry.platform == LOCAL_PLATFORM for entry in without):
        print('Rows of a recording of your own (platform "local") without a video: give the file with '
              "--video-file ID=PATH. --download does not apply to them.", file=sys.stderr)
    if summary.refused is not None:
        print("The platform refused a download, as a rate limit or as a check for automated access:\n"
              f"  {summary.refused}\n"
              f"The run stopped at once. {summary.remaining} row(s) were not tried and have no line in the index.\n"
              "Wait before you run the command again: more requests now make a refusal last longer. "
              "Rows that have a frame are skipped then.", file=sys.stderr)
        return EXIT_REFUSED
    if summary.stopped is not None:
        print(f"{summary.stopped} downloads in a row failed. This may be a rate limit or a network problem.\n"
              f"The run stopped. {summary.remaining} row(s) were not tried and have no line in the index.\n"
              "Check the connection, and wait before you run the command again. Rows that have a frame are "
              "skipped then. --stop-after-failures N sets how many failures stop a run (0: never).", file=sys.stderr)
        return EXIT_STOPPED
    return 0 if summary.complete else 1


# --------------------------------------------------------------------------- record-reachable
def _reachable_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path, metavar="MANIFEST",
                        help="the manifest whose rows get reachable and reachable_checked; it is rewritten in place")
    parser.add_argument("--index", type=Path, required=True, metavar="DIR/index.jsonl",
                        help="the index a cardvision fetch --download run wrote (the file, or its folder): "
                             "that run is the check")
    parser.add_argument("--date", metavar="YYYY-MM-DD",
                        help="the date of the check, for every row. Left out, each row gets the date on which its "
                             "line of the index was written (in UTC). An index written by an older version of "
                             "cardvision fetch records no time, and then the date of that run must be given")


def _reachable(args: argparse.Namespace) -> int:
    from cardvision.data import reachable
    from cardvision.data.manifest import read_manifest, write_manifest
    from cardvision.data.schema import ManifestError

    try:
        date = reachable.check_date(args.date) if args.date else None
        rows = read_manifest(args.manifest)
        index = reachable.read_index_file(args.index)
        updated, summary = reachable.record_reachable(rows, index, date)
        if summary.unmatched == len(index):
            raise ValueError(f"{args.index} lists no row of {args.manifest}; nothing was changed")
        write_manifest(args.manifest, updated)
    except (ManifestError, OSError, ValueError) as error:
        return _fail("record-reachable", str(error))
    for row in summary.lost:
        print(f"not reachable: {row.platform} {row.video_id} t={row.t}: {row.reason}")
    for row in summary.failed_unchanged:
        print(f"left as it was: {row.platform} {row.video_id} t={row.t}: the download failed "
              f"({row.cause or 'no cause recorded'}), which does not show that the video is gone: {row.reason}")
    checked = ""
    if summary.dates:
        checked = f", checked {summary.dates[0]}" + (f" to {summary.dates[-1]}" if len(summary.dates) > 1 else "")
    failed = f", {len(summary.failed_unchanged)} of them after a failed download" if summary.failed_unchanged else ""
    print(f"{summary.reachable} row(s) reachable, {summary.not_reachable} not reachable{checked}; "
          f"{summary.unchanged} row(s) left as they were{failed}; wrote {args.manifest}")
    if summary.unmatched:
        print(f"{summary.unmatched} line(s) of the index name no row of this manifest.", file=sys.stderr)
    return 0


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None], Callable[[argparse.Namespace], int]]] = {
    "validate-manifest": ("Check label manifests: format, ranges, duplicates, groups, card ids, no images.",
                          _validate_arguments, _validate),
    "assign-roles": ("Give new recording groups a role by the seeded rule.",
                     _assign_arguments, _assign),
    "fetch": ("Obtain the frames of a manifest's rows from video files, or by downloading if asked.",
              _fetch_arguments, _fetch),
    "record-reachable": ("Set a manifest's reachable fields from the index of a fetch run that downloaded.",
                         _reachable_arguments, _reachable),
}
