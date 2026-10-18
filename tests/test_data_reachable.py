# SPDX-License-Identifier: Apache-2.0
"""``cardvision record-reachable``: a manifest's reachable fields set from the index of a fetch
run, on evidence about the video only, and nothing else in the file changed. Nothing here
touches the network."""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path

import pytest

from cardvision import cli
from cardvision.data import reachable
from cardvision.data.extract import IndexEntry, append_index
from cardvision.data.manifest import read_manifest, row_line, write_manifest
from cardvision.data.schema import Row, parse_row
from support_data import OTHER_VIDEO_ID, TWITCH_ID, VIDEO_ID, box_doc, row_doc, write_lines

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-01-31"
WRITTEN = "2026-01-31T21:15:00Z"      # when the lines of the index were written
EARLIER = ("2025-12-01", True)        # a check recorded before, on rows the run must leave alone


def rows_under_test() -> list[dict]:
    """Rows of every kind the command meets, with labels and attributes that must not change."""
    marked = {"attributes": {"frame_dhash": "00ff00ff00ff00ff", "camera": "overhead"}, "prefill_offered": True}
    twitch = {"platform": "twitch", "video_id": TWITCH_ID, "group": "g-test-03"}
    checked = {"reachable": EARLIER[1], "reachable_checked": EARLIER[0]}
    return [
        row_doc(t=1.017, **marked),                                                           # found
        row_doc(t=2.5, boxes=[box_doc(), box_doc(quad=[[0.5, 0.5], [0.6, 0.5], [0.6, 0.75], [0.5, 0.75]], card_id="SET-002")]),
        row_doc(t=3.0, boxes=[]),                                                             # fingerprint not matched
        row_doc(t=4.0, video_id=OTHER_VIDEO_ID, group="g-test-02"),                           # the platform says it is gone
        row_doc(t=5.0, video_id=OTHER_VIDEO_ID, group="g-test-02"),                           # another shape
        row_doc(t=6.0, **twitch),                                                             # no video given
        row_doc(t=7.0, **twitch),                                                             # not in the index
        row_doc(t=8.0, **twitch, **checked),                                                  # from a file the user gave
        row_doc(t=9.0, **twitch, reachable=False, reachable_checked="2025-12-01"),            # failed first, found on a second try
        row_doc(t=10.0, platform="local", video_id="kitchen-table", group="g-own-01"),        # a recording of one's own
        row_doc(t=11.0, **twitch, **checked),                                                 # the network, or the cutting program
        row_doc(t=12.0, **twitch, **checked),                                                 # an age gate
        row_doc(t=13.0, **twitch),                                                            # a failed line with no cause
        row_doc(t=14.0, **twitch, **checked),                                                 # from a kept file
        row_doc(t=15.0, **twitch, **checked),                                                 # the file came and could not be read
    ]


def entry(doc: dict, status: str, source: str | None = "download", written: str | None = WRITTEN, **details) -> IndexEntry:
    return IndexEntry(platform=doc["platform"], video_id=doc["video_id"], t=float(doc["t"]), status=status, source=source,
                      written=written, **details)


SHAPE = "the frame is 1440 x 1080 (ratio 1.3333) but the labels were made on 1920 x 1080"
GONE = "youtube EXAMPLE0002: ERROR: [youtube] EXAMPLE0002: Video unavailable. This video is private"
OFFLINE = "twitch 0000000001: ERROR: ffmpeg exited with code 8"
GATED = "twitch 0000000001: ERROR: [twitch:vod] 0000000001: You must be logged into an account that has access to this subscriber-only content"


def index_lines(docs: list[dict], written: str | None = WRITTEN) -> list[IndexEntry]:
    def line(doc: dict, status: str, **details) -> IndexEntry:
        return entry(doc, status, written=written, **details)

    return [
        line(docs[0], "ok", file="frames/a.jpg", distance=0, offset=-0.016),
        line(docs[1], "ok", file="frames/b.jpg"),
        line(docs[2], "unusable", reason="fingerprint", distance=31),
        line(docs[3], "failed", reason=GONE, cause="unavailable"),
        line(docs[4], "unusable", reason=SHAPE),
        line(docs[5], "no-video", source=None, reason="no video file was given for this video and --download was not asked for"),
        line(docs[7], "failed", source="local", reason="the video ends before t=8.0"),
        line(docs[8], "failed", reason=OFFLINE, cause="error"),
        line(docs[8], "ok", file="frames/c.jpg"),
        line(docs[9], "ok", source="local", file="frames/d.jpg"),
        line(docs[10], "failed", reason=OFFLINE, cause="error", attempts=2),
        line(docs[11], "failed", reason=GATED, cause="restricted"),
        line(docs[12], "failed", reason="twitch 0000000001: not available"),
        line(docs[13], "ok", source="kept", file="frames/e.jpg"),
        line(docs[14], "failed", reason="the video ends before t=15.0", cause="error"),
        IndexEntry(platform="youtube", video_id="ELSEWHERE01", t=1.0, status="ok", source="download", file="frames/f.jpg",
                   written=written),
    ]


def write_index(folder: Path, docs: list[dict], written: str | None = WRITTEN) -> Path:
    for line in index_lines(docs, written):
        append_index(folder, line)
    return folder / "index.jsonl"


def expected(date: str = DATE) -> list[tuple]:
    kept = (EARLIER[1], EARLIER[0])
    return [(True, date), (True, date), (False, date), (False, date), (False, date), (None, None), (None, None),
            kept, (True, date), (None, None), kept, kept, (None, None), kept, kept]


EXPECTED = expected()
LOST = [("youtube", VIDEO_ID, 3.0), ("youtube", OTHER_VIDEO_ID, 4.0), ("youtube", OTHER_VIDEO_ID, 5.0)]
FAILED_UNCHANGED = [("twitch", TWITCH_ID, 11.0, "error"), ("twitch", TWITCH_ID, 12.0, "restricted"), ("twitch", TWITCH_ID, 13.0, None),
                    ("twitch", TWITCH_ID, 15.0, "error")]


def test_each_row_is_judged_by_its_last_line_in_the_index(tmp_path):
    docs = rows_under_test()
    rows = [parse_row(doc) for doc in docs]
    index = reachable.read_index_file(write_index(tmp_path / "frames", docs))
    updated, summary = reachable.record_reachable(rows, index)
    assert [(row.reachable, row.reachable_checked) for row in updated] == EXPECTED
    assert (summary.reachable, summary.not_reachable, summary.unchanged, summary.unmatched) == (3, 3, 9, 1)
    assert [(row.platform, row.video_id, row.t) for row in summary.lost] == LOST and summary.dates == (DATE,)
    assert [(row.platform, row.video_id, row.t, row.cause) for row in summary.failed_unchanged] == FAILED_UNCHANGED
    assert summary.lost[0].reason == "no frame within a second of t looks like the labeled one (nearest differs by 31 bits)"
    assert (summary.lost[1].reason, summary.lost[2].reason) == (GONE, SHAPE)
    # The index is read from its file or from its folder, and the rows are otherwise the same objects' values.
    assert reachable.read_index_file(tmp_path / "frames") == index
    for before, after in zip(rows, updated):
        assert Row.from_json({**after.to_json(), "reachable": before.reachable, "reachable_checked": before.reachable_checked}) == before


def test_what_the_index_says_about_a_row():
    row = parse_row(row_doc())
    own = parse_row(row_doc(platform="local", video_id="kitchen-table"))
    doc = row_doc()
    assert reachable.reachable_verdict(row, entry(doc, "ok")) is True
    assert reachable.reachable_verdict(row, entry(doc, "unusable", reason="fingerprint")) is False
    assert reachable.reachable_verdict(row, entry(doc, "unusable", reason=SHAPE)) is False
    assert reachable.reachable_verdict(row, entry(doc, "failed", cause="unavailable")) is False
    for silent in (None, entry(doc, "no-video", source=None), entry(doc, "ok", source="local"), entry(doc, "failed", source="local"),
                   entry(doc, "ok", source="kept"), entry(doc, "unusable", source="kept"), entry(doc, "some-later-status")):
        assert reachable.reachable_verdict(row, silent) is None
    # A recording of one's own is on no platform: no index line gives it a check.
    assert reachable.reachable_verdict(own, entry(row_doc(platform="local", video_id="kitchen-table"), "ok")) is None


@pytest.mark.parametrize("cause", ["error", "restricted", "refused", None, "some-later-cause"])
def test_a_download_that_failed_for_another_reason_proves_nothing(cause):
    """Being offline, the cutting program failing, a timeout, an age gate: none of them says
    that the video is gone, so none of them makes a row unreachable. Neither does a failed line
    with no cause, which is what an index written by an older version holds."""
    doc = row_doc()
    assert reachable.reachable_verdict(parse_row(doc), entry(doc, "failed", cause=cause, reason="ffmpeg exited with code 8")) is None


def test_a_failed_download_leaves_an_earlier_check_as_it_was():
    """The failure reported was this: a run made offline turned every row to false. Such a run
    now changes nothing."""
    docs = [row_doc(t=float(t), reachable=True, reachable_checked="2025-12-01") for t in range(1, 6)]
    rows = [parse_row(doc) for doc in docs]
    offline = "youtube EXAMPLE0001: ERROR: [youtube] EXAMPLE0001: Unable to download API page: [Errno 8] nodename nor servname provided"
    index = {row.key: entry(doc, "failed", reason=offline, cause="error", attempts=2) for row, doc in zip(rows, docs)}
    updated, summary = reachable.record_reachable(rows, index)
    assert updated == rows and (summary.reachable, summary.not_reachable, summary.unchanged) == (0, 0, 5)
    assert len(summary.failed_unchanged) == 5 and summary.dates == ()


def test_the_command_changes_the_two_fields_and_no_other_byte(tmp_path, capsys):
    """Every line of the rewritten file is the old line with the two reachability values
    replaced, and a row the index says nothing about keeps its line byte for byte."""
    docs = rows_under_test()
    path = tmp_path / "labels" / "m.jsonl"
    path.parent.mkdir()
    # The canonical row form: the form write_manifest and the labeling tool write.
    write_manifest(path, [parse_row(doc) for doc in docs])
    before = path.read_bytes().split(b"\n")
    index = write_index(tmp_path / "frames", docs)
    assert cli.main(["record-reachable", str(path), "--index", str(index)]) == 0
    captured = capsys.readouterr()
    failed = "the download failed ({}), which does not show that the video is gone: {}"
    assert captured.out.splitlines() == [
        "not reachable: youtube EXAMPLE0001 t=3.0: no frame within a second of t looks like the labeled one (nearest differs by 31 bits)",
        f"not reachable: youtube EXAMPLE0002 t=4.0: {GONE}",
        f"not reachable: youtube EXAMPLE0002 t=5.0: {SHAPE}",
        "left as it was: twitch 0000000001 t=11.0: " + failed.format("error", OFFLINE),
        "left as it was: twitch 0000000001 t=12.0: " + failed.format("restricted", GATED),
        "left as it was: twitch 0000000001 t=13.0: " + failed.format("no cause recorded", "twitch 0000000001: not available"),
        "left as it was: twitch 0000000001 t=15.0: " + failed.format("error", "the video ends before t=15.0"),
        f"3 row(s) reachable, 3 not reachable, checked {DATE}; 9 row(s) left as they were, 4 of them after a failed download; "
        f"wrote {path}",
    ]
    assert captured.err == "1 line(s) of the index name no row of this manifest.\n"
    after = path.read_bytes().split(b"\n")
    assert len(after) == len(before) and after[-1] == before[-1] == b""
    fields = re.compile(rb'"reachable":(true|false|null),"reachable_checked":("[0-9-]{10}"|null)')
    for old, new, (flag, date) in zip(before[:-1], after[:-1], EXPECTED):
        assert len(fields.findall(old)) == len(fields.findall(new)) == 1
        wanted = b'"reachable":%s,"reachable_checked":%s' % (json.dumps(flag).encode(), json.dumps(date).encode())
        assert fields.search(new).group(0) == wanted
        assert fields.sub(b"<>", new) == fields.sub(b"<>", old)          # every other byte of the row is the same
    assert [after[i] == before[i] for i in range(len(docs))] == [False] * 5 + [True, True, True, False] + [True] * 6
    assert [(row.reachable, row.reachable_checked) for row in read_manifest(path)] == EXPECTED
    # The file is still a valid manifest, and running the command again changes nothing.
    assert cli.main(["validate-manifest", str(path)]) == 0
    assert cli.main(["record-reachable", str(path), "--index", str(index.parent)]) == 0
    assert path.read_bytes().split(b"\n") == after


def test_a_manifest_in_another_form_is_put_in_the_canonical_form(tmp_path, capsys):
    """Spaces after separators and fields in another order are not kept: the file is rewritten
    in the canonical row form, with the same rows."""
    docs = rows_under_test()
    path = write_lines(tmp_path / "labels" / "m.jsonl", docs)          # json.dumps defaults: a space after each separator
    index = write_index(tmp_path / "frames", docs)
    assert cli.main(["record-reachable", str(path), "--index", str(index), "--date", DATE]) == 0
    capsys.readouterr()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [row_line(row) for row in read_manifest(path)]
    assert [(row.reachable, row.reachable_checked) for row in read_manifest(path)] == EXPECTED


def test_the_date_is_the_date_of_each_line_of_the_index(tmp_path, capsys):
    """A row gets the UTC date on which its own line was written, whenever the command is run.
    A run that went past midnight records two dates."""
    docs = rows_under_test()[:3]
    path = write_lines(tmp_path / "labels" / "m.jsonl", docs)
    folder = tmp_path / "frames"
    append_index(folder, entry(docs[0], "ok", written="2025-06-30T23:59:59Z", file="frames/a.jpg"))
    append_index(folder, entry(docs[1], "ok", written="2025-07-01T00:00:01Z", file="frames/b.jpg"))
    append_index(folder, entry(docs[2], "unusable", written="2025-07-01T00:05:00Z", reason="fingerprint", distance=20))
    assert cli.main(["record-reachable", str(path), "--index", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "2 row(s) reachable, 1 not reachable, checked 2025-06-30 to 2025-07-01; 0 row(s) left as they were; wrote" in out
    assert [(row.reachable, row.reachable_checked) for row in read_manifest(path)] == [
        (True, "2025-06-30"), (True, "2025-07-01"), (False, "2025-07-01")]


def test_a_date_given_is_recorded_for_every_row_and_must_be_a_date(tmp_path, capsys):
    docs = rows_under_test()
    path = write_lines(tmp_path / "labels" / "m.jsonl", docs)
    index = write_index(tmp_path / "frames", docs)
    assert cli.main(["record-reachable", str(path), "--index", str(index), "--date", "2026-02-02"]) == 0
    assert "checked 2026-02-02;" in capsys.readouterr().out
    assert [(row.reachable, row.reachable_checked) for row in read_manifest(path)] == expected("2026-02-02")
    for bad in ("2026-02-30", "31-01-2026", "2026-1-5", "yesterday"):
        before = path.read_bytes()
        assert cli.main(["record-reachable", str(path), "--index", str(index), "--date", bad]) == 1
        assert "a date is written YYYY-MM-DD" in capsys.readouterr().err and path.read_bytes() == before


def test_an_index_that_records_no_time_needs_the_date_of_its_run(tmp_path, capsys):
    """An index written by an older version of the fetch command has no time on its lines. The
    command does not guess the date: today could be months after the run."""
    docs = rows_under_test()
    path = write_lines(tmp_path / "labels" / "m.jsonl", docs)
    before = path.read_bytes()
    index = write_index(tmp_path / "frames", docs, written=None)
    assert "written" not in index.read_text(encoding="utf-8")
    assert cli.main(["record-reachable", str(path), "--index", str(index)]) == 1
    err = capsys.readouterr().err
    assert ("6 line(s) of the index that set a row record no time: the index was written by an older version of cardvision "
            "fetch. Pass --date YYYY-MM-DD with the date of that fetch run") in err
    assert path.read_bytes() == before
    assert cli.main(["record-reachable", str(path), "--index", str(index), "--date", DATE]) == 0
    assert f"checked {DATE};" in capsys.readouterr().out
    assert [(row.reachable, row.reachable_checked) for row in read_manifest(path)] == EXPECTED
    # Lines that set nothing need no date: a run without a download, for one.
    plain = tmp_path / "plain"
    append_index(plain, entry(docs[5], "no-video", source=None, written=None))
    assert cli.main(["record-reachable", str(path), "--index", str(plain)]) == 0
    assert "0 row(s) reachable, 0 not reachable; 15 row(s) left as they were; wrote" in capsys.readouterr().out


def test_the_command_refuses_what_it_cannot_use_and_leaves_the_file_alone(tmp_path, capsys):
    docs = rows_under_test()
    path = write_lines(tmp_path / "labels" / "m.jsonl", docs)
    before = path.read_bytes()
    assert cli.main(["record-reachable", str(path), "--index", str(tmp_path / "absent" / "index.jsonl")]) == 1
    assert "no such index file" in capsys.readouterr().err
    broken = tmp_path / "broken" / "index.jsonl"
    broken.parent.mkdir()
    broken.write_text('{"platform": "youtube"}\n', encoding="utf-8")
    assert cli.main(["record-reachable", str(path), "--index", str(broken)]) == 1
    assert "line 1 is not a line of a frame folder's index" in capsys.readouterr().err
    undated = tmp_path / "undated"
    append_index(undated, entry(docs[0], "ok", written="31 January 2026"))
    assert cli.main(["record-reachable", str(path), "--index", str(undated)]) == 1
    assert "line 1 is not a line of a frame folder's index" in capsys.readouterr().err
    other = tmp_path / "other"
    append_index(other, IndexEntry(platform="youtube", video_id="ELSEWHERE01", t=1.0, status="ok", source="download", written=WRITTEN))
    assert cli.main(["record-reachable", str(path), "--index", str(other / "index.jsonl")]) == 1
    assert "lists no row of" in capsys.readouterr().err
    assert cli.main(["record-reachable", str(tmp_path / "absent.jsonl"), "--index", str(other)]) == 1
    capsys.readouterr()
    assert path.read_bytes() == before


def test_recording_opens_no_network_connection(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("record-reachable tried to open a network connection")

    docs = rows_under_test()
    path = write_lines(tmp_path / "labels" / "m.jsonl", docs)
    index = write_index(tmp_path / "frames", docs)
    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refuse)
    assert cli.main(["record-reachable", str(path), "--index", str(index), "--date", DATE]) == 0


def test_a_fetch_run_and_the_record_made_from_it(tmp_path, capsys):
    """The two commands together, with a stand-in downloader: a video the platform says is
    private turns its row to false, and a download that fails for no stated reason leaves its
    row alone."""
    import datetime

    from cardvision.data import fetch

    docs = [row_doc(t=2.017, video_id=VIDEO_ID, reachable=True, reachable_checked="2025-12-01"),
            row_doc(t=3.017, video_id=OTHER_VIDEO_ID, group="g-test-02", reachable=True, reachable_checked="2025-12-01")]
    path = tmp_path / "labels" / "m.jsonl"
    path.parent.mkdir()
    write_manifest(path, [parse_row(doc) for doc in docs])
    words = {VIDEO_ID: "ERROR: [youtube] EXAMPLE0001: Private video. Sign in if you've been granted access to this video",
             OTHER_VIDEO_ID: "ERROR: unable to download video data: <urlopen error [Errno 8] nodename nor servname provided>"}

    def downloader(platform, video_id, start, end, folder):
        raise fetch.DownloadFailed(f"{platform} {video_id}: {words[video_id]}")

    summary = fetch.fetch_frames(read_manifest(path), tmp_path / "frames", download=True, downloader=downloader, pause=0,
                                 say=lambda line: None, now=lambda: datetime.datetime(2026, 3, 4, 12, tzinfo=datetime.timezone.utc))
    assert [(line.status, line.cause, line.written) for line in summary.entries] == [
        ("failed", "unavailable", "2026-03-04T12:00:00Z"), ("failed", "error", "2026-03-04T12:00:00Z")]
    assert cli.main(["record-reachable", str(path), "--index", str(tmp_path / "frames")]) == 0
    out = capsys.readouterr().out
    assert "not reachable: youtube EXAMPLE0001 t=2.017: youtube EXAMPLE0001: ERROR: [youtube] EXAMPLE0001: Private video." in out
    assert "left as it was: youtube EXAMPLE0002 t=3.017: the download failed (error), which does not show that the video is gone" in out
    assert "0 row(s) reachable, 1 not reachable, checked 2026-03-04; 1 row(s) left as they were, 1 of them after a failed download" in out
    assert [(row.reachable, row.reachable_checked) for row in read_manifest(path)] == [(False, "2026-03-04"), (True, "2025-12-01")]


def test_the_later_parts_of_a_video_that_is_gone_are_not_asked_for_and_are_judged_alike(tmp_path, capsys, monkeypatch):
    """One video, three parts, the platform says it is private: one request is made, the next
    video is still asked for, and ``record-reachable`` sets all three rows of the first video to
    false and leaves the other video's row alone."""
    import datetime

    from cardvision.data import fetch

    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    times = (2.017, 200.017, 900.017)
    docs = [row_doc(t=t, video_id=VIDEO_ID, reachable=True, reachable_checked="2025-12-01") for t in times]
    docs.append(row_doc(t=3.017, video_id=OTHER_VIDEO_ID, group="g-test-02", reachable=True, reachable_checked="2025-12-01"))
    path = tmp_path / "labels" / "m.jsonl"
    path.parent.mkdir()
    write_manifest(path, [parse_row(doc) for doc in docs])
    asked: list[tuple[str, float | None]] = []

    def downloader(platform, video_id, start, end, folder):
        asked.append((video_id, start))
        if video_id == VIDEO_ID:
            raise fetch.DownloadFailed(f"{platform} {video_id}: ERROR: [youtube] {video_id}: Private video. Sign in if you've "
                                       "been granted access to this video")
        raise fetch.DownloadFailed(f"{platform} {video_id}: ERROR: ffmpeg exited with code 8")

    summary = fetch.fetch_frames(read_manifest(path), tmp_path / "frames", download=True, downloader=downloader, pause=0, retries=0,
                                 say=lambda line: None, now=lambda: datetime.datetime(2026, 3, 4, 12, tzinfo=datetime.timezone.utc))
    assert asked == [(VIDEO_ID, 0.0), (OTHER_VIDEO_ID, 0.0)]
    assert [(line.status, line.cause, line.attempts) for line in summary.entries] == [
        ("failed", "unavailable", 1), ("failed", "unavailable", 0), ("failed", "unavailable", 0), ("failed", "error", 1)]
    assert summary.entries[1].reason == "youtube EXAMPLE0001: not asked for; an earlier part of this video was answered unavailable"
    assert cli.main(["record-reachable", str(path), "--index", str(tmp_path / "frames")]) == 0
    assert "0 row(s) reachable, 3 not reachable, checked 2026-03-04; 1 row(s) left as they were" in capsys.readouterr().out
    assert [(row.reachable, row.reachable_checked) for row in read_manifest(path)] == [
        (False, "2026-03-04")] * 3 + [(True, "2025-12-01")]


def test_the_published_manifests_are_in_the_canonical_row_form():
    """``record-reachable`` rewrites a manifest in the canonical row form. The manifests under
    ``data/`` are in that form already, so the command changes their two fields and nothing else."""
    for path in sorted((ROOT / "data").glob("*.manifest.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines == [row_line(row) for row in read_manifest(path)], path.name


def test_the_published_evaluation_rows_record_their_check():
    """Every row of the public evaluation set says whether its video was found, and when."""
    rows = read_manifest(ROOT / "data" / "riftbound-evaluation.manifest.jsonl")
    assert rows and all(row.reachable is not None and row.reachable_checked for row in rows)
