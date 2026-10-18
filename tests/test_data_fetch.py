# SPDX-License-Identifier: Apache-2.0
"""``cardvision fetch``: frames from a generated local video, resuming, and the download path
with a stand-in downloader. Nothing here touches the network, and the real downloader is never
imported. Tests that read video are skipped when OpenCV is not installed."""

from __future__ import annotations

import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cardvision import cli
from cardvision.data import extract, fetch, manifest
from cardvision.data.schema import Row
from support_data import OTHER_VIDEO_ID, TWITCH_ID, VIDEO_ID, counter_frame, read_counter, write_counter_video

RATE = 30.0


def make_row(index: int, video_id: str = VIDEO_ID, width: int = 1920, height: int = 1080, platform: str = "youtube",
             role: str = "extra") -> Row:
    """A row whose frame is number ``index`` of a 30 frames-a-second video."""
    return Row(platform=platform, video_id=video_id, t=extract.mid_frame_time(index, RATE), frame_width=width,
               frame_height=height, source_type="physical", group=f"g-{video_id.lower()}", role=role, trained_on=False,
               completeness="exhaustive-boxes", credit="Example Labeler", boxes=())


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    pytest.importorskip("cv2")
    return write_counter_video(tmp_path_factory.mktemp("video") / "counter.mp4", rate=RATE, frames=300)


def shown(path: Path) -> int:
    with Image.open(path) as image:
        return read_counter(np.asarray(image.convert("RGB")))


def quiet(_: str) -> None:
    pass


# ---------------------------------------------------------------------------- small pieces
def test_wanted_times_are_grouped_into_stretches():
    assert fetch.plan_sections([10.5, 20.25, 500.0]) == [(7.0, 24.0, [10.5, 20.25]), (497.0, 503.0, [500.0])]
    assert fetch.plan_sections([1.0]) == [(0.0, 4.0, [1.0])]
    assert fetch.plan_sections([70.0, 10.0, 10.0]) == [(7.0, 73.0, [10.0, 70.0])]
    assert fetch.plan_sections([10.0, 71.0]) == [(7.0, 13.0, [10.0]), (68.0, 74.0, [71.0])]
    assert fetch.plan_sections([]) == []


def test_row_selections_and_video_file_arguments(tmp_path):
    assert fetch.parse_row_selection("1-3,7", 10) == [0, 1, 2, 6]
    assert fetch.parse_row_selection("10, 2-2", 10) == [1, 9]
    for bad in ("0", "3-1", "11", "a", "1,,2", "1-"):
        with pytest.raises(ValueError):
            fetch.parse_row_selection(bad, 10)
    file = tmp_path / "v.mp4"
    file.write_bytes(b"")
    assert fetch.parse_video_files([f"{VIDEO_ID}={file}", f"twitch:{TWITCH_ID}={file}"]) == {
        VIDEO_ID: file, f"twitch:{TWITCH_ID}": file}
    for bad in (VIDEO_ID, f"={file}", f"{VIDEO_ID}=", f"{VIDEO_ID}={tmp_path / 'absent.mp4'}"):
        with pytest.raises(ValueError):
            fetch.parse_video_files([bad])


def test_video_addresses():
    assert fetch.video_url("youtube", VIDEO_ID) == "https://www.youtube.com/watch?v=EXAMPLE0001"
    assert fetch.video_url("twitch", TWITCH_ID) == "https://www.twitch.tv/videos/0000000001"


def test_the_downloader_is_optional_and_not_imported_until_asked_for(monkeypatch, tmp_path):
    assert sys.modules.get("yt_dlp") is None
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # as if it were not installed
    with pytest.raises(fetch.DownloadSupportMissing, match=r'pip install -e "\.\[download\]" in the repository folder'):
        fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path)


class StandInLibrary:
    """Plays the downloader's library: records the options it is given and writes one file."""

    def __init__(self, produce: bool = True, fail: bool = False, duration: float | None = None) -> None:
        self.options: dict = {}
        self.urls: list[str] = []
        self.produce, self.fail = produce, fail
        self.duration = duration        # the length the video's information reports, when it does
        library = self

        class Session:
            def __init__(self, options):
                library.options = options

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                library.urls.extend(urls)
                if library.fail:
                    raise RuntimeError("ERROR: this video is not available")
                if "match_filter" in library.options:     # the library asks it about the video before it downloads
                    info = {} if library.duration is None else {"duration": library.duration}
                    if library.options["match_filter"](info, incomplete=False) is not None:
                        return
                if library.produce:
                    target = Path(library.options["outtmpl"].replace("%(ext)s", "mp4"))
                    target.write_bytes(b"video")
                    target.with_suffix(".mp4.part").write_bytes(b"leftover")

        class Utils:
            @staticmethod
            def download_range_func(chapters, ranges):
                return ("ranges", chapters, ranges)

        self.YoutubeDL = Session
        self.utils = Utils


def test_the_downloader_is_asked_for_the_contracts_stream_and_stretch(monkeypatch, tmp_path):
    library = StandInLibrary()
    monkeypatch.setitem(sys.modules, "yt_dlp", library)
    result = fetch.download_video("youtube", VIDEO_ID, 77.0, 84.0, tmp_path / "part")
    assert result == fetch.Download(path=tmp_path / "part" / "video.mp4", offset=77.0)
    assert library.urls == ["https://www.youtube.com/watch?v=EXAMPLE0001"]
    assert library.options["format"] == extract.FORMAT_SELECTOR and library.options["noplaylist"] is True
    assert library.options["download_ranges"] == ("ranges", None, [(77.0, 84.0)])
    assert library.options["force_keyframes_at_cuts"] is True

    whole = fetch.download_video("twitch", TWITCH_ID, None, None, tmp_path / "whole")
    assert whole.offset == 0.0 and library.urls[-1] == "https://www.twitch.tv/videos/0000000001"
    assert "download_ranges" not in library.options and "force_keyframes_at_cuts" not in library.options


def test_a_download_that_fails_or_produces_nothing_is_reported(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "yt_dlp", StandInLibrary(fail=True))
    with pytest.raises(fetch.DownloadFailed, match="not available"):
        fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "a")
    monkeypatch.setitem(sys.modules, "yt_dlp", StandInLibrary(produce=False))
    with pytest.raises(fetch.DownloadFailed, match="produced no file"):
        fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "b")


# Texts in the form the downloader writes them. The stock advice at the end of the second is on
# many ordinary errors and is no sign of a refusal.
REFUSALS = [
    "ERROR: [youtube] EXAMPLE0001: Unable to download webpage: HTTP Error 429: Too Many Requests (caused by <HTTPError 429: Too Many Requests>)",
    "ERROR: [youtube] EXAMPLE0001: Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies for the authentication.",
    "ERROR: [youtube] EXAMPLE0001: Sign in to confirm you\u2019re not a bot.",
    "ERROR: unable to download video data: HTTP Error 429: Too Many Requests",
    "ERROR: [generic] Unable to download webpage: redirected to https://www.example.com/sorry/index?continue=https://www.example.com/watch",
    "ERROR: fragment 1 not found: <HTTPError 429: too many requests>",
]
ORDINARY = [
    "ERROR: ffmpeg exited with code 8",
    "ERROR: [youtube] EXAMPLE0001: Video unavailable. This video is private",
    "ERROR: [youtube] EXAMPLE0001: Unable to download API page: timed out; please report this issue on  https://example.com/issues?q= , "
    "filling out the appropriate issue template. Confirm you are on the latest version using  yt-dlp -U",
    "ERROR: [youtube] EXAMPLE0001: This video is not available",
    "ERROR: unable to download video data: HTTP Error 403: Forbidden",
    "ERROR: [twitch:vod] 1429004290: the part 42900 of 4290 could not be read",
    "",
]


# ---------------------------------------------------------- the downloader's own messages
# The texts below are the downloader's own. The downloader is the package yt-dlp, and they were
# copied from its source at version 2026.08.19, in the form its errors reach this package:
# "ERROR: [<extractor>] <video id>: <message>". Each group names the file it comes from.
#
# The downloader writes some messages itself and passes others on from the platform. Where a
# message is the platform's, the note says so, and says where the downloader's source shows the
# words: in the code that looks for them, or in the notes on its own test videos. A few of the
# platform's sentences are not in the source at all. They are marked "the platform's words":
# they are what the platform is known to say, and nothing here checked them against it.
#
# yt_dlp/extractor/common.py, _login_hint, and yt_dlp/extractor/youtube/_base.py,
# _youtube_login_hint: the advice put after any reason that holds the words "sign in"
# (yt_dlp/extractor/youtube/_video.py, where the reason is turned into the error).
LOGIN_HINT = ("Use --cookies-from-browser or --cookies for the authentication. See  "
              "https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp  for how to manually pass cookies. "
              "Also see  https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies  for tips on effectively "
              "exporting YouTube cookies")

# yt_dlp/extractor/youtube/_video.py. The first sentence is the platform's reason, which the
# source tests for word by word; the rest is the downloader's.
RATE_LIMITED = ("ERROR: [youtube] EXAMPLE0001: This content isn't available, try again later. The current session has been "
                "rate-limited by YouTube for up to an hour. It is recommended to use `-t sleep` to add a delay between video "
                "requests to avoid exceeding the rate limit. For more information, refer to  "
                "https://github.com/yt-dlp/yt-dlp/wiki/Extractors#this-content-isnt-available-try-again-later")
RATE_LIMITED_ACCOUNT = RATE_LIMITED.replace("The current session has been", "Your account has been")
# The same file: the sentence the downloader adds to the platform's reason when the reply
# carries a captcha. "Video unavailable" stands in for that reason here.
CAPTCHA = "ERROR: [youtube] EXAMPLE0001: Video unavailable. YouTube is requiring a captcha challenge before playback"
# The same file, whole.
BLOCKED = "ERROR: [youtube] EXAMPLE0001: All player responses are invalid. Your IP is likely being blocked by Youtube"
# The platform's words for its bot check, followed by the downloader's advice.
BOT_CHECK = f"ERROR: [youtube] EXAMPLE0001: Sign in to confirm you\u2019re not a bot. {LOGIN_HINT}"
# The platform's reason for the rate limit by itself, as a version of the downloader without
# that sentence passes it on.
RATE_LIMITED_PLAIN = "ERROR: [youtube] EXAMPLE0001: Video unavailable. This content isn\u2019t available, try again later."

DOWNLOADER_REFUSALS = [RATE_LIMITED, RATE_LIMITED_ACCOUNT, CAPTCHA, BLOCKED, BOT_CHECK, RATE_LIMITED_PLAIN]

# Errors about one video. None of them is a refusal: the row fails and the run goes on. Each
# is given with the cause the index records for it.
#
# The age gate: the platform's words (the source looks for "confirm your age" and
# "inappropriate" in yt_dlp/extractor/youtube/_video.py), followed by the downloader's advice.
AGE_GATE = f"ERROR: [youtube] EXAMPLE0001: Sign in to confirm your age. This video may be inappropriate for some users. {LOGIN_HINT}"
# A private video: the platform's words ("Private video" is the note on a test video in
# yt_dlp/extractor/youtube/_video.py), followed by the downloader's advice.
PRIVATE = f"ERROR: [youtube] EXAMPLE0001: Private video. Sign in if you've been granted access to this video. {LOGIN_HINT}"
# yt_dlp/YoutubeDL.py, for the error raised in yt_dlp/extractor/youtube/_video.py when the
# platform's reason starts with these words about a country.
OTHER_COUNTRY = ("ERROR: [youtube] EXAMPLE0001: The uploader has not made this video available in your country\n"
                 "This video is available in Japan.\nYou might want to use a VPN or a proxy server (with --proxy) to workaround.")

PER_VIDEO = {
    AGE_GATE: "restricted",
    OTHER_COUNTRY: "restricted",
    # yt_dlp/extractor/common.py, raise_geo_restricted, and the line yt_dlp/YoutubeDL.py adds.
    "ERROR: [twitch:vod] 0000000001: This video is not available from your location due to geo restriction\n"
    "You might want to use a VPN or a proxy server (with --proxy) to workaround.": "restricted",
    # yt_dlp/extractor/twitch.py and yt_dlp/extractor/common.py, raise_login_required.
    "ERROR: [twitch:vod] 0000000001: You must be logged into an account that has access to this subscriber-only content. "
    "Use --cookies, --cookies-from-browser, --username and --password, --netrc-cmd, or --netrc (twitch) to provide account "
    "credentials. See  https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp  for how to manually pass "
    "cookies": "restricted",
    # yt_dlp/extractor/common.py, report_drm.
    "ERROR: [youtube] EXAMPLE0001: This video is DRM protected": "restricted",
    # The platform's words for a video of a channel's members, and for a video blocked in a country.
    "ERROR: [youtube] EXAMPLE0001: Join this channel to get access to members-only content like this video, and other "
    "exclusive perks.": "restricted",
    "ERROR: [youtube] EXAMPLE0001: Video unavailable. This video contains content from Example Records, who has blocked it "
    "in your country on copyright grounds": "restricted",
    PRIVATE: "unavailable",
    # yt_dlp/extractor/twitch.py.
    "ERROR: [twitch:vod] 0000000001: Video 0000000001 does not exist": "unavailable",
    # The platform's words as the notes on the downloader's test videos record them, in
    # yt_dlp/extractor/youtube/_video.py.
    "ERROR: [youtube] EXAMPLE0001: Private video": "unavailable",
    "ERROR: [youtube] EXAMPLE0001: This video does not exist.": "unavailable",
    "ERROR: [youtube] EXAMPLE0001: This video is not available.": "unavailable",
    "ERROR: [youtube] EXAMPLE0001: Video unavailable": "unavailable",
    "ERROR: [youtube] EXAMPLE0001: This video has been removed for violating YouTube's policy on hate speech.": "unavailable",
    # The platform's words for a video removed by its uploader, and for a closed account.
    "ERROR: [youtube] EXAMPLE0001: Video unavailable. This video has been removed by the uploader": "unavailable",
    "ERROR: [youtube] EXAMPLE0001: Video unavailable. This video is no longer available because the YouTube account "
    "associated with this video has been terminated.": "unavailable",
    # The platform's words for a premiere and for a live stream that has not begun, and the
    # downloader's for a channel that is not live (yt_dlp/utils/_utils.py, UserNotLive). The
    # video is not there yet, which is not the same as gone.
    "ERROR: [youtube] EXAMPLE0001: Premieres in 5 hours": "error",
    "ERROR: [youtube] EXAMPLE0001: This live event will begin in 2 hours.": "error",
    "ERROR: [twitch:stream] examplechannel: The channel is not currently live": "error",
    # "Video unavailable" with a reason after it that is none of the above proves nothing.
    "ERROR: [youtube] EXAMPLE0001: Video unavailable. This content isn't available on this app": "error",
    # yt_dlp/YoutubeDL.py: no stream of the kind asked for.
    "ERROR: [youtube] EXAMPLE0001: Requested format is not available. Use --list-formats for a list of available formats": "error",
    # yt_dlp/extractor/common.py with yt_dlp/networking/exceptions.py: no network, and a server that says no.
    "ERROR: [youtube] EXAMPLE0001: Unable to download API page: [Errno 8] nodename nor servname provided, or not known "
    "(caused by TransportError('[Errno 8] nodename nor servname provided, or not known'))": "error",
    "ERROR: [youtube] EXAMPLE0001: Unable to download webpage: HTTP Error 403: Forbidden (caused by <HTTPError 403: Forbidden>)": "error",
    "ERROR: [twitch:vod] 0000000001: Unable to download JSON metadata: HTTP Error 404: Not Found (caused by <HTTPError 404: Not Found>)": "error",
    # yt_dlp/downloader/external.py: all that is known when the program that cuts a section fails.
    "ERROR: ffmpeg exited with code 8": "error",
}


def test_a_refusal_by_the_platform_is_told_from_an_ordinary_error():
    for text in REFUSALS:
        assert fetch.refusal_sign(text) is not None, text
    for text in ORDINARY:
        assert fetch.refusal_sign(text) is None, text
    assert fetch.refusal_sign(REFUSALS[0]) == "HTTP Error 429" and fetch.refusal_sign(REFUSALS[1]) == "not a bot"
    # The downloader's stock advice is on many ordinary errors, and by itself is no sign of anything.
    advice = "please report this issue. Confirm you are on the latest version using  yt-dlp -U"
    assert fetch.refusal_sign(advice) is None and fetch.failure_cause(advice) == "error"
    assert issubclass(fetch.DownloadRefused, fetch.DownloadFailed)


@pytest.mark.parametrize("text", DOWNLOADER_REFUSALS)
def test_the_downloader_s_own_words_for_a_refusal_are_recognized(text):
    """A rate limit, a captcha, a blocked address and the bot check, in the downloader's words."""
    assert fetch.refusal_sign(text) is not None and fetch.failure_cause(text) == "refused"
    # The sign is still found when the text is cut down to the length of an index line.
    assert fetch.refusal_sign(fetch.error_summary(text)) is not None


@pytest.mark.parametrize("text", sorted(PER_VIDEO))
def test_an_error_about_one_video_is_no_refusal_and_has_its_cause(text):
    assert fetch.refusal_sign(text) is None, text
    assert fetch.failure_cause(text) == PER_VIDEO[text]
    assert fetch.failure_cause(f"youtube EXAMPLE0001: {text}\n") == PER_VIDEO[text]      # as this package passes it on


def test_the_bot_check_stops_a_run_and_the_age_gate_does_not():
    """Both start with "Sign in to confirm". The first is about whoever asks, and every later
    request meets it too. The second is about one video."""
    assert BOT_CHECK.split(": ", 2)[2].startswith("Sign in to confirm") and AGE_GATE.split(": ", 2)[2].startswith("Sign in to confirm")
    assert fetch.failure_cause(BOT_CHECK) == "refused" and fetch.refusal_sign(BOT_CHECK) == "not a bot"
    assert fetch.failure_cause(AGE_GATE) == "restricted" and fetch.refusal_sign(AGE_GATE) is None
    assert set(fetch.CAUSES) == {"unavailable", "restricted", "refused", "error"}


class Raising:
    """Plays the downloader's library: every download raises the given text."""

    def __init__(self, text: str) -> None:
        library = self
        self.text = text

        class Session:
            def __init__(self, options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                raise RuntimeError(library.text)

        self.YoutubeDL = Session


def test_the_downloader_s_error_says_which_kind_it_is(monkeypatch, tmp_path):
    for text in [*REFUSALS, *DOWNLOADER_REFUSALS]:
        monkeypatch.setitem(sys.modules, "yt_dlp", Raising(text))
        with pytest.raises(fetch.DownloadRefused) as refused:
            fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "a")
        assert refused.value.cause == "refused"
    for text in ORDINARY[:-1]:
        monkeypatch.setitem(sys.modules, "yt_dlp", Raising(text))
        with pytest.raises(fetch.DownloadFailed) as caught:
            fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "b")
        assert not isinstance(caught.value, fetch.DownloadRefused), text
    for text, cause in PER_VIDEO.items():
        monkeypatch.setitem(sys.modules, "yt_dlp", Raising(text))
        with pytest.raises(fetch.DownloadFailed) as caught:
            fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "c")
        assert not isinstance(caught.value, fetch.DownloadRefused) and caught.value.cause == cause, text
    monkeypatch.setitem(sys.modules, "yt_dlp", StandInLibrary(produce=False))
    with pytest.raises(fetch.DownloadFailed) as caught:
        fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "d")
    assert caught.value.cause == "error"


def test_the_cause_is_read_from_the_whole_error_before_it_is_shortened(monkeypatch, tmp_path):
    """An index line keeps the start and the end of a long error. The words that say why can
    lie between the two, so the cause is settled first."""
    text = "ERROR: [youtube] EXAMPLE0001: " + "a long preamble " * 25 + "This video is private. " + "boilerplate " * 40
    assert "is private" not in fetch.error_summary(text) and fetch.failure_cause(fetch.error_summary(text)) == "error"
    monkeypatch.setitem(sys.modules, "yt_dlp", Raising(text))
    with pytest.raises(fetch.DownloadFailed) as caught:
        fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "a")
    assert caught.value.cause == "unavailable" and "is private" not in str(caught.value)


def test_a_long_error_keeps_its_cause_and_its_end():
    cause = ("ERROR: [youtube] EXAMPLE0001: Unable to download API page: the connection was reset by the peer after the "
             "first bytes of the reply")
    advice = "; please report this issue on  https://example.com/issues?q= , filling out the appropriate issue template. " * 6
    text = cause + advice + "Confirm you are on the latest version using  yt-dlp -U"
    kept = fetch.error_summary(text)
    assert len(text) > 700 and len(kept) <= fetch.ERROR_HEAD_CHARACTERS + fetch.ERROR_TAIL_CHARACTERS + 7
    assert kept.startswith(cause) and kept.endswith("yt-dlp -U") and " [...] " in kept
    # The last 300 characters alone, which is what used to be kept, do not hold the cause.
    assert "connection was reset" not in text[-300:]
    # A short error is kept whole, and the first line that says something is the one kept.
    assert fetch.error_summary("  ERROR: ffmpeg exited with code 8\n") == "ERROR: ffmpeg exited with code 8"
    lines = "\n\n" + "first line with the cause " * 5 + "\n" + "x" * 900
    assert fetch.error_summary(lines).startswith("first line with the cause first line") and fetch.error_summary(lines).endswith("x" * 300)


def test_a_download_error_carries_the_cause_into_the_message(monkeypatch, tmp_path):
    text = "ERROR: [youtube] EXAMPLE0001: the cause comes first. " + "boilerplate " * 80
    monkeypatch.setitem(sys.modules, "yt_dlp", Raising(text))
    with pytest.raises(fetch.DownloadFailed, match="youtube EXAMPLE0001: ERROR: .youtube. EXAMPLE0001: the cause comes first"):
        fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "a")


def test_a_certificate_failure_says_what_to_install(monkeypatch, tmp_path):
    text = ("ERROR: [youtube] EXAMPLE0001: Unable to download API page: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify "
            "failed: unable to get local issuer certificate (_ssl.c:997); please report this issue. Confirm you are on the "
            "latest version using  yt-dlp -U")
    monkeypatch.setitem(sys.modules, "yt_dlp", Raising(text))
    with pytest.raises(fetch.CertificatesMissing) as caught:
        fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "a")
    message = str(caught.value)
    assert 'python -m pip install -e ".[download]"' in message and "certifi" in message and "CA certificates" in message
    assert not isinstance(caught.value, fetch.DownloadFailed)      # it is no failed row: the run cannot go on


def test_the_notice_says_whose_act_a_download_is():
    assert "your own act" in fetch.DOWNLOAD_NOTICE and "terms of service" in fetch.DOWNLOAD_NOTICE
    assert "--video-file" in fetch.DOWNLOAD_NOTICE


# ------------------------------------------------------------------------ from a local file
def test_frames_are_read_from_a_local_video_file(video, tmp_path):
    rows = [make_row(index) for index in (0, 45, 150, 299)]
    summary = fetch.fetch_frames(rows, tmp_path / "out", video_files={VIDEO_ID: video}, say=quiet)
    assert summary.complete and summary.count("ok") == 4 and summary.skipped == 0
    index = extract.read_index(tmp_path / "out")
    for number, row in zip((0, 45, 150, 299), rows):
        entry = index[row.key]
        assert entry.status == "ok" and entry.source == "local" and (entry.width, entry.height) == (320, 180)
        assert entry.file == f"frames/{extract.frame_file_name(row)}"
        assert entry.frame_time == pytest.approx(number / RATE, abs=1e-3) and entry.scale == pytest.approx(320 / 1920, abs=1e-4)
        assert shown(tmp_path / "out" / entry.file) == number
    store = extract.FrameStore(tmp_path / "out")
    assert read_counter(store.load(rows[2])) == 150
    assert not list((tmp_path / "out" / "frames").glob(".partial-*"))


def test_png_frames_are_lossless_and_exact_seeking_gives_the_same_frames(video, tmp_path):
    rows = [make_row(77), make_row(12)]
    fetch.fetch_frames(rows, tmp_path / "png", video_files={VIDEO_ID: video}, image_format="png", say=quiet)
    fetch.fetch_frames(rows, tmp_path / "exact", video_files={VIDEO_ID: video}, image_format="png", exact_seek=True, say=quiet)
    for row, number in zip(rows, (77, 12)):
        name = extract.frame_file_name(row, ".png")
        direct = extract.read_frame_at(video, row.t).rgb
        with Image.open(tmp_path / "png" / "frames" / name) as saved:
            assert np.array_equal(np.asarray(saved), direct)
        assert (tmp_path / "exact" / "frames" / name).read_bytes() == (tmp_path / "png" / "frames" / name).read_bytes()
        assert read_counter(direct) == number
    with pytest.raises(ValueError, match="unknown image format"):
        fetch.fetch_frames(rows, tmp_path / "bmp", video_files={VIDEO_ID: video}, image_format="bmp", say=quiet)


def test_a_fetch_can_be_repeated_and_only_does_what_is_missing(video, tmp_path):
    out = tmp_path / "out"
    first, second, third = make_row(10), make_row(20), make_row(30, OTHER_VIDEO_ID)
    partial = fetch.fetch_frames([first, second, third], out, video_files={VIDEO_ID: video}, say=quiet)
    assert not partial.complete and partial.count("ok") == 2 and partial.count("no-video") == 1
    assert extract.read_index(out)[third.key].status == "no-video"

    stamp = (out / "frames" / extract.frame_file_name(first)).stat().st_mtime_ns
    (out / "frames" / extract.frame_file_name(second)).unlink()   # a frame lost since
    again = fetch.fetch_frames([first, second, third], out, video_files={VIDEO_ID: video, OTHER_VIDEO_ID: video}, say=quiet)
    assert again.complete and again.skipped == 1
    assert [(entry.video_id, entry.status) for entry in again.entries] == [(VIDEO_ID, "ok"), (OTHER_VIDEO_ID, "ok")]
    assert (out / "frames" / extract.frame_file_name(first)).stat().st_mtime_ns == stamp
    assert shown(out / "frames" / extract.frame_file_name(second)) == 20
    assert shown(out / "frames" / extract.frame_file_name(third)) == 30

    nothing = fetch.fetch_frames([first, second, third], out, say=quiet)
    assert nothing.entries == [] and nothing.skipped == 3 and nothing.complete


def test_a_video_of_another_shape_makes_the_row_unusable(video, tmp_path):
    wide, square = make_row(40), make_row(50, width=1440, height=1080)
    summary = fetch.fetch_frames([wide, square], tmp_path / "out", video_files={VIDEO_ID: video}, say=quiet)
    assert [entry.status for entry in summary.entries] == ["ok", "unusable"] and not summary.complete
    entry = extract.read_index(tmp_path / "out")[square.key]
    assert "shapes differ" in entry.reason and entry.file is None and (entry.width, entry.height) == (320, 180)
    assert [path.name for path in (tmp_path / "out" / "frames").iterdir()] == [extract.frame_file_name(wide)]
    assert not extract.FrameStore(tmp_path / "out").has(square)
    # A looser tolerance is the caller's decision.
    loose = fetch.fetch_frames([square], tmp_path / "out", video_files={VIDEO_ID: video}, tolerance=0.5, say=quiet)
    assert loose.complete


def test_a_time_past_the_end_and_a_file_that_is_no_video_fail_row_by_row(video, tmp_path):
    inside, beyond = make_row(5), make_row(100_000)
    text = tmp_path / "notes.mp4"
    text.write_text("not a video")
    summary = fetch.fetch_frames([inside, beyond, make_row(1, OTHER_VIDEO_ID)], tmp_path / "out",
                                 video_files={VIDEO_ID: video, OTHER_VIDEO_ID: text}, say=quiet)
    assert [entry.status for entry in summary.entries] == ["ok", "failed", "failed"]
    assert "ends before" in summary.entries[1].reason


def test_a_local_file_can_be_named_with_its_platform(video, tmp_path):
    on_twitch = make_row(15, TWITCH_ID, platform="twitch")
    summary = fetch.fetch_frames([on_twitch], tmp_path / "out", video_files={f"twitch:{TWITCH_ID}": video}, say=quiet)
    assert summary.complete and shown(tmp_path / "out" / "frames" / "twitch-0000000001-t0000000517.jpg") == 15


def test_rows_timed_by_the_inference_command_get_their_own_frames(tmp_path):
    """``cardvision infer`` gives each frame it takes from a video a time, and a row made from
    it has no other address when it carries no fingerprint. At 29.97 frames a second the frames
    taken once a second are numbers 0, 30, 60 and 90, and those are the frames fetched."""
    pytest.importorskip("cv2")
    from cardvision.infer.frames import read_video

    recording = write_counter_video(tmp_path / "recording.mp4", rate=30000 / 1001, frames=100)
    taken = list(read_video(recording))
    assert [read_counter(frame.rgb) for frame in taken] == [0, 30, 60, 90]
    rows = [Row(platform="local", video_id="own-table", t=frame.t, frame_width=320, frame_height=180, source_type="physical",
                group="g-own", role="extra", trained_on=False, completeness="exhaustive-boxes", credit="Example Labeler", boxes=())
            for frame in taken]
    assert all(row.frame_dhash is None for row in rows)
    summary = fetch.fetch_frames(rows, tmp_path / "out", video_files={"own-table": recording}, image_format="png", say=quiet)
    assert summary.complete
    assert [shown(tmp_path / "out" / "frames" / extract.frame_file_name(row, ".png")) for row in rows] == [0, 30, 60, 90]


# ------------------------------------------------------------------ a recording of your own
LOCAL_ID = "kitchen-table-01"


def local_row(index: int, video_id: str = LOCAL_ID) -> Row:
    return make_row(index, video_id, platform="local")


def test_a_recording_of_your_own_has_no_address():
    with pytest.raises(ValueError, match=f"has no address; give its file with --video-file {LOCAL_ID}=PATH"):
        fetch.video_url("local", LOCAL_ID)


def test_a_recording_of_your_own_is_read_from_the_file_given_for_it(video, tmp_path):
    rows = [local_row(12), local_row(200), local_row(40, "second-game")]
    files = {LOCAL_ID: video, "local:second-game": video}
    summary = fetch.fetch_frames(rows, tmp_path / "out", video_files=files, say=quiet)
    assert summary.complete and [entry.source for entry in summary.entries] == ["local"] * 3
    assert shown(tmp_path / "out" / "frames" / "local-kitchen-table-01-t0000000417.jpg") == 12
    assert shown(tmp_path / "out" / "frames" / extract.frame_file_name(rows[2])) == 40
    assert read_counter(extract.FrameStore(tmp_path / "out").load(rows[1])) == 200


def test_a_recording_of_your_own_is_never_downloaded(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = StandInDownloader()
    said: list[str] = []
    rows = [local_row(12), local_row(13), make_row(60)]
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=downloader, pause=0, say=said.append)
    assert [entry.status for entry in summary.entries] == ["no-video", "no-video", "ok"]
    assert [call[:2] for call in downloader.calls] == [("youtube", VIDEO_ID)]      # nothing was asked for the local rows
    assert summary.entries[0].reason == fetch.NO_LOCAL_VIDEO_REASON and "pass --video-file ID=PATH" in summary.entries[0].reason
    assert f"local {LOCAL_ID}: no video file given; pass --video-file {LOCAL_ID}=PATH; 2 row(s) left without a frame" in said
    # The same rows without --download: the same status, and the message still names --video-file.
    plain = fetch.fetch_frames(rows[:2], tmp_path / "plain", say=quiet)
    assert [(entry.status, entry.reason) for entry in plain.entries] == [("no-video", fetch.NO_LOCAL_VIDEO_REASON)] * 2
    # A second run with the file fills the rows in.
    again = fetch.fetch_frames(rows, tmp_path / "out", video_files={LOCAL_ID: video}, download=True, downloader=downloader,
                               pause=0, say=quiet)
    assert again.complete and again.skipped == 1 and len(downloader.calls) == 1


# ----------------------------------------------------------------------- the download path
class StandInDownloader:
    """Plays the downloader: makes a counter video for the stretch asked for, and remembers
    what it was asked and whether an earlier file was still there."""

    def __init__(self, fail_for: str | None = None) -> None:
        self.calls: list[tuple[str, str, float | None, float | None]] = []
        self.folders: list[Path] = []
        self.left_behind = 0
        self.fail_for = fail_for

    def __call__(self, platform, video_id, start, end, folder):
        self.left_behind += sum(1 for earlier in self.folders if earlier.exists())
        self.calls.append((platform, video_id, start, end))
        self.folders.append(folder)
        if video_id == self.fail_for:
            raise fetch.DownloadFailed(f"{platform} {video_id}: not available")
        import cv2

        first = int(round((start or 0.0) * RATE))
        count = int(round(((end - start) if start is not None else 10.0) * RATE))
        writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), RATE, (320, 180))
        for index in range(first, first + count):
            writer.write(np.ascontiguousarray(counter_frame(index)[:, :, ::-1]))
        writer.release()
        return fetch.Download(path=folder / "video.mp4", offset=float(start or 0.0))


def test_nothing_is_downloaded_unless_asked(video, tmp_path):
    downloader = StandInDownloader()
    summary = fetch.fetch_frames([make_row(10)], tmp_path / "out", downloader=downloader, say=quiet)
    assert downloader.calls == [] and summary.count("no-video") == 1
    assert "--download was not asked for" in summary.entries[0].reason


def test_downloads_happen_one_at_a_time_in_stretches_and_leave_no_video_behind(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = StandInDownloader()
    waits: list[float] = []
    rows = [make_row(60), make_row(90), make_row(2400), make_row(150, OTHER_VIDEO_ID), make_row(33, "EXAMPLE0003")]
    local = {"EXAMPLE0003": video}
    summary = fetch.fetch_frames(rows, tmp_path / "out", video_files=local, download=True, downloader=downloader,
                                 pause=7.0, say=quiet, sleep=waits.append)
    assert summary.complete
    # Two stretches of the first video (2 s and 3 s share one; 80 s is on its own), then the second video.
    assert downloader.calls == [("youtube", VIDEO_ID, 0.0, 7.0), ("youtube", VIDEO_ID, 77.0, 84.0),
                                ("youtube", OTHER_VIDEO_ID, 2.0, 9.0)]
    assert waits == [7.0, 7.0]                      # a pause before every download but the first
    assert downloader.left_behind == 0              # each file was gone before the next was asked for
    assert not any(folder.exists() for folder in downloader.folders)
    index = extract.read_index(tmp_path / "out")
    for row, number in zip(rows, (60, 90, 2400, 150, 33)):
        assert shown(tmp_path / "out" / index[row.key].file) == number
    assert [index[row.key].source for row in rows] == ["download"] * 4 + ["local"]


def test_whole_videos_are_downloaded_when_asked_or_when_stretches_cannot_be_cut(video, tmp_path, monkeypatch):
    rows = [make_row(60), make_row(240)]
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    asked = StandInDownloader()
    assert fetch.fetch_frames(rows, tmp_path / "a", download=True, downloader=asked, whole_video=True, say=quiet).complete
    assert asked.calls == [("youtube", VIDEO_ID, None, None)]
    monkeypatch.setattr(fetch, "sections_supported", lambda: False)
    forced = StandInDownloader()
    assert fetch.fetch_frames(rows, tmp_path / "b", download=True, downloader=forced, say=quiet).complete
    assert forced.calls == [("youtube", VIDEO_ID, None, None)]


def test_a_failed_download_fails_its_rows_and_the_rest_go_on(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = StandInDownloader(fail_for=VIDEO_ID)
    rows = [make_row(60), make_row(61), make_row(150, OTHER_VIDEO_ID)]
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=downloader, pause=0, say=quiet)
    assert [entry.status for entry in summary.entries] == ["failed", "failed", "ok"]
    assert "not available" in summary.entries[0].reason
    # A second run asks again only for what failed.
    retry = StandInDownloader()
    again = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=retry, pause=0, say=quiet)
    assert again.complete and again.skipped == 1 and [call[1] for call in retry.calls] == [VIDEO_ID]


# ------------------------------------------------------------- failures on a real network
class Scripted(StandInDownloader):
    """A stand-in downloader that fails some of its calls: ``script`` maps the number of a call
    (from 1) to the text of the error that call raises."""

    def __init__(self, script: dict[int, str]) -> None:
        super().__init__()
        self.script = script

    def __call__(self, platform, video_id, start, end, folder):
        text = self.script.get(len(self.calls) + 1)
        if text is not None:
            self.left_behind += sum(1 for earlier in self.folders if earlier.exists())
            self.calls.append((platform, video_id, start, end))
            self.folders.append(folder)
            raise fetch.DownloadFailed(f"{platform} {video_id}: {text}")
        return super().__call__(platform, video_id, start, end, folder)


def test_a_failed_part_is_asked_for_once_more_after_the_pause(video, tmp_path, monkeypatch):
    """A download that fails for a passing reason succeeds on the second try, and the index
    says that two requests were made for it."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = Scripted({2: "ERROR: ffmpeg exited with code 8"})
    waits: list[float] = []
    said: list[str] = []
    rows = [make_row(60), make_row(2400), make_row(150, OTHER_VIDEO_ID)]
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=downloader, pause=4.0, say=said.append,
                                 sleep=waits.append)
    assert summary.complete and [entry.attempts for entry in summary.entries] == [1, 2, 1]
    assert downloader.calls == [("youtube", VIDEO_ID, 0.0, 6.0), ("youtube", VIDEO_ID, 77.0, 84.0),
                                ("youtube", VIDEO_ID, 77.0, 84.0), ("youtube", OTHER_VIDEO_ID, 2.0, 9.0)]
    assert waits == [4.0, 4.0, 4.0]                     # the usual pause before the second try too
    assert downloader.left_behind == 0 and not any(folder.exists() for folder in downloader.folders)
    assert any("a download failed (youtube EXAMPLE0001: ERROR: ffmpeg exited with code 8); it is tried once more" in line
               for line in said)
    index = extract.read_index(tmp_path / "out")
    assert json.loads((tmp_path / "out" / "index.jsonl").read_text().splitlines()[1])["attempts"] == 2
    assert shown(tmp_path / "out" / index[rows[1].key].file) == 2400


def test_the_number_of_retries_can_be_set(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    always = {number: "ERROR: ffmpeg exited with code 8" for number in range(1, 20)}
    rows = [make_row(60)]
    for retries, calls in ((0, 1), (1, 2), (3, 4)):
        downloader = Scripted(always)
        summary = fetch.fetch_frames(rows, tmp_path / f"out-{retries}", download=True, downloader=downloader, pause=0,
                                     retries=retries, say=quiet)
        assert len(downloader.calls) == calls and summary.count("failed") == 1 and summary.entries[0].attempts == calls
        assert summary.refused is None and "ffmpeg exited with code 8" in summary.entries[0].reason
    assert fetch.DEFAULT_RETRIES == 1
    with pytest.raises(ValueError, match="0 or more"):
        fetch.fetch_frames(rows, tmp_path / "bad", download=True, downloader=Scripted({}), retries=-1, say=quiet)


@pytest.mark.parametrize("text", [*REFUSALS[:3], *DOWNLOADER_REFUSALS])
def test_a_refusal_stops_the_whole_run_at_once(video, tmp_path, monkeypatch, text):
    """A rate limit or a bot check is not a failed row. Nothing is asked for after it, the rows
    done keep their index lines, and the rows not tried get none."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = Scripted({2: text})
    waits: list[float] = []
    rows = [make_row(60), make_row(2400), make_row(4800), make_row(150, OTHER_VIDEO_ID), make_row(33, "EXAMPLE0003")]
    summary = fetch.fetch_frames(rows, tmp_path / "out", video_files={"EXAMPLE0003": video}, download=True,
                                 downloader=downloader, pause=5.0, say=quiet, sleep=waits.append)
    assert len(downloader.calls) == 2 and waits == [5.0]          # no second try, and no later request
    assert summary.refused == f"youtube {VIDEO_ID}: {text}" and summary.remaining == 4 and not summary.complete
    assert [(entry.t, entry.status) for entry in summary.entries] == [(rows[0].t, "ok")]
    lines = [json.loads(line) for line in (tmp_path / "out" / "index.jsonl").read_text().splitlines()]
    assert [(line["t"], line["status"]) for line in lines] == [(rows[0].t, "ok")]       # no "failed" line for the refused part
    # The run can be repeated later: it picks up where it stopped.
    again = fetch.fetch_frames(rows, tmp_path / "out", video_files={"EXAMPLE0003": video}, download=True,
                               downloader=StandInDownloader(), pause=0, say=quiet)
    assert again.complete and again.skipped == 1 and again.count("ok") == 4 and again.refused is None


def test_an_ordinary_error_with_the_downloader_s_stock_advice_does_not_stop_the_run(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    advice = ORDINARY[2]
    assert "Confirm you are on the latest version" in advice
    downloader = Scripted({1: advice, 2: advice})
    rows = [make_row(60), make_row(150, OTHER_VIDEO_ID)]
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=downloader, pause=0, say=quiet)
    assert summary.refused is None and summary.remaining == 0
    assert [(entry.status, entry.attempts) for entry in summary.entries] == [("failed", 2), ("ok", 1)]
    assert len(downloader.calls) == 3


def test_an_age_gate_fails_one_video_and_the_run_goes_on(video, tmp_path, monkeypatch):
    """ "Sign in to confirm your age" is about one video. Its rows fail, with the cause
    ``restricted``, and the videos after it are still asked for."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = Scripted({1: AGE_GATE})
    rows = [make_row(60), make_row(150, OTHER_VIDEO_ID), make_row(33, "EXAMPLE0003")]
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=downloader, pause=0, say=quiet)
    assert summary.refused is None and summary.stopped is None and summary.remaining == 0
    assert [(entry.status, entry.cause) for entry in summary.entries] == [("failed", "restricted"), ("ok", None), ("ok", None)]
    assert len(downloader.calls) == 3                   # the video behind the age gate is asked for once


@pytest.mark.parametrize(("text", "cause"), [(PRIVATE, "unavailable"), (AGE_GATE, "restricted")])
def test_a_video_that_is_gone_or_closed_is_not_asked_for_again(video, tmp_path, monkeypatch, text, cause):
    """A failed download is tried again because most failures pass. The platform's statement
    that a video is private, removed or missing, or that this request may not have it, does
    not pass: a second request would cost a request and get the same answer."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    rows = [make_row(60), make_row(2400), make_row(150, OTHER_VIDEO_ID)]
    downloader = Scripted({1: text})
    waits: list[float] = []
    said: list[str] = []
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=downloader, pause=4.0, retries=3,
                                 say=said.append, sleep=waits.append)
    # The video's second part is not asked for either: the first part's answer covers the video.
    assert [(entry.status, entry.cause, entry.attempts) for entry in summary.entries] == [
        ("failed", cause, 1), ("failed", cause, 0), ("ok", None, 1)]
    assert downloader.calls == [("youtube", VIDEO_ID, 0.0, 6.0), ("youtube", OTHER_VIDEO_ID, 2.0, 9.0)]
    assert waits == [4.0] and not any("tried once more" in line for line in said)
    assert json.loads((tmp_path / "out" / "index.jsonl").read_text().splitlines()[0])["attempts"] == 1
    # The cause may also come with the error itself, whatever its words.
    stated = getattr(fetch, f"CAUSE_{cause.upper()}")
    calls: list[str] = []

    def answering(platform, video_id, start, end, folder):
        calls.append(video_id)
        raise fetch.DownloadFailed(f"{platform} {video_id}: no", stated)

    summary = fetch.fetch_frames(rows[:1], tmp_path / "stated", download=True, downloader=answering, pause=0, retries=5, say=quiet)
    assert calls == [VIDEO_ID] and [(entry.cause, entry.attempts) for entry in summary.entries] == [(cause, 1)]
    # A failure that says nothing about the video is still tried again, as often as asked.
    passing = Scripted({1: "ERROR: ffmpeg exited with code 8", 2: "ERROR: ffmpeg exited with code 8"})
    summary = fetch.fetch_frames(rows[:1], tmp_path / "passing", download=True, downloader=passing, pause=0, retries=3, say=quiet)
    assert len(passing.calls) == 3 and [(entry.status, entry.attempts) for entry in summary.entries] == [("ok", 3)]


# ------------------------------------------------------- several downloads in a row that fail
FFMPEG = "ERROR: ffmpeg exited with code 8"      # all that comes back when the media server turns the cutting program away
FIVE_PARTS = (60, 2400, 4800, 7200, 9600)        # frames 80 seconds apart: each is a part of its own
FIVE_VIDEOS = (VIDEO_ID, OTHER_VIDEO_ID, "EXAMPLE0003", "EXAMPLE0004", "EXAMPLE0005")


def test_three_failures_in_a_row_stop_the_run(video, tmp_path, monkeypatch):
    """A rate limit on the media server cannot be read from the error. Three parts in a row
    that fail, each after its retry, stop the run as a refusal does."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = Scripted({number: FFMPEG for number in range(1, 7)})
    waits: list[float] = []
    rows = [*(make_row(index) for index in FIVE_PARTS), make_row(150, OTHER_VIDEO_ID)]
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=downloader, pause=5.0, say=quiet,
                                 sleep=waits.append)
    assert fetch.DEFAULT_STOP_AFTER_FAILURES == 3
    assert len(downloader.calls) == 6 and waits == [5.0] * 5          # three parts, each asked for twice, and nothing after
    assert summary.stopped == 3 and summary.refused is None and summary.remaining == 3 and not summary.complete
    assert [(entry.status, entry.cause, entry.attempts) for entry in summary.entries] == [("failed", "error", 2)] * 3
    # The rows that failed have their lines; the rows not tried have none.
    assert len((tmp_path / "out" / "index.jsonl").read_text().splitlines()) == 3
    # A later run tries the failed rows again and goes on to the rest.
    again = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=StandInDownloader(), pause=0, say=quiet)
    assert again.complete and again.count("ok") == 6 and again.stopped is None


def test_a_download_that_succeeds_sets_the_count_back(video, tmp_path, monkeypatch):
    """Failure, success, failure, failure: never three in a row, so the run goes to its end."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = Scripted({1: FFMPEG, 3: FFMPEG, 4: FFMPEG})
    rows = [make_row(index) for index in FIVE_PARTS]
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=downloader, pause=0, retries=0, say=quiet)
    assert len(downloader.calls) == 5 and summary.stopped is None and summary.remaining == 0
    assert [entry.status for entry in summary.entries] == ["failed", "ok", "failed", "failed", "ok"]
    # One more failure before the success would have been the third in a row.
    stopping = Scripted({2: FFMPEG, 3: FFMPEG, 4: FFMPEG})
    stopped = fetch.fetch_frames(rows, tmp_path / "three", download=True, downloader=stopping, pause=0, retries=0, say=quiet)
    assert len(stopping.calls) == 4 and stopped.stopped == 3 and stopped.remaining == 1
    assert [entry.status for entry in stopped.entries] == ["ok", "failed", "failed", "failed"]


def test_the_number_of_failures_that_stops_a_run_can_be_set_and_zero_never_stops(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    rows = [make_row(index) for index in FIVE_PARTS]
    always = {number: FFMPEG for number in range(1, 30)}
    never = Scripted(always)
    summary = fetch.fetch_frames(rows, tmp_path / "never", download=True, downloader=never, pause=0, stop_after_failures=0, say=quiet)
    assert len(never.calls) == 10 and summary.count("failed") == 5 and summary.stopped is None and summary.remaining == 0
    for limit, parts in ((1, 1), (2, 2), (4, 4)):
        downloader = Scripted(always)
        summary = fetch.fetch_frames(rows, tmp_path / f"limit-{limit}", download=True, downloader=downloader, pause=0, retries=0,
                                     stop_after_failures=limit, say=quiet)
        assert (len(downloader.calls), summary.stopped, summary.remaining) == (parts, limit, 5 - parts)
    # The last part of a run has nothing after it: its failure ends the run and stops nothing.
    last = Scripted(always)
    summary = fetch.fetch_frames(rows, tmp_path / "last", download=True, downloader=last, pause=0, retries=0,
                                 stop_after_failures=5, say=quiet)
    assert len(last.calls) == 5 and summary.stopped is None and summary.remaining == 0 and not summary.complete
    with pytest.raises(ValueError, match="0 or more"):
        fetch.fetch_frames(rows, tmp_path / "bad", download=True, downloader=Scripted({}), stop_after_failures=-1, say=quiet)


def test_videos_that_are_gone_do_not_stop_a_run(video, tmp_path, monkeypatch):
    """A failure with a stated reason is the platform's answer about one video. Any number of
    them in a row is no sign of a rate limit, and they do not set the count back either."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    rows = [make_row(index, video_id) for index, video_id in zip(FIVE_PARTS, FIVE_VIDEOS, strict=True)]
    gone = Scripted({1: PRIVATE, 2: PRIVATE, 3: AGE_GATE, 4: PRIVATE})
    summary = fetch.fetch_frames(rows, tmp_path / "gone", download=True, downloader=gone, pause=0, retries=0, say=quiet)
    assert len(gone.calls) == 5 and summary.stopped is None
    assert [entry.cause for entry in summary.entries] == ["unavailable", "unavailable", "restricted", "unavailable", None]
    mixed = Scripted({1: FFMPEG, 2: PRIVATE, 3: FFMPEG, 4: FFMPEG})
    summary = fetch.fetch_frames(rows, tmp_path / "mixed", download=True, downloader=mixed, pause=0, retries=0, say=quiet)
    assert len(mixed.calls) == 4 and summary.stopped == 3 and summary.remaining == 1


# ------------------------------------------------------------ a whole video that is too long
def test_the_downloader_reads_a_videos_length_from_what_it_already_has(monkeypatch, tmp_path):
    """The limit is a filter the downloader's library applies to the video's information before
    it downloads. It asks for nothing of its own, and a video of unknown length is downloaded."""
    library = StandInLibrary(duration=7200.0)
    monkeypatch.setitem(sys.modules, "yt_dlp", library)
    fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "none")
    assert "match_filter" not in library.options                         # no limit given: nothing is added
    with pytest.raises(fetch.VideoTooLong) as raised:
        fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "long", max_minutes=60)
    assert str(raised.value) == ("youtube EXAMPLE0001: the video is 120 minutes long, longer than the limit of 60 minutes "
                                 "(--max-whole-minutes); it was not downloaded")
    assert raised.value.cause == fetch.CAUSE_ERROR and library.urls == [fetch.video_url("youtube", VIDEO_ID)] * 2
    assert not list((tmp_path / "long").glob("video.*"))
    assert fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "exact", max_minutes=120).path.is_file()
    unknown = StandInLibrary()
    monkeypatch.setitem(sys.modules, "yt_dlp", unknown)
    assert fetch.download_video("youtube", VIDEO_ID, None, None, tmp_path / "unknown", max_minutes=1).path.is_file()


class LengthAwareDownloader(StandInDownloader):
    """A stand-in that takes the limit and knows the length of each video, in minutes."""

    def __init__(self, minutes: dict[str, float]) -> None:
        super().__init__()
        self.minutes = minutes
        self.limits: list[float | None] = []

    def __call__(self, platform, video_id, start, end, folder, max_minutes=None):
        self.limits.append(max_minutes)
        if max_minutes is not None and self.minutes[video_id] > max_minutes:
            self.calls.append((platform, video_id, start, end))
            raise fetch.VideoTooLong(f"{platform} {video_id}: the video is {self.minutes[video_id]:g} minutes long")
        return super().__call__(platform, video_id, start, end, folder)


def test_a_whole_video_longer_than_the_limit_is_not_downloaded_and_the_run_goes_on(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    rows = [make_row(60), make_row(75), make_row(150, OTHER_VIDEO_ID)]
    downloader = LengthAwareDownloader({VIDEO_ID: 240, OTHER_VIDEO_ID: 20})
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=downloader, whole_video=True, pause=0,
                                 retries=3, stop_after_failures=1, max_whole_minutes=60, say=quiet)
    # Asked once (a long video is not tried again), no stop (it is no sign of a rate limit), the next video downloaded.
    assert downloader.calls == [("youtube", VIDEO_ID, None, None), ("youtube", OTHER_VIDEO_ID, None, None)]
    assert downloader.limits == [60, 60] and summary.stopped is None and summary.remaining == 0
    assert [(entry.status, entry.source, entry.cause, entry.attempts) for entry in summary.entries] == [
        ("failed", "download", "error", 1), ("failed", "download", "error", 1), ("ok", "download", None, 1)]
    assert "240 minutes long" in summary.entries[0].reason and summary.entries[0].reason == summary.entries[1].reason
    # Without a limit nothing changes: the downloader is called as before, with no extra argument.
    plain = LengthAwareDownloader({VIDEO_ID: 240, OTHER_VIDEO_ID: 20})
    summary = fetch.fetch_frames(rows, tmp_path / "plain", download=True, downloader=plain, whole_video=True, pause=0, say=quiet)
    assert plain.limits == [None, None] and summary.complete
    with pytest.raises(ValueError, match="more than 0 minutes"):
        fetch.fetch_frames(rows, tmp_path / "bad", download=True, downloader=plain, max_whole_minutes=0, say=quiet)


def test_the_limit_applies_to_the_whole_video_fallback_and_not_to_sections(video, tmp_path, monkeypatch):
    rows = [make_row(60), make_row(150, OTHER_VIDEO_ID)]
    lengths = {VIDEO_ID: 240, OTHER_VIDEO_ID: 20}
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    sections = LengthAwareDownloader(lengths)
    summary = fetch.fetch_frames(rows, tmp_path / "sections", download=True, downloader=sections, pause=0, max_whole_minutes=60, say=quiet)
    assert sections.limits == [None, None] and summary.complete                    # parts of a video are cut, not measured
    monkeypatch.setattr(fetch, "sections_supported", lambda: False)                # no ffmpeg: each video is fetched whole
    whole = LengthAwareDownloader(lengths)
    summary = fetch.fetch_frames(rows, tmp_path / "fallback", download=True, downloader=whole, pause=0, max_whole_minutes=60, say=quiet)
    assert whole.limits == [60, 60] and [entry.status for entry in summary.entries] == ["failed", "ok"]


def test_the_command_takes_a_longest_whole_video(video, tmp_path, capsys, monkeypatch):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [make_row(60), make_row(150, OTHER_VIDEO_ID)])
    monkeypatch.setattr(fetch, "downloader_installed", lambda: True)
    downloader = LengthAwareDownloader({VIDEO_ID: 240, OTHER_VIDEO_ID: 20})
    monkeypatch.setattr(fetch, "download_video", downloader)
    base = ["fetch", str(path), "--download", "--whole-video", "--pause", "0"]
    assert cli.main([*base, "--out", str(tmp_path / "out"), "--max-whole-minutes", "60"]) == 1
    captured = capsys.readouterr()
    assert downloader.limits == [60.0, 60.0] and "1 frame(s) written, 0 already present, 0 unusable, 1 failed" in captured.out
    assert "240 minutes long" in captured.err
    for bad in ("0", "-5"):
        assert cli.main([*base, "--out", str(tmp_path / "bad"), "--max-whole-minutes", bad]) == 1
        assert "--max-whole-minutes is more than 0" in capsys.readouterr().err


# -------------------------------------------------------------- what an index line records
MOMENT = datetime.datetime(2026, 1, 31, 23, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))   # 04:30 the next day, in UTC


def test_every_index_line_says_when_it_was_written_and_a_failed_download_says_why(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    text = tmp_path / "notes.mp4"
    text.write_text("not a video")
    rows = [make_row(60), make_row(2400), make_row(4800, "EXAMPLE0005"), make_row(7200, "EXAMPLE0006"),
            make_row(5, OTHER_VIDEO_ID), make_row(9, "EXAMPLE0003"), make_row(11, "EXAMPLE0004")]
    downloader = Scripted({2: PRIVATE, 3: AGE_GATE, 4: FFMPEG})
    summary = fetch.fetch_frames(rows, tmp_path / "out", video_files={OTHER_VIDEO_ID: text}, download=False, say=quiet,
                                 now=lambda: MOMENT)
    assert [(entry.status, entry.source, entry.cause) for entry in summary.entries] == [
        ("no-video", None, None)] * 4 + [("failed", "local", None), ("no-video", None, None), ("no-video", None, None)]
    summary = fetch.fetch_frames(rows[:4], tmp_path / "out", download=True, downloader=downloader, pause=0, retries=0, say=quiet,
                                 now=lambda: MOMENT)
    assert [(entry.status, entry.cause) for entry in summary.entries] == [
        ("ok", None), ("failed", "unavailable"), ("failed", "restricted"), ("failed", "error")]
    lines = [json.loads(line) for line in (tmp_path / "out" / "index.jsonl").read_text().splitlines()]
    assert len(lines) == 11 and all(line["written"] == "2026-02-01T04:30:00Z" for line in lines)
    assert [line.get("cause") for line in lines[7:]] == [None, "unavailable", "restricted", "error"]
    assert "cause" not in lines[7] and lines[8]["reason"].startswith("youtube EXAMPLE0001: ERROR: [youtube] EXAMPLE0001: Private video")
    # The index reads back with both fields, and the date of a line is its date in UTC.
    index = extract.read_index(tmp_path / "out")
    assert index[rows[1].key].cause == "unavailable" and index[rows[1].key].written == "2026-02-01T04:30:00Z"
    assert extract.index_date(index[rows[0].key].written) == "2026-02-01"
    # Without a clock given, the line carries the present moment in the same form.
    fetch.fetch_frames(rows[:1], tmp_path / "now", video_files={VIDEO_ID: video}, say=quiet)
    (line,) = (tmp_path / "now" / "index.jsonl").read_text().splitlines()
    written = datetime.datetime.strptime(json.loads(line)["written"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    assert abs((datetime.datetime.now(datetime.timezone.utc) - written).total_seconds()) < 120


def test_a_download_that_came_and_cannot_be_read_is_an_error(video, tmp_path, monkeypatch):
    """The file arrived, and it is no video or stops short of the row. That says nothing about
    whether the video is on its platform."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)

    def broken(platform, video_id, start, end, folder):
        (folder / "video.mp4").write_text("cut off")
        return fetch.Download(path=folder / "video.mp4", offset=float(start or 0.0))

    summary = fetch.fetch_frames([make_row(60)], tmp_path / "out", download=True, downloader=broken, say=quiet)
    assert [(entry.status, entry.source, entry.cause) for entry in summary.entries] == [("failed", "download", "error")]
    # A whole video that ends before the row's time fails the same way.
    short = fetch.fetch_frames([make_row(2400)], tmp_path / "short", download=True, downloader=StandInDownloader(), whole_video=True,
                               say=quiet)
    assert [(entry.status, entry.cause) for entry in short.entries] == [("failed", "error")] and "ends before" in short.entries[0].reason


def test_a_certificate_failure_ends_the_run_and_is_no_failed_row(video, tmp_path, monkeypatch):
    def no_certificates(platform, video_id, start, end, folder):
        raise fetch.CertificatesMissing(f"{platform} {video_id}: install certificates")

    with pytest.raises(fetch.CertificatesMissing):
        fetch.fetch_frames([make_row(60), make_row(150, OTHER_VIDEO_ID)], tmp_path / "out", download=True,
                           downloader=no_certificates, pause=0, say=quiet)
    assert extract.read_index(tmp_path / "out") == {}


def test_a_frame_of_another_size_is_written_and_marked(video, tmp_path):
    """The video is 320 x 180. A row labeled at 1920 x 1080 has the same shape, so its frame
    is written, and the index says that the size differs. A row labeled at 320 x 180 is not marked."""
    larger, same = make_row(40), make_row(50, width=320, height=180)
    summary = fetch.fetch_frames([larger, same], tmp_path / "out", video_files={VIDEO_ID: video}, say=quiet)
    assert summary.complete and [entry.other_size for entry in summary.entries] == [True, None]
    assert summary.other_sizes() == {(320, 180, 1920, 1080): 1}
    lines = [json.loads(line) for line in (tmp_path / "out" / "index.jsonl").read_text().splitlines()]
    assert lines[0]["other_size"] is True and lines[0]["scale"] == pytest.approx(320 / 1920, abs=1e-4)
    assert "other_size" not in lines[1] and lines[1]["scale"] == 1.0
    assert extract.read_index(tmp_path / "out")[larger.key].other_size is True
    assert extract.FrameStore(tmp_path / "out").has(larger)       # the frame is used: labels are fractions


# ------------------------------------------------------------------------- kept video files
def test_a_kept_file_s_name_says_which_video_it_is_and_where_it_starts():
    assert fetch.kept_video_name("youtube", VIDEO_ID, 77.0, ".mp4") == "youtube-EXAMPLE0001+77.mp4"
    assert fetch.kept_video_name("youtube", VIDEO_ID, 0.0, ".webm") == "youtube-EXAMPLE0001+0.webm"
    assert fetch.kept_video_name("twitch", TWITCH_ID, None, ".mp4") == "twitch-0000000001.mp4"
    assert fetch.kept_video_name("youtube", VIDEO_ID, 12.5, ".mkv") == "youtube-EXAMPLE0001+12.5.mkv"
    assert [fetch.seconds_text(value) for value in (0, 4578.0, 100.0, 0.033, 12.5004, 1e-9)] == ["0", "4578", "100", "0.033", "12.5", "0"]
    with pytest.raises(ValueError):
        fetch.seconds_text(-1.0)


def kept_index(folder: Path) -> dict:
    return json.loads((folder / "video-files.json").read_text())


def test_downloaded_files_are_kept_under_their_names_with_an_index(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = StandInDownloader()
    rows = [make_row(60), make_row(90), make_row(2400), make_row(150, OTHER_VIDEO_ID), make_row(33, "EXAMPLE0003")]
    kept = tmp_path / "kept"
    summary = fetch.fetch_frames(rows, tmp_path / "out", video_files={"EXAMPLE0003": video}, download=True,
                                 downloader=downloader, keep_video=kept, pause=0, say=quiet)
    assert summary.complete
    # One file per download, none for the video that was given as a local file.
    names = ["youtube-EXAMPLE0001+0.mp4", "youtube-EXAMPLE0001+77.mp4", "youtube-EXAMPLE0002+2.mp4"]
    assert sorted(path.name for path in kept.iterdir()) == ["video-files.json", *names]
    assert [entry.file for entry in summary.kept] == names
    assert not any(folder.exists() for folder in downloader.folders)       # the scratch folders are still removed
    assert kept_index(kept) == {"kind": "cardvision-video-files", "version": 1, "files": [
        {"file": names[0], "platform": "youtube", "video_id": VIDEO_ID, "start": 0.0, "end": 7.0, "rows": [rows[0].t, rows[1].t]},
        {"file": names[1], "platform": "youtube", "video_id": VIDEO_ID, "start": 77.0, "end": 84.0, "rows": [rows[2].t]},
        {"file": names[2], "platform": "youtube", "video_id": OTHER_VIDEO_ID, "start": 2.0, "end": 9.0, "rows": [rows[3].t]},
    ]}
    assert fetch.read_kept_videos(kept)[names[1]] == fetch.KeptVideo(
        file=names[1], platform="youtube", video_id=VIDEO_ID, start=77.0, end=84.0, rows=(rows[2].t,))


def test_a_kept_section_read_with_its_recorded_start_gives_each_row_s_frame(video, tmp_path, monkeypatch):
    """The start in the index is the offset the fetch itself read the file with: time ``x`` of
    the file is time ``start + x`` of the video."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    rows = [make_row(60), make_row(90), make_row(2400), make_row(2430)]
    kept = tmp_path / "kept"
    fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=StandInDownloader(), keep_video=kept, pause=0,
                       image_format="png", say=quiet)
    listed = fetch.read_kept_videos(kept)
    assert sorted(t for entry in listed.values() for t in entry.rows) == sorted(row.t for row in rows)
    for row, number in zip(rows, (60, 90, 2400, 2430)):
        (entry,) = [entry for entry in listed.values() if row.t in entry.rows]
        assert entry.start <= row.t <= entry.end
        frame = extract.read_frame_at(kept / entry.file, row.t, offset=entry.start)
        assert read_counter(frame.rgb) == number
        with Image.open(tmp_path / "out" / "frames" / extract.frame_file_name(row, ".png")) as saved:
            assert np.array_equal(np.asarray(saved), frame.rgb)       # the very frame the fetch wrote
        # Without the start, the same time of the file is another frame, or none.
        if entry.start > 0:
            with pytest.raises(extract.FrameNotFound):
                extract.read_frame_at(kept / entry.file, row.t)


def test_the_recorded_start_is_the_offset_the_downloader_reports(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    inner = StandInDownloader()

    def a_second_late(platform, video_id, start, end, folder):
        """A downloader whose file starts one second after the time asked for, and says so."""
        return inner(platform, video_id, start + 1.0, end, folder)

    row = make_row(2400)     # t = 80.017 s: asked for 77 s to 84 s, served from 78 s
    kept = tmp_path / "kept"
    summary = fetch.fetch_frames([row], tmp_path / "out", download=True, downloader=a_second_late, keep_video=kept, say=quiet)
    assert summary.complete and shown(tmp_path / "out" / "frames" / extract.frame_file_name(row)) == 2400
    assert [(entry.file, entry.start, entry.end) for entry in summary.kept] == [("youtube-EXAMPLE0001+78.mp4", 78.0, 84.0)]
    assert read_counter(extract.read_frame_at(kept / "youtube-EXAMPLE0001+78.mp4", row.t, offset=78.0).rgb) == 2400


def test_a_whole_video_is_kept_without_a_start(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    rows = [make_row(60), make_row(240), make_row(15, TWITCH_ID, platform="twitch")]
    kept = tmp_path / "kept"
    summary = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=StandInDownloader(), whole_video=True,
                                 keep_video=kept, pause=0, say=quiet)
    assert summary.complete
    assert kept_index(kept)["files"] == [
        {"file": "twitch-0000000001.mp4", "platform": "twitch", "video_id": TWITCH_ID, "start": None, "end": None, "rows": [rows[2].t]},
        {"file": "youtube-EXAMPLE0001.mp4", "platform": "youtube", "video_id": VIDEO_ID, "start": None, "end": None,
         "rows": [rows[0].t, rows[1].t]},
    ]
    assert read_counter(extract.read_frame_at(kept / "youtube-EXAMPLE0001.mp4", rows[1].t).rgb) == 240


def test_a_second_run_adds_to_the_kept_files_and_keeps_nothing_for_a_failed_download(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    rows = [make_row(60), make_row(2400), make_row(150, OTHER_VIDEO_ID)]
    kept = tmp_path / "kept"
    first = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=StandInDownloader(fail_for=VIDEO_ID),
                               keep_video=kept, pause=0, say=quiet)
    assert [entry.status for entry in first.entries] == ["failed", "failed", "ok"]
    assert [entry["file"] for entry in kept_index(kept)["files"]] == ["youtube-EXAMPLE0002+2.mp4"]
    again = fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=StandInDownloader(), keep_video=kept, pause=0, say=quiet)
    assert again.complete and again.skipped == 1 and [entry.file for entry in again.kept] == [
        "youtube-EXAMPLE0001+0.mp4", "youtube-EXAMPLE0001+77.mp4"]
    assert [(entry["file"], entry["rows"]) for entry in kept_index(kept)["files"]] == [
        ("youtube-EXAMPLE0001+0.mp4", [rows[0].t]), ("youtube-EXAMPLE0001+77.mp4", [rows[1].t]),
        ("youtube-EXAMPLE0002+2.mp4", [rows[2].t])]
    # A file fetched again replaces the one kept, and the rows it still holds stay listed.
    (tmp_path / "out" / "frames" / extract.frame_file_name(rows[0])).unlink()
    later = make_row(75)        # 2.517 s: inside the stretch of the first row
    fetch.fetch_frames([later], tmp_path / "out", download=True, downloader=StandInDownloader(), keep_video=kept, say=quiet)
    assert kept_index(kept)["files"][0] == {"file": "youtube-EXAMPLE0001+0.mp4", "platform": "youtube", "video_id": VIDEO_ID,
                                            "start": 0.0, "end": 6.0, "rows": [rows[0].t, later.t]}
    assert not list(kept.glob(".partial-*")) and len(list(kept.iterdir())) == 4


def test_a_second_run_reads_the_kept_files_and_asks_for_nothing(video, tmp_path, monkeypatch):
    """Rows that a kept file covers are read from it. A second run into a new folder of
    frames makes no request, and neither does a run without ``download``."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    rows = [make_row(60), make_row(90), make_row(2400), make_row(150, OTHER_VIDEO_ID)]
    kept = tmp_path / "kept"
    first = StandInDownloader()
    fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=first, keep_video=kept, pause=0, image_format="png", say=quiet)
    assert len(first.calls) == 3
    before = (kept / "video-files.json").read_bytes()
    stamps = {path.name: path.stat().st_mtime_ns for path in kept.iterdir()}

    second = StandInDownloader()
    said: list[str] = []
    again = fetch.fetch_frames(rows, tmp_path / "again", download=True, downloader=second, keep_video=kept, pause=0,
                               image_format="png", say=said.append)
    assert again.complete and second.calls == [] and again.kept == []
    assert [entry.source for entry in again.entries] == ["kept"] * 4 and all(entry.attempts is None for entry in again.entries)
    assert "youtube EXAMPLE0001: 2 frame(s) from the kept file youtube-EXAMPLE0001+0.mp4" in said
    for row in rows:
        name = extract.frame_file_name(row, ".png")
        assert (tmp_path / "again" / "frames" / name).read_bytes() == (tmp_path / "out" / "frames" / name).read_bytes()
    # Nothing in the folder of kept files was touched.
    assert (kept / "video-files.json").read_bytes() == before
    assert {path.name: path.stat().st_mtime_ns for path in kept.iterdir()} == stamps

    # Without download the kept files are still read: frames again, and no network.
    offline = fetch.fetch_frames(rows, tmp_path / "offline", keep_video=kept, say=quiet)
    assert offline.complete and [entry.source for entry in offline.entries] == ["kept"] * 4
    # A row no kept file covers is left without a video when nothing may be downloaded.
    other = make_row(4800)
    alone = fetch.fetch_frames([other, rows[0]], tmp_path / "alone", keep_video=kept, say=quiet)
    assert [(entry.t, entry.status) for entry in alone.entries] == [(rows[0].t, "ok"), (other.t, "no-video")]


def test_a_row_near_a_kept_section_but_not_listed_for_it_is_downloaded(video, tmp_path, monkeypatch):
    """A section is cut for the rows it lists. Another row whose time falls inside it may lack
    the second either side that the fingerprint rule reads, so it gets a part of its own."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    kept = tmp_path / "kept"
    fetch.fetch_frames([make_row(60)], tmp_path / "out", download=True, downloader=StandInDownloader(), keep_video=kept, say=quiet)
    near = make_row(75)          # 2.517 s: inside the kept section 0 to 6 s, and not one of its rows
    downloader = StandInDownloader()
    summary = fetch.fetch_frames([near], tmp_path / "out", download=True, downloader=downloader, keep_video=kept, say=quiet)
    assert summary.complete and len(downloader.calls) == 1 and summary.entries[0].source == "download"
    assert fetch.kept_file_for(fetch.kept_videos_of(kept, "youtube", VIDEO_ID), near.t).rows == (make_row(60).t, near.t)


def test_kept_files_are_known_by_the_index_or_by_their_name(video, tmp_path):
    kept = tmp_path / "kept"
    kept.mkdir()
    listed = fetch.KeptVideo(file="youtube-EXAMPLE0001+77.mp4", platform="youtube", video_id=VIDEO_ID, start=77.0, end=84.0,
                             rows=(80.017,))
    fetch.write_kept_videos(kept, {listed.file: listed, "gone.mp4": fetch.KeptVideo("gone.mp4", "youtube", VIDEO_ID, 0.0, 7.0, (2.0,))})
    for name in ("youtube-EXAMPLE0001+77.mp4", "youtube-EXAMPLE0001+300.webm", "youtube-EXAMPLE0001+12.5.mp4",
                 "youtube-EXAMPLE0001+late.mp4", "youtube-EXAMPLE0001+900.mp4.part", ".youtube-EXAMPLE0001+5.mp4",
                 "youtube-EXAMPLE00012+3.mp4", "notes.txt"):
        (kept / name).write_bytes(b"")
    found = fetch.kept_videos_of(kept, "youtube", VIDEO_ID)
    # The listed file that exists, then the files known by name; a file the index lists and the folder lacks is left out.
    assert [(entry.file, entry.start, entry.end) for entry in found] == [
        ("youtube-EXAMPLE0001+77.mp4", 77.0, 84.0), ("youtube-EXAMPLE0001+12.5.mp4", 12.5, None),
        ("youtube-EXAMPLE0001+300.webm", 300.0, None)]
    assert fetch.kept_file_for(found, 80.017) is found[0]
    assert fetch.kept_file_for(found, 81.0).file == "youtube-EXAMPLE0001+12.5.mp4"      # not listed for the indexed section
    assert fetch.kept_file_for(found, 400.0).file == "youtube-EXAMPLE0001+300.webm"     # the section that starts last before it
    assert fetch.kept_file_for(found, 5.0) is None and fetch.kept_file_for(found, 80.017, whole_only=True) is None
    (kept / "youtube-EXAMPLE0001.mkv").write_bytes(b"")
    whole = fetch.kept_videos_of(kept, "youtube", VIDEO_ID)
    assert fetch.kept_file_for(whole, 5.0).file == "youtube-EXAMPLE0001.mkv"            # a whole video covers every row
    assert fetch.kept_file_for(whole, 80.017, whole_only=True).start is None
    assert fetch.kept_videos_of(kept, "twitch", TWITCH_ID) == [] and fetch.kept_videos_of(tmp_path / "absent", "youtube", VIDEO_ID) == []
    assert fetch.has_kept_videos(kept) and not fetch.has_kept_videos(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("no video here")
    assert not fetch.has_kept_videos(empty)


def test_a_whole_video_file_placed_in_the_folder_serves_its_rows_by_its_name(video, tmp_path):
    """A file given the name of a kept whole video is read like one, with no index. That is
    also a way to hand in a recording of your own."""
    kept = tmp_path / "kept"
    kept.mkdir()
    shutil.copy(video, kept / "youtube-EXAMPLE0001.mp4")
    shutil.copy(video, kept / "local-kitchen-table-01.mp4")
    rows = [make_row(60), make_row(200), local_row(12)]
    summary = fetch.fetch_frames(rows, tmp_path / "out", keep_video=kept, say=quiet)
    assert summary.complete and [entry.source for entry in summary.entries] == ["kept"] * 3
    assert shown(tmp_path / "out" / "frames" / extract.frame_file_name(rows[1])) == 200
    assert shown(tmp_path / "out" / "frames" / extract.frame_file_name(rows[2])) == 12


def test_a_row_that_a_section_known_by_name_does_not_reach_has_no_video(video, tmp_path, monkeypatch):
    """A section file the index does not list is taken to cover the rows after its start. That
    is a guess. Without a download, a row far past the end of the file is a row without a
    video, not a row that failed."""
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    made = tmp_path / "made"
    fetch.fetch_frames([make_row(60)], tmp_path / "first", download=True, downloader=StandInDownloader(), keep_video=made, say=quiet)
    kept = tmp_path / "kept"
    kept.mkdir()
    shutil.copy(made / "youtube-EXAMPLE0001+0.mp4", kept / "youtube-EXAMPLE0001+0.mp4")      # 7 seconds of video, and no index
    near, far = make_row(60), make_row(4800)                                               # 2 s and 160 s
    said: list[str] = []
    summary = fetch.fetch_frames([near, far], tmp_path / "out", keep_video=kept, say=said.append)
    assert [(entry.t, entry.status, entry.source) for entry in summary.entries] == [(near.t, "ok", "kept"), (far.t, "no-video", None)]
    assert summary.entries[1].reason == fetch.NO_VIDEO_REASON and summary.kept_used == ["youtube-EXAMPLE0001+0.mp4"]
    assert "youtube EXAMPLE0001: the kept file youtube-EXAMPLE0001+0.mp4 does not reach the frame of 1 row(s)" in said
    # With a download the same row is asked for, as before.
    downloader = StandInDownloader()
    again = fetch.fetch_frames([far], tmp_path / "out", download=True, downloader=downloader, keep_video=kept, say=quiet)
    assert again.complete and downloader.calls == [("youtube", VIDEO_ID, 157.0, 164.0)]
    # A whole video is no guess: a time past its end is a failed row.
    whole = tmp_path / "whole"
    whole.mkdir()
    shutil.copy(video, whole / "youtube-EXAMPLE0001.mp4")
    past = fetch.fetch_frames([make_row(100_000)], tmp_path / "past", keep_video=whole, say=quiet)
    assert [(entry.status, entry.source) for entry in past.entries] == [("failed", "kept")]


def test_a_kept_file_that_does_not_give_the_frame_is_downloaded_again(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    kept = tmp_path / "kept"
    rows = [make_row(60), make_row(2400)]
    fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=StandInDownloader(), keep_video=kept, pause=0, say=quiet)
    (kept / "youtube-EXAMPLE0001+77.mp4").write_bytes(b"no longer a video")
    # Without download the row's own outcome is recorded.
    offline = fetch.fetch_frames(rows, tmp_path / "offline", keep_video=kept, say=quiet)
    assert [(entry.status, entry.source) for entry in offline.entries] == [("ok", "kept"), ("failed", "kept")]
    # With download the damaged file is replaced by a new one, and only that part is asked for.
    downloader = StandInDownloader()
    said: list[str] = []
    again = fetch.fetch_frames(rows, tmp_path / "again", download=True, downloader=downloader, keep_video=kept, pause=0,
                               say=said.append)
    assert again.complete and downloader.calls == [("youtube", VIDEO_ID, 77.0, 84.0)]
    assert sorted(entry.source for entry in again.entries) == ["download", "kept"]
    assert any("did not give the frame of 1 row(s); they are downloaded" in line for line in said)
    assert read_counter(extract.read_frame_at(kept / "youtube-EXAMPLE0001+77.mp4", rows[1].t, offset=77.0).rgb) == 2400


def test_whole_videos_asked_for_are_not_served_from_kept_sections(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    kept = tmp_path / "kept"
    rows = [make_row(60)]
    fetch.fetch_frames(rows, tmp_path / "out", download=True, downloader=StandInDownloader(), keep_video=kept, say=quiet)
    downloader = StandInDownloader()
    fetch.fetch_frames(rows, tmp_path / "whole", download=True, downloader=downloader, keep_video=kept, whole_video=True, say=quiet)
    assert downloader.calls == [("youtube", VIDEO_ID, None, None)]
    # The whole video is then kept, and serves every later run.
    later = StandInDownloader()
    summary = fetch.fetch_frames([make_row(60), make_row(250)], tmp_path / "later", download=True, downloader=later, keep_video=kept,
                                 whole_video=True, say=quiet)
    assert summary.complete and later.calls == [] and [entry.source for entry in summary.entries] == ["kept", "kept"]


def test_nothing_is_kept_unless_asked_and_a_foreign_index_is_refused(video, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    summary = fetch.fetch_frames([make_row(60)], tmp_path / "out", download=True, downloader=StandInDownloader(), say=quiet)
    assert summary.complete and summary.kept == []
    assert fetch.read_kept_videos(tmp_path / "absent") == {}
    other = tmp_path / "other"
    other.mkdir()
    for text in ("[]", '{"kind": "something-else", "version": 1, "files": []}', '{"kind": "cardvision-video-files", "version": 2}',
                 '{"kind": "cardvision-video-files", "version": 1, "files": [{"file": "a.mp4"}]}', "not json"):
        (other / "video-files.json").write_text(text)
        with pytest.raises(ValueError, match="not an index of kept video files"):
            fetch.read_kept_videos(other)


# ------------------------------------------------------------------------- the command line
def write_rows(path: Path, rows: list[Row]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_manifest(path, rows)
    return path


def test_the_command_fetches_from_a_local_file(video, tmp_path, capsys):
    rows = [make_row(10), make_row(20, role="evaluation"), make_row(30, OTHER_VIDEO_ID, role="evaluation")]
    path = write_rows(tmp_path / "labels" / "m.jsonl", rows)
    out = tmp_path / "frames-out"
    arguments = ["fetch", str(path), "--out", str(out), "--video-file", f"{VIDEO_ID}={video}"]
    assert cli.main(arguments) == 1      # one video has no file
    captured = capsys.readouterr()
    assert "2 frame(s) written, 0 already present, 0 unusable, 0 failed, 1 without a video" in captured.out
    assert "--video-file ID=PATH, or pass --download" in captured.err
    assert "your own act" not in captured.err      # no download was asked for
    assert shown(out / "frames" / extract.frame_file_name(rows[1])) == 20

    assert cli.main([*arguments, "--rows", "1-2"]) == 0
    assert "0 frame(s) written, 2 already present" in capsys.readouterr().out
    assert cli.main([*arguments, "--role", "evaluation", "--video-file", f"{OTHER_VIDEO_ID}={video}", "--format", "png"]) == 0
    assert "1 frame(s) written, 1 already present" in capsys.readouterr().out
    lines = [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines()]
    assert [line["status"] for line in lines] == ["ok", "ok", "no-video", "ok"]


def test_the_command_says_how_to_give_a_recording_of_your_own(video, tmp_path, capsys, monkeypatch):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [local_row(10), local_row(20, "second-game"), make_row(30)])
    out = tmp_path / "frames-out"
    downloader = StandInDownloader()
    monkeypatch.setattr(fetch, "downloader_installed", lambda: True)
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    monkeypatch.setattr(fetch, "download_video", downloader)
    arguments = ["fetch", str(path), "--out", str(out)]
    assert cli.main([*arguments, "--download", "--pause", "0"]) == 1
    captured = capsys.readouterr()
    assert "1 frame(s) written, 0 already present, 0 unusable, 0 failed, 2 without a video" in captured.out
    assert ('Rows of a recording of your own (platform "local") without a video: give the file with --video-file ID=PATH. '
            "--download does not apply to them.") in captured.err
    assert "or pass --download" not in captured.err         # every row a download could serve has its frame
    assert [call[:2] for call in downloader.calls] == [("youtube", VIDEO_ID)]
    assert cli.main([*arguments, "--video-file", f"{LOCAL_ID}={video}", "--video-file", f"local:second-game={video}"]) == 0
    assert "2 frame(s) written, 1 already present" in capsys.readouterr().out
    assert shown(out / "frames" / "local-second-game-t0000000683.jpg") == 20


def test_the_command_refuses_to_write_frames_into_the_manifest_folder(video, tmp_path, capsys):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [make_row(10)])
    for out in (tmp_path / "labels", tmp_path / "labels" / "frames"):
        assert cli.main(["fetch", str(path), "--out", str(out), "--video-file", f"{VIDEO_ID}={video}"]) == 1
        assert "holds text only" in capsys.readouterr().err
    assert sorted(p.name for p in (tmp_path / "labels").iterdir()) == ["m.jsonl"]


def test_the_command_refuses_bad_arguments(tmp_path, capsys):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [make_row(10)])
    out = str(tmp_path / "out")
    assert cli.main(["fetch", str(path), "--out", out, "--rows", "5"]) == 1
    assert "rows are numbered 1 to 1" in capsys.readouterr().err
    assert cli.main(["fetch", str(path), "--out", out, "--video-file", "EXAMPLE0001"]) == 1
    assert "expected ID=PATH" in capsys.readouterr().err
    assert cli.main(["fetch", str(path), "--out", out, "--role", "train"]) == 1
    assert "no row is selected" in capsys.readouterr().err
    assert cli.main(["fetch", str(tmp_path / "absent.jsonl"), "--out", out]) == 1
    capsys.readouterr()


def test_the_command_says_what_to_install_when_the_downloader_is_missing(tmp_path, capsys, monkeypatch):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [make_row(10)])
    monkeypatch.setattr(fetch, "downloader_installed", lambda: False)
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "out"), "--download"]) == 1
    assert 'python -m pip install -e ".[download]"' in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_the_command_prints_the_notice_before_any_download(video, tmp_path, capsys, monkeypatch):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [make_row(60)])
    downloader = StandInDownloader()
    monkeypatch.setattr(fetch, "downloader_installed", lambda: True)
    monkeypatch.setattr(fetch, "sections_supported", lambda: shutil.which("no-such-program") is not None)
    monkeypatch.setattr(fetch, "download_video", downloader)
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "out"), "--download", "--pause", "0"]) == 0
    captured = capsys.readouterr()
    assert captured.err.index("Downloading is your own act") < captured.err.index("downloading 1 part(s)")
    assert "ffmpeg was not found, so each video is downloaded whole" in captured.err
    assert downloader.calls == [("youtube", VIDEO_ID, None, None)]
    assert "1 frame(s) written" in captured.out


def test_the_command_keeps_downloaded_files_when_asked(video, tmp_path, capsys, monkeypatch):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [make_row(60), make_row(2400)])
    monkeypatch.setattr(fetch, "downloader_installed", lambda: True)
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    monkeypatch.setattr(fetch, "download_video", StandInDownloader())
    kept = tmp_path / "kept"
    arguments = ["fetch", str(path), "--out", str(tmp_path / "out"), "--download", "--pause", "0", "--keep-video", str(kept)]
    assert cli.main(arguments) == 0
    captured = capsys.readouterr()
    assert f"2 new video file(s) kept in {kept}; index: {kept / 'video-files.json'}" in captured.out
    assert sorted(p.name for p in kept.iterdir()) == ["video-files.json", "youtube-EXAMPLE0001+0.mp4", "youtube-EXAMPLE0001+77.mp4"]
    assert cli.main(arguments) == 0      # every row has its frame: nothing is downloaded and no kept file is read
    assert f"0 new video file(s) kept in {kept}; index:" in capsys.readouterr().out
    assert len(json.loads((kept / "video-files.json").read_text())["files"]) == 2
    # Frames into another folder come from the two kept files, and the summary says so.
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "again"), "--download", "--pause", "0", "--keep-video", str(kept)]) == 0
    assert (f"0 new video file(s) kept in {kept}; 2 already there were used; index: {kept / 'video-files.json'}"
            in capsys.readouterr().out)


def test_the_command_stops_with_its_own_exit_code_when_the_platform_refuses(video, tmp_path, capsys, monkeypatch):
    from cardvision.data import commands

    rows = [make_row(60), make_row(2400), make_row(150, OTHER_VIDEO_ID)]
    path = write_rows(tmp_path / "labels" / "m.jsonl", rows)
    downloader = Scripted({2: REFUSALS[0]})
    monkeypatch.setattr(fetch, "downloader_installed", lambda: True)
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    monkeypatch.setattr(fetch, "download_video", downloader)
    arguments = ["fetch", str(path), "--out", str(tmp_path / "out"), "--download", "--pause", "0"]
    assert cli.main(arguments) == commands.EXIT_REFUSED == 3
    captured = capsys.readouterr()
    assert "1 frame(s) written, 0 already present, 0 unusable, 0 failed, 0 without a video" in captured.out
    assert "The platform refused a download, as a rate limit or as a check for automated access:" in captured.err
    assert "HTTP Error 429: Too Many Requests" in captured.err
    assert "The run stopped at once. 2 row(s) were not tried and have no line in the index." in captured.err
    assert "Wait before you run the command again" in captured.err
    assert len(downloader.calls) == 2
    # An ordinary failure keeps the exit code 1, and the run goes on to the next video.
    ordinary = Scripted({1: ORDINARY[2], 2: ORDINARY[2]})
    monkeypatch.setattr(fetch, "download_video", ordinary)
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "ordinary"), "--download", "--pause", "0"]) == 1
    captured = capsys.readouterr()
    assert "2 frame(s) written, 0 already present, 0 unusable, 1 failed" in captured.out and "refused" not in captured.err
    assert "Unable to download API page: timed out" in captured.err
    # --retries 0 asks once.
    once = Scripted({1: "ERROR: ffmpeg exited with code 8"})
    monkeypatch.setattr(fetch, "download_video", once)
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "once"), "--download", "--pause", "0", "--retries", "0",
                     "--rows", "1"]) == 1
    assert len(once.calls) == 1 and "0 frame(s) written, 0 already present, 0 unusable, 1 failed" in capsys.readouterr().out
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "bad"), "--download", "--retries", "-1"]) == 1
    assert "--retries is 0 or more" in capsys.readouterr().err


def test_the_command_stops_when_downloads_fail_one_after_another(video, tmp_path, capsys, monkeypatch):
    from cardvision.data import commands

    rows = [*(make_row(index) for index in FIVE_PARTS), make_row(150, OTHER_VIDEO_ID)]
    path = write_rows(tmp_path / "labels" / "m.jsonl", rows)
    always = {number: FFMPEG for number in range(1, 30)}
    monkeypatch.setattr(fetch, "downloader_installed", lambda: True)
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    downloader = Scripted(always)
    monkeypatch.setattr(fetch, "download_video", downloader)
    base = ["fetch", str(path), "--download", "--pause", "0"]
    assert cli.main([*base, "--out", str(tmp_path / "out")]) == commands.EXIT_STOPPED == commands.EXIT_REFUSED == 3
    captured = capsys.readouterr()
    assert len(downloader.calls) == 6
    assert "0 frame(s) written, 0 already present, 0 unusable, 3 failed, 0 without a video" in captured.out
    assert "3 downloads in a row failed. This may be a rate limit or a network problem." in captured.err
    assert "The run stopped. 3 row(s) were not tried and have no line in the index." in captured.err
    assert "Check the connection, and wait before you run the command again." in captured.err
    assert "--stop-after-failures N sets how many failures stop a run (0: never)." in captured.err
    assert "The platform refused" not in captured.err
    # Another limit, and none at all: with 0 the run goes to its end, and its exit code is 1.
    two = Scripted(always)
    monkeypatch.setattr(fetch, "download_video", two)
    assert cli.main([*base, "--out", str(tmp_path / "two"), "--stop-after-failures", "2", "--retries", "0"]) == 3
    assert len(two.calls) == 2 and "2 downloads in a row failed." in capsys.readouterr().err
    never = Scripted(always)
    monkeypatch.setattr(fetch, "download_video", never)
    assert cli.main([*base, "--out", str(tmp_path / "never"), "--stop-after-failures", "0", "--retries", "0"]) == 1
    captured = capsys.readouterr()
    assert len(never.calls) == 6 and "6 failed" in captured.out and "in a row failed" not in captured.err
    assert cli.main([*base, "--out", str(tmp_path / "bad"), "--stop-after-failures", "-1"]) == 1
    assert "--stop-after-failures is 0 or more" in capsys.readouterr().err


def test_the_command_says_what_to_install_when_certificates_are_missing(video, tmp_path, capsys, monkeypatch):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [make_row(60)])

    def no_certificates(platform, video_id, start, end, folder):
        raise fetch.CertificatesMissing(f"{platform} {video_id}: {fetch.CERTIFICATE_HELP}")

    monkeypatch.setattr(fetch, "downloader_installed", lambda: True)
    monkeypatch.setattr(fetch, "download_video", no_certificates)
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "out"), "--download"]) == 1
    err = capsys.readouterr().err
    assert "cardvision fetch: failed" in err and "could not be verified" in err and "certifi" in err
    assert 'python -m pip install -e ".[download]" in the repository folder' in err


def test_the_command_says_when_frames_have_another_size_than_their_rows(video, tmp_path, capsys):
    rows = [make_row(10), make_row(20), make_row(30, width=320, height=180), make_row(40, OTHER_VIDEO_ID, width=640, height=360)]
    path = write_rows(tmp_path / "labels" / "m.jsonl", rows)
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "out"), "--video-file", f"{VIDEO_ID}={video}",
                     "--video-file", f"{OTHER_VIDEO_ID}={video}"]) == 0
    out = capsys.readouterr().out
    assert "4 frame(s) written, 0 already present, 0 unusable, 0 failed, 0 without a video" in out
    assert ("3 of the frame(s) written have another size than their row records (other_size in the index). "
            "The shape is the same, so the labels fit:") in out
    assert "  2 frame(s) of 320 x 180, labeled at 1920 x 1080" in out and "  1 frame(s) of 320 x 180, labeled at 640 x 360" in out
    # Rows of the video's own size say nothing of the kind.
    same = write_rows(tmp_path / "same" / "m.jsonl", [make_row(30, width=320, height=180)])
    assert cli.main(["fetch", str(same), "--out", str(tmp_path / "same-out"), "--video-file", f"{VIDEO_ID}={video}"]) == 0
    assert "another size" not in capsys.readouterr().out


def test_the_command_reads_kept_files_without_a_download(video, tmp_path, capsys, monkeypatch):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [make_row(60), make_row(2400)])
    downloader = StandInDownloader()
    monkeypatch.setattr(fetch, "downloader_installed", lambda: True)
    monkeypatch.setattr(fetch, "sections_supported", lambda: True)
    monkeypatch.setattr(fetch, "download_video", downloader)
    kept = tmp_path / "kept"
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "out"), "--download", "--pause", "0", "--keep-video", str(kept)]) == 0
    capsys.readouterr()
    assert len(downloader.calls) == 2
    # Frames again, as PNG, from the kept files alone: no --download, and no request.
    assert cli.main(["fetch", str(path), "--out", str(tmp_path / "png"), "--format", "png", "--keep-video", str(kept)]) == 0
    captured = capsys.readouterr()
    assert "2 frame(s) written, 0 already present" in captured.out
    assert f"0 new video file(s) kept in {kept}; 2 already there were used; index:" in captured.out
    assert "from the kept file youtube-EXAMPLE0001+77.mp4" in captured.err and "your own act" not in captured.err
    assert len(downloader.calls) == 2
    assert shown(tmp_path / "png" / "frames" / extract.frame_file_name(make_row(2400), ".png")) == 2400


def test_the_command_refuses_a_bad_folder_for_kept_files(video, tmp_path, capsys, monkeypatch):
    path = write_rows(tmp_path / "labels" / "m.jsonl", [make_row(60)])
    downloader = StandInDownloader()
    monkeypatch.setattr(fetch, "downloader_installed", lambda: True)
    monkeypatch.setattr(fetch, "download_video", downloader)
    base = ["fetch", str(path), "--out", str(tmp_path / "out")]
    assert cli.main([*base, "--download", "--keep-video", str(tmp_path / "labels" / "videos")]) == 1
    assert "holds text only" in capsys.readouterr().err
    assert cli.main([*base, "--keep-video", str(tmp_path / "kept")]) == 1
    assert "pass --download as well" in capsys.readouterr().err
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "video-files.json").write_text("{}")
    assert cli.main([*base, "--download", "--keep-video", str(foreign)]) == 1
    assert "not an index of kept video files" in capsys.readouterr().err
    assert downloader.calls == [] and not (tmp_path / "out").exists() and not (tmp_path / "kept").exists()
    assert sorted(p.name for p in (tmp_path / "labels").iterdir()) == ["m.jsonl"]
