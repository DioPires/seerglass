# SPDX-License-Identifier: Apache-2.0
"""The frame fingerprint: the hash itself, its place in the manifest format, and fetching a
frame by content. Frames and videos are generated here; nothing is downloaded. Tests that read
video are skipped when OpenCV is not installed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cardvision import cli
from cardvision.data import extract, fetch, manifest, schema
from cardvision.data.dhash import DHASH_PATTERN, SAME_FRAME_MAX_HAMMING, dhash, hamming, is_dhash
from cardvision.data.schema import Row, row_problems
from cardvision.data.validate import validate_manifests
from support_data import VIDEO_ID, row_doc, write_lines

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data" / "example.manifest.jsonl"
RATE = 30.0


# ------------------------------------------------------------------- generated pictures
def horizontal_ramp(width: int = 90, height: int = 80) -> np.ndarray:
    columns = (np.arange(width) * 255 // (width - 1)).astype(np.uint8)
    return np.broadcast_to(columns[None, :, None], (height, width, 3)).copy()


def stripes(width: int = 64, height: int = 48) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    v = (x * 37 + y * 17) % 256
    return np.stack([v, (v * 2) % 256, (v * 3) % 256], axis=-1).astype(np.uint8)


def diagonal(width: int = 71, height: int = 99) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    v = ((x * 3 + y * 5) % 64) * 4
    return np.stack([v, 255 - v, (v * 7) % 256], axis=-1).astype(np.uint8)


def scene_frame(seed: int, width: int = 320, height: int = 180) -> np.ndarray:
    """A picture of 9 x 8 flat patches whose brightness depends on ``seed``. Each row of patches
    is a permutation of nine well-spaced levels, so no patch is nearly as bright as its
    neighbor and compression cannot flip a bit; pictures of different seeds are unrelated."""
    rng = np.random.default_rng(seed)
    levels = np.array([20 + 25 * i for i in range(9)])
    grid = np.stack([rng.permutation(levels) for _ in range(8)])
    rows = np.minimum(7, (np.arange(height) * 8) // height)
    columns = np.minimum(8, (np.arange(width) * 9) // width)
    gray = grid[rows[:, None], columns[None, :]].astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


# the picture that the example manifest's fingerprint was computed from
EXAMPLE_SEED = 7
EXAMPLE_SIZE = (1920, 1080)


# --------------------------------------------------------------------------- the hash
def test_the_hash_is_stable_and_matches_fixed_values():
    assert dhash(stripes()) == "f88f738cf33cc779"
    assert dhash(diagonal()) == "29d2add25aa45b4b"
    assert dhash(stripes()) == dhash(stripes().copy())
    assert dhash(horizontal_ramp()) == "ffffffffffffffff"            # every patch brighter than the one before
    assert dhash(horizontal_ramp()[:, ::-1]) == "0000000000000000"   # every patch darker
    assert dhash(np.full((40, 60, 3), 128, np.uint8)) == "0000000000000000"   # no patch brighter than another


def test_the_hash_is_sixteen_lower_case_hex_digits():
    for picture in (stripes(), diagonal(), horizontal_ramp(), scene_frame(1)):
        assert is_dhash(dhash(picture)) and len(dhash(picture)) == 16


def test_the_hash_reads_gray_in_the_documented_weights():
    # Red and blue patches of different weight: red 0.299 is brighter than blue 0.114.
    picture = np.zeros((8, 9, 3), np.uint8)
    picture[:, 0] = (255, 0, 0)
    picture[:, 1] = (0, 0, 255)
    picture[:, 2] = (0, 255, 0)
    bits = int(dhash(picture), 16)
    first_row = (bits >> 56) & 0xFF   # row 0, cells 0..7, first cell the most significant bit
    # red (76) > blue (29): 0 for cell 0; blue < green (150): 1 for cell 1; green > black: 0 after
    assert format(first_row, "08b") == "01000000"


def test_the_hash_is_the_same_at_other_resolutions_and_after_compression(tmp_path):
    large = scene_frame(3, 1920, 1080)
    small = scene_frame(3, 320, 180)
    assert hamming(dhash(large), dhash(small)) == 0
    path = tmp_path / "frame.jpg"
    Image.fromarray(large).save(path, format="JPEG", quality=60)
    with Image.open(path) as saved:
        assert hamming(dhash(large), dhash(np.asarray(saved.convert("RGB")))) <= SAME_FRAME_MAX_HAMMING


def test_the_hash_refuses_what_is_not_a_color_frame():
    for bad in (np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8), np.zeros((10, 10, 3), np.float32),
                np.zeros((0, 10, 3), np.uint8), np.zeros((10, 0, 3), np.uint8)):
        with pytest.raises(ValueError):
            dhash(bad)


def test_pictures_one_bit_and_many_bits_apart():
    base = dhash(stripes())
    one_bit = f"{int(base, 16) ^ (1 << 20):016x}"
    assert hamming(base, base) == 0
    assert hamming(base, one_bit) == 1
    assert hamming(one_bit, base) == 1
    assert hamming("0" * 16, "f" * 16) == 64
    assert hamming(base, f"{int(base, 16) ^ 0xFF:016x}") == 8
    # unrelated pictures differ in many bits
    distances = [hamming(dhash(scene_frame(a)), dhash(scene_frame(b))) for a in range(6) for b in range(a + 1, 6)]
    assert min(distances) > SAME_FRAME_MAX_HAMMING
    assert SAME_FRAME_MAX_HAMMING == 6


def test_hamming_refuses_a_fingerprint_in_the_wrong_form():
    for bad in ("", "abc", "0" * 15, "0" * 17, "F" * 16, "g" * 16, 12):
        with pytest.raises(ValueError):
            hamming(bad, "0" * 16)
        with pytest.raises(ValueError):
            hamming("0" * 16, bad)


# ------------------------------------------------------------------ the manifest format
VALID = "0123456789abcdef"


def test_the_written_form_is_accepted():
    assert row_problems(row_doc(attributes={"frame_dhash": VALID})) == []
    assert row_problems(row_doc(attributes={"frame_dhash": "0" * 16, "camera_angle": "steep"})) == []
    row = schema.parse_row(row_doc(attributes={"frame_dhash": VALID}))
    assert row.frame_dhash == VALID and row.to_json()["attributes"] == {"frame_dhash": VALID}
    assert schema.parse_row(row_doc()).frame_dhash is None


@pytest.mark.parametrize("value", [
    VALID[:-1], VALID + "0", "", VALID.upper(), "0123456789ABCDEF", "0123456789abcdeF", "0123456789abcdeg",
    "0123456789abcde-", "0123456789abcd e", " 0123456789abcde", "0123456789abcde\n", "0x23456789abcdef",
    "٠١٢٣٤٥٦٧٨٩abcdef", 1234567890123456, 1.5, True, None, ["0123456789abcdef"],
])
def test_any_other_form_is_refused(value):
    problems = row_problems(row_doc(attributes={"frame_dhash": value}))
    assert problems and "frame_dhash" in problems[0]


def test_the_pattern_in_the_schema_is_the_one_the_hash_uses():
    spec = schema.ROW_SCHEMA["properties"]["attributes"]["properties"]["frame_dhash"]
    assert spec["pattern"] == DHASH_PATTERN and spec["type"] == "string" and "description" in spec


def test_the_fingerprint_rule_applies_to_rows_only():
    # on a box the key is just another attribute (the generic rule); it is the row's that is fixed
    assert "frame_dhash" not in schema.BOX_SCHEMA["properties"]["attributes"].get("properties", {})


def test_the_built_in_checker_agrees_with_a_json_schema_library_on_fingerprints():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(schema.ROW_SCHEMA)
    for value in (VALID, VALID.upper(), VALID[:-1], VALID + "0", "g" * 16, 5, None, True):
        candidate = row_doc(attributes={"frame_dhash": value})
        assert (not schema.schema_errors(candidate, schema.ROW_SCHEMA)) == validator.is_valid(candidate), value


def test_the_schema_document_explains_the_fingerprint():
    text = (ROOT / "data" / "SCHEMA.md").read_text(encoding="utf-8")
    for needle in ("`frame_dhash`", "16 lower-case hexadecimal digits", "9 columns by 8 rows", "1 second",
                   "6 bits", "`fingerprint`", "not the frame"):
        assert needle in text, needle


def test_the_example_manifest_fingerprint_is_that_of_a_generated_frame(tmp_path):
    rows = manifest.read_manifest(EXAMPLE)
    carrying = [row for row in rows if row.frame_dhash is not None]
    assert len(carrying) == 1
    width, height = EXAMPLE_SIZE
    assert (carrying[0].frame_width, carrying[0].frame_height) == (width, height)
    assert carrying[0].frame_dhash == dhash(scene_frame(EXAMPLE_SEED, width, height))
    assert validate_manifests([EXAMPLE]).ok


# ------------------------------------------------------------------- fetching by content
def make_row(index: int, fingerprint: str | None, width: int = 1920, height: int = 1080) -> Row:
    """A row whose time is that of frame ``index`` of a 30 frames-a-second video."""
    attributes = {} if fingerprint is None else {"frame_dhash": fingerprint}
    return Row(platform="youtube", video_id=VIDEO_ID, t=extract.mid_frame_time(index, RATE), frame_width=width,
               frame_height=height, source_type="physical", group="g-test-01", role="extra", trained_on=False,
               completeness="exhaustive-boxes", credit="Example Labeler", boxes=(), attributes=attributes)


def labeled_fingerprint(number: int) -> str:
    """The fingerprint taken on the labeled frame: a larger picture of the video's frame."""
    return dhash(scene_frame(number, 1920, 1080))


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path_factory.mktemp("video") / "scenes.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), RATE, (320, 180))
    assert writer.isOpened()
    for index in range(240):
        writer.write(np.ascontiguousarray(scene_frame(index)[:, :, ::-1]))
    writer.release()
    return path


def quiet(_: str) -> None:
    pass


def run(rows, tmp_path, video):
    summary = fetch.fetch_frames(rows, tmp_path / "out", video_files={VIDEO_ID: video}, say=quiet)
    return summary, extract.read_index(tmp_path / "out")


def kept_fingerprint(tmp_path, row) -> str:
    with Image.open(tmp_path / "out" / "frames" / extract.frame_file_name(row)) as image:
        return dhash(np.asarray(image.convert("RGB")))


@pytest.mark.parametrize("shift", [0, 1, -1, 7, -12, 29, -29, 30, -30])
def test_the_labeled_frame_is_found_when_the_video_is_off_by_up_to_a_second(video, tmp_path, shift):
    labeled = 100 + shift       # the frame that carries the labels
    row = make_row(100, labeled_fingerprint(labeled))   # but the row's time names frame 100
    summary, index = run([row], tmp_path, video)
    entry = index[row.key]
    assert summary.complete and entry.status == "ok" and entry.source == "local"
    assert entry.distance is not None and entry.distance <= SAME_FRAME_MAX_HAMMING
    assert entry.frame_time == pytest.approx(labeled / RATE, abs=2e-3)
    assert entry.offset == pytest.approx(labeled / RATE - row.t, abs=2e-3)
    assert hamming(kept_fingerprint(tmp_path, row), labeled_fingerprint(labeled)) <= SAME_FRAME_MAX_HAMMING
    assert hamming(kept_fingerprint(tmp_path, row), labeled_fingerprint(100)) > SAME_FRAME_MAX_HAMMING or shift == 0
    assert entry.file == f"frames/{extract.frame_file_name(row)}"


def test_the_frame_is_not_found_beyond_a_second(video, tmp_path):
    for shift in (31, -31, 60):
        row = make_row(100, labeled_fingerprint(100 + shift))
        out = tmp_path / f"out{shift}"
        summary = fetch.fetch_frames([row], out, video_files={VIDEO_ID: video}, say=quiet)
        entry = extract.read_index(out)[row.key]
        assert not summary.complete and entry.status == "unusable" and entry.reason == "fingerprint"
        assert entry.file is None and entry.distance > SAME_FRAME_MAX_HAMMING
        assert not (out / "frames" / extract.frame_file_name(row)).exists()


def test_content_that_differs_is_reported_as_a_fingerprint_failure(video, tmp_path):
    row = make_row(100, labeled_fingerprint(5000))     # a frame this video does not hold
    summary, index = run([row], tmp_path, video)
    entry = index[row.key]
    assert entry.status == "unusable" and entry.reason == "fingerprint" and entry.file is None
    assert entry.offset is None and entry.distance > SAME_FRAME_MAX_HAMMING
    assert summary.count("unusable") == 1
    assert not list((tmp_path / "out").glob("frames/*"))
    assert extract.FrameStore(tmp_path / "out").has(row) is False


def test_the_index_line_of_a_fingerprint_row_carries_the_distance_and_the_offset(video, tmp_path):
    row = make_row(100, labeled_fingerprint(110))
    run([row], tmp_path, video)
    line = json.loads((tmp_path / "out" / "index.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["status"] == "ok" and line["distance"] == 0 and line["offset"] == pytest.approx(110 / RATE - row.t, abs=2e-3)
    again = extract.IndexEntry.from_json(line)
    assert again.to_json() == line


def test_a_row_without_a_fingerprint_is_handled_as_before(video, tmp_path):
    row = make_row(100, None)
    summary, index = run([row], tmp_path, video)
    entry = index[row.key]
    assert entry.status == "ok" and entry.distance is None and entry.offset is None
    assert entry.frame_time == pytest.approx(100 / RATE, abs=2e-3)
    assert "distance" not in entry.to_json() and "offset" not in entry.to_json()
    assert hamming(kept_fingerprint(tmp_path, row), labeled_fingerprint(100)) <= SAME_FRAME_MAX_HAMMING


def test_the_closest_of_two_candidates_wins_and_a_tie_goes_to_the_nearer_time(video, tmp_path):
    row = make_row(100, labeled_fingerprint(100))
    _, index = run([row], tmp_path, video)
    assert index[row.key].offset == pytest.approx(-0.5 / RATE, abs=2e-3) and index[row.key].distance == 0

    frames = [extract.VideoFrame(rgb=scene_frame(1), index=i, time=t) for i, t in enumerate((9.0, 10.0, 11.0))]
    frame, distance = extract.nearest_frame(frames, dhash(scene_frame(1)), 10.2)
    assert distance == 0 and frame.time == 10.0                     # equal distances: nearest to t
    frame, _ = extract.nearest_frame(frames, dhash(scene_frame(1)), 10.0)
    assert frame.time == 10.0
    frame, _ = extract.nearest_frame(frames, dhash(scene_frame(1)), 10.5)
    assert frame.time == 10.0                                       # 10.0 and 11.0 are as near; the earlier wins
    with pytest.raises(ValueError):
        extract.nearest_frame([], VALID, 0.0)


def test_the_shape_check_still_applies_to_a_row_with_a_fingerprint(video, tmp_path):
    row = make_row(100, labeled_fingerprint(100), width=1440, height=1080)     # a 4:3 row, a 16:9 video
    _, index = run([row], tmp_path, video)
    entry = index[row.key]
    assert entry.status == "unusable" and entry.reason != "fingerprint" and "shapes differ" in entry.reason
    assert not list((tmp_path / "out").glob("frames/*"))


def test_a_window_at_the_start_of_the_video_uses_the_frames_that_exist(video, tmp_path):
    row = make_row(3, labeled_fingerprint(10))
    _, index = run([row], tmp_path, video)
    assert index[row.key].status == "ok" and index[row.key].frame_time == pytest.approx(10 / RATE, abs=2e-3)


def test_frames_between_gives_the_frames_on_screen_in_the_window(video):
    with extract.VideoReader(video) as reader:
        frames = reader.frames_between(2.0, 3.0)
        assert [frame.index for frame in frames] == list(range(60, 91))
        assert frames[0].time == pytest.approx(2.0, abs=2e-3)
        between = reader.frames_between(2.02, 2.2)       # starts in the middle of frame 60
        assert [frame.index for frame in between] == list(range(60, 67))
    with extract.VideoReader(video, exact=True) as reader:
        assert [frame.index for frame in reader.frames_between(2.0, 3.0)] == list(range(60, 91))
    with extract.VideoReader(video) as reader:
        assert [frame.index for frame in reader.frames_between(7.5, 12.0)] == list(range(225, 240))
        with pytest.raises(extract.FrameNotFound):
            reader.frames_between(20.0, 21.0)


def test_a_downloaded_stretch_is_matched_in_the_same_way(video, tmp_path):
    """The stretch a downloader cuts is a file that starts part-way into the video."""
    cv2 = pytest.importorskip("cv2")
    section = tmp_path / "section.mp4"
    writer = cv2.VideoWriter(str(section), cv2.VideoWriter_fourcc(*"mp4v"), RATE, (320, 180))
    for index in range(60, 180):
        writer.write(np.ascontiguousarray(scene_frame(index)[:, :, ::-1]))
    writer.release()
    row = make_row(120, labeled_fingerprint(135))
    downloader = lambda platform, video_id, start, end, folder: fetch.Download(path=section, offset=2.0)  # noqa: E731
    out = tmp_path / "out"
    summary = fetch.fetch_frames([row], out, download=True, downloader=downloader, say=quiet, pause=0)
    entry = extract.read_index(out)[row.key]
    assert summary.complete and entry.status == "ok" and entry.source == "download" and entry.distance == 0
    assert entry.frame_time == pytest.approx(135 / RATE, abs=2e-3)


def test_the_command_names_a_fingerprint_failure(video, tmp_path, capsys):
    rows = [make_row(100, labeled_fingerprint(5000))]
    path = write_lines(tmp_path / "m" / "rows.jsonl", [rows[0].to_json()])
    code = cli.main(["fetch", str(path), "--out", str(tmp_path / "frames"), "--video-file", f"{VIDEO_ID}={video}"])
    assert code == 1
    assert "looks like the labeled one" in capsys.readouterr().err
