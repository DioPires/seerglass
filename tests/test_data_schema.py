# SPDX-License-Identifier: Apache-2.0
"""The label manifest format: the schema, the row objects and reading and writing."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cardvision.data import manifest, schema
from cardvision.data.schema import Box, ManifestError, Point, Row, row_problems
from support_data import (
    OTHER_VIDEO_ID,
    TWITCH_ID,
    VIDEO_ID,
    box_doc,
    plain_box_doc,
    point_doc,
    points_row_doc,
    row_doc,
    write_lines,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data" / "example.manifest.jsonl"


# ------------------------------------------------------------------------------ the schema
def test_the_schema_file_is_a_copy_of_the_schema_in_code():
    assert (ROOT / "data" / "manifest.schema.json").read_text(encoding="utf-8") == schema.schema_json()


def test_every_field_of_the_schema_has_a_description():
    def undescribed(properties: dict, where: str) -> list[str]:
        return [f"{where}.{name}" for name, spec in properties.items() if "description" not in spec]

    missing = (undescribed(schema.ROW_SCHEMA["properties"], "row")
               + undescribed(schema.BOX_SCHEMA["properties"], "box")
               + undescribed(schema.POINT_SCHEMA["properties"], "point"))
    assert missing == []


def test_every_field_is_explained_in_the_schema_document():
    text = (ROOT / "data" / "SCHEMA.md").read_text(encoding="utf-8")
    names = set(schema.ROW_SCHEMA["properties"]) | set(schema.BOX_SCHEMA["properties"]) | set(schema.POINT_SCHEMA["properties"])
    assert [name for name in sorted(names) if f"`{name}`" not in text] == []
    values = (*schema.PLATFORMS, *schema.ROLES, *schema.COMPLETENESS, *schema.KINDS, *schema.IDENTITIES,
              *schema.READABLE, *schema.SHAPES, *schema.SOURCE_TYPES)
    assert [value for value in values if f"`{value}`" not in text] == []


def test_the_example_manifest_is_valid_and_shows_every_kind_of_label():
    rows = manifest.read_manifest(EXAMPLE)
    boxes = [box for row in rows for box in row.boxes or ()]
    points = [point for row in rows for point in row.points or ()]
    assert {row.platform for row in rows} == set(schema.PLATFORMS)
    assert {row.role for row in rows} == set(schema.ROLES)
    assert {row.completeness for row in rows} == set(schema.COMPLETENESS)
    assert {row.reachable for row in rows} == {True, False, None}
    assert {box.kind for box in boxes} == set(schema.KINDS)
    assert {point.kind for point in points} == set(schema.KINDS)
    assert {box.identity for box in boxes if box.is_face} == set(schema.IDENTITIES)
    assert any(box.ignore for box in boxes) and any(box.attributes for box in boxes)
    assert any(row.boxes == () for row in rows)


def test_the_example_manifest_uses_placeholder_video_ids_only():
    for row in manifest.read_manifest(EXAMPLE):
        assert row.video_id.startswith("EXAMPLE") or set(row.video_id[:-1]) == {"0"}


# -------------------------------------------------------------------------------- round trips
def test_a_row_survives_the_trip_through_json_unchanged():
    doc = row_doc(
        prefill_offered=True,
        attributes={"camera_angle": "steep", "table": 3, "mirrored": False},
        boxes=[
            box_doc(id="b1", origin="prefill", attributes={"visible_share": "half"}),
            box_doc(id="b2", card_id=None, identity="unknown", readable=None,
                    quad=[[0.5, 0.5], [0.6, 0.5], [0.6, 0.7], [0.5, 0.7]]),
            plain_box_doc("back", id="b3"),
            plain_box_doc("not-card", quad=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.3], [0.1, 0.3]]),
            plain_box_doc("unreadable", ignore=True, quad=[[-0.05, 0.8], [0.1, 0.8], [0.1, 1.2], [-0.05, 1.2]]),
        ],
        points=[point_doc(id="p1", origin="seeded"), {"x": 0.1, "y": 0.9, "kind": "not-card"}],
    )
    assert row_problems(doc) == []
    row = Row.from_json(doc)
    assert row.to_json() == doc
    assert Row.from_json(row.to_json()) == row
    assert json.loads(manifest.row_line(row)) == doc


def test_a_points_row_survives_the_trip_and_keeps_boxes_absent():
    doc = points_row_doc(points=[point_doc(), {"x": 0.25, "y": 0.75, "kind": "back"}])
    row = Row.from_json(doc)
    assert row.boxes is None and len(row.points) == 2
    assert "boxes" not in row.to_json() and row.to_json() == doc


def test_a_manifest_file_survives_the_trip_byte_for_byte(tmp_path):
    rows = manifest.read_manifest(EXAMPLE)
    again = tmp_path / "again.jsonl"
    assert manifest.write_manifest(again, rows) == len(rows)
    assert again.read_bytes() == EXAMPLE.read_bytes()
    assert manifest.read_manifest(again) == rows


def test_objects_built_in_code_write_a_valid_manifest(tmp_path):
    row = Row(platform="twitch", video_id=TWITCH_ID, t=3.5, frame_width=1280, frame_height=720,
              source_type="digital", group="g-test-02", role="evaluation", trained_on=False,
              completeness="points", credit="Someone",
              points=(Point(x=0.5, y=0.5, kind="face", card_id="SET-002", identity="named", readable="maybe"),
                      Point(x=0.9, y=0.9, kind="not-card")))
    path = tmp_path / "made.jsonl"
    manifest.write_manifest(path, [row])
    assert manifest.read_manifest(path) == [row]
    assert row.key == ("twitch", TWITCH_ID, 3.5) and row.source_family == "digital"


def test_writing_refuses_a_row_that_does_not_fit_the_format(tmp_path):
    bad = Row(platform="youtube", video_id=OTHER_VIDEO_ID, t=1.0, frame_width=1920, frame_height=1080,
              source_type="physical", group="g-test-03", role="train", trained_on=False,
              completeness="exhaustive-boxes", credit="Someone",
              boxes=(Box(kind="face", quad=((0.1, 0.1), (0.2, 0.1), (0.2, 0.3), (0.1, 0.3))),))
    path = tmp_path / "bad.jsonl"
    with pytest.raises(ManifestError, match='missing "identity"|must be one of'):
        manifest.write_manifest(path, [bad])
    assert not path.exists()


def test_reading_names_every_broken_line(tmp_path):
    path = write_lines(tmp_path / "m.jsonl", [row_doc(), "{not json", row_doc(t=2.0, role="judge"), "", row_doc(t=3.0)])
    with pytest.raises(ManifestError) as raised:
        manifest.read_manifest(path)
    message = str(raised.value)
    assert "line 2: not valid JSON" in message and "line 3: role" in message and "line 5" not in message


def test_reading_refuses_numbers_that_json_does_not_have(tmp_path):
    line = json.dumps(row_doc()).replace("12.517", "NaN")
    with pytest.raises(ManifestError, match="not valid JSON"):
        manifest.read_manifest(write_lines(tmp_path / "m.jsonl", [line]))


# ----------------------------------------------------------------------------------- filters
def test_rows_are_filtered_by_role_completeness_and_source():
    rows = manifest.read_manifest(EXAMPLE)
    assert [row.role for row in manifest.iter_rows(rows, role="evaluation")] == ["evaluation", "evaluation"]
    assert len(list(manifest.iter_rows(rows, role=("train", "extra")))) == 2
    assert {row.completeness for row in manifest.iter_rows(rows, completeness="points")} == {"points"}
    assert len(list(manifest.iter_rows(rows, completeness="exhaustive-boxes"))) == 3
    assert [row.source_type for row in manifest.iter_rows(rows, source_family="digital")] == ["digital"]
    assert len(list(manifest.iter_rows(rows, source_type="physical/phone"))) == 1
    assert [row.role for row in manifest.iter_rows(rows, trained_on=True)] == ["train"]
    assert len(list(manifest.iter_rows(rows, reachable=True))) == 2
    assert len(list(manifest.iter_rows(rows, reachable=False))) == 1
    assert len(list(manifest.iter_rows(rows, role="evaluation", completeness="points"))) == 1
    with pytest.raises(ValueError, match="unknown role"):
        list(manifest.iter_rows(rows, role="test"))


def test_rows_are_grouped_by_role_and_by_group():
    rows = manifest.read_manifest(EXAMPLE)
    by_role = manifest.rows_by_role(rows)
    assert {role: len(found) for role, found in by_role.items()} == {"train": 1, "evaluation": 2, "extra": 1}
    assert list(manifest.rows_by_group(rows)) == [row.group for row in rows]


# ------------------------------------------------------------- what a single row may not be
def _problems(**fields) -> str:
    return " | ".join(row_problems(row_doc(**fields)))


@pytest.mark.parametrize("field", sorted(schema.ROW_SCHEMA["required"]))
def test_a_missing_required_field_is_refused(field):
    assert f'missing "{field}"' in _problems(drop=(field,))


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"manifest_version": 2}, "manifest_version: must be 1"),
        ({"platform": "elsewhere"}, "platform: must be one of"),
        ({"video_id": "short"}, "video_id"),
        ({"video_id": "EXAMPLE 001"}, "video_id"),
        ({"platform": "twitch"}, "a Twitch video id is a number"),
        ({"t": -1}, "t: -1 is below the minimum"),
        ({"t": "12"}, "t: must be number"),
        ({"t": True}, "t: must be number"),
        ({"frame_width": 0}, "frame_width: 0 is below the minimum"),
        ({"frame_height": 1080.5}, "frame_height: must be integer"),
        ({"source_type": "studio"}, "source_type"),
        ({"source_type": "physical/Tournament"}, "source_type"),
        ({"group": "G 1"}, "group"),
        ({"role": "test"}, "role: must be one of"),
        ({"trained_on": "yes"}, "trained_on: must be boolean"),
        ({"trained_on": 1}, "trained_on: must be boolean"),
        ({"completeness": "partial"}, "completeness: must be one of"),
        ({"credit": ""}, "credit"),
        ({"credit": " padded "}, "credit"),
        ({"credit": "two\nlines"}, "credit"),
        ({"credit": "x" * 81}, "credit: must have at most 80"),
        ({"reachable": "yes"}, "reachable: must be boolean or null"),
        ({"reachable": True}, "reachable_checked: must be string"),
        ({"reachable_checked": "2026-01-01"}, "reachable_checked: must be null"),
        ({"reachable": True, "reachable_checked": "01/01/2026"}, "reachable_checked"),
        ({"note": "a free-text note"}, 'unknown field "note"'),
        ({"channel": "Someone"}, 'unknown field "channel"'),
        ({"prefill_offered": None}, "prefill_offered: must be boolean"),
        ({"attributes": {}}, "attributes: must have at least 1"),
        ({"attributes": {"Note": "x"}}, "attributes.Note (the key)"),
        ({"attributes": {"note": "free text with spaces"}}, "attributes.note"),
        ({"attributes": {"note": None}}, "attributes.note"),
        ({"attributes": {"deep": {"er": 1}}}, "attributes.deep"),
        ({"boxes": "none"}, "boxes: must be array"),
    ],
)
def test_a_row_field_outside_its_form_or_range_is_refused(fields, expected):
    assert expected in _problems(**fields)


def test_role_and_completeness_must_agree_with_the_rest_of_the_row():
    assert "only a train row can have trained" in _problems(role="evaluation", trained_on=True)
    assert "only a train row can have trained" in _problems(role="extra", trained_on=True)
    assert _problems(role="train", trained_on=True) == ""
    assert 'missing "boxes"' in _problems(drop=("boxes",))
    assert _problems(boxes=[]) == ""
    assert 'missing "points"' in " | ".join(row_problems(points_row_doc(drop=("boxes", "points"))))
    assert "points: must have at least 1" in " | ".join(row_problems(points_row_doc(points=[])))
    assert "boxes: must have at most 0" in " | ".join(row_problems(points_row_doc(boxes=[box_doc()], drop=())))
    assert row_problems(points_row_doc(boxes=[], drop=())) == []
    assert _problems(points=[point_doc()]) == ""


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        (box_doc(kind="card"), "boxes[0].kind: must be one of"),
        (box_doc(ignore=None), 'boxes[0]: missing "ignore"'),
        (box_doc(quad=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.3]]), "boxes[0].quad: must have at least 4"),
        (box_doc(quad=[[0.1, 0.1, 0.5], [0.2, 0.1], [0.2, 0.3], [0.1, 0.3]]), "boxes[0].quad[0]: must have at most 2"),
        (box_doc(quad=[[480, 270], [672, 270], [672, 540], [480, 540]]), "boxes[0].quad[0][0]: 480 is above the maximum"),
        (box_doc(quad=[[-1.5, 0.1], [0.2, 0.1], [0.2, 0.3], [-1.5, 0.3]]), "is below the minimum"),
        (box_doc(oriented="yes"), "boxes[0].oriented: must be boolean"),
        (box_doc(shape="circle"), "boxes[0].shape: must be one of"),
        (box_doc(card_id=""), "boxes[0].card_id: must have at least 1"),
        (box_doc(card_id="SET 001"), 'boxes[0].card_id: "SET 001" does not have the required form'),
        (box_doc(card_id="SET-001\t"), "boxes[0].card_id"),
        (box_doc(card_id="\u00a0SET-001"), "boxes[0].card_id"),
        (box_doc(card_id="SET\u0007001"), "boxes[0].card_id"),
        (box_doc(card_id="S" * 65), "boxes[0].card_id: must have at most 64"),
        (box_doc(identity=None), 'boxes[0]: missing "identity"'),
        (box_doc(identity="named", card_id=None), "boxes[0].card_id: must be string"),
        (box_doc(identity="unknown"), "boxes[0].card_id: must be null"),
        (box_doc(identity="unlabeled"), "boxes[0].card_id: must be null"),
        (box_doc(readable="sometimes"), "boxes[0].readable: must be one of"),
        (plain_box_doc("back", card_id="SET-001"), "boxes[0].card_id: is not allowed here"),
        (plain_box_doc("not-card", identity="unknown"), "boxes[0].identity: is not allowed here"),
        (plain_box_doc("unreadable", readable="no"), "boxes[0].readable: is not allowed here"),
        (box_doc(origin="By Hand"), "boxes[0].origin"),
        (box_doc(id="has space"), "boxes[0].id"),
        (box_doc(pixels=[1, 2, 3]), 'boxes[0]: unknown field "pixels"'),
    ],
)
def test_a_box_outside_its_form_is_refused(box, expected):
    assert expected in _problems(boxes=[box])


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (point_doc(x=1.01), "points[0].x: 1.01 is above the maximum"),
        (point_doc(y=-0.01), "points[0].y: -0.01 is below the minimum"),
        (point_doc(x=960), "points[0].x: 960 is above the maximum"),
        (point_doc(kind="nothing"), "points[0].kind: must be one of"),
        (point_doc(identity=None), 'points[0]: missing "identity"'),
        (point_doc(card_id=None), "points[0].card_id: must be string"),
        ({"x": 0.5, "y": 0.5, "kind": "not-card", "card_id": "SET-001"}, "points[0].card_id: is not allowed here"),
        ({"x": 0.5, "kind": "back"}, 'points[0]: missing "y"'),
    ],
)
def test_a_point_outside_its_form_is_refused(point, expected):
    assert expected in " | ".join(row_problems(points_row_doc(points=[point])))


def test_rules_that_a_schema_cannot_state():
    assert "more than three decimals" in _problems(t=12.5174)
    assert "is not a calendar date" in _problems(reachable=True, reachable_checked="2026-02-30")
    assert _problems(reachable=False, reachable_checked="2026-02-28") == ""
    crossed = box_doc(quad=[[0.25, 0.25], [0.35, 0.5], [0.35, 0.25], [0.25, 0.5]])
    assert "do not form a convex outline" in _problems(boxes=[crossed])
    dented = box_doc(quad=[[0.2, 0.2], [0.4, 0.2], [0.25, 0.25], [0.2, 0.4]])
    assert "do not form a convex outline" in _problems(boxes=[dented])
    backwards = box_doc(quad=[[0.25, 0.25], [0.25, 0.5], [0.35, 0.5], [0.35, 0.25]])
    assert "counter-clockwise" in _problems(boxes=[backwards])
    sliver = box_doc(quad=[[0.25, 0.25], [0.2502, 0.25], [0.2502, 0.5], [0.25, 0.5]])
    assert "shorter than one pixel" in _problems(boxes=[sliver])
    outside = box_doc(quad=[[1.1, 0.25], [1.2, 0.25], [1.2, 0.5], [1.1, 0.5]])
    assert "wholly outside the frame" in _problems(boxes=[outside])
    past_the_edge = box_doc(quad=[[0.95, 0.25], [1.05, 0.25], [1.05, 0.5], [0.95, 0.5]])
    assert _problems(boxes=[past_the_edge]) == ""
    assert "repeats an earlier box" in _problems(boxes=[box_doc(), box_doc()])
    assert '"b1" is used twice' in _problems(
        boxes=[box_doc(id="b1"), plain_box_doc(id="b1")])
    twice = [point_doc(), {"x": 0.5, "y": 0.5, "kind": "back"}]
    assert "repeats the position" in " | ".join(row_problems(points_row_doc(points=twice)))


# ------------------------------------------------------------------- a recording of your own
def _local(**fields) -> dict:
    return row_doc(**{"platform": "local", "video_id": "my-recording_01", **fields})


def test_a_row_may_name_a_recording_of_your_own():
    """``local`` is a platform a row may have beside the hosting platforms. Its video id is a
    name chosen by whoever made it, and the schema document explains both."""
    assert schema.LOCAL_PLATFORM == "local" and schema.LOCAL_PLATFORM not in schema.PLATFORMS
    assert schema.ROW_PLATFORMS == (*schema.PLATFORMS, schema.LOCAL_PLATFORM)
    assert schema.ROW_SCHEMA["properties"]["platform"]["enum"] == list(schema.ROW_PLATFORMS)
    assert schema.MANIFEST_VERSION == 1
    assert row_problems(_local()) == []
    row = schema.parse_row(_local())
    assert row.key == ("local", "my-recording_01", 12.517) and Row.from_json(row.to_json()) == row
    for video_id in ("a", "table", "2026-10-10_game-3", "EXAMPLE0001", "x" * 64):
        assert row_problems(_local(video_id=video_id)) == [], video_id
    text = (ROOT / "data" / "SCHEMA.md").read_text(encoding="utf-8")
    assert "`local`" in text and "--video-file" in text


@pytest.mark.parametrize("video_id", ["", "-starts-with-a-hyphen", "_starts_with_an_underscore", "two words", "folder/name",
                                      "folder\\name", "name.mp4", "name+77", "local:name", "x" * 65, "é"])
def test_the_video_id_of_a_recording_of_your_own_has_a_safe_form(video_id):
    assert "video_id" in " | ".join(row_problems(_local(video_id=video_id)))


def test_a_recording_of_your_own_has_no_reachability_check():
    """Nobody else can look for the video, so both fields stay null."""
    assert "reachable" in " | ".join(row_problems(_local(reachable=True, reachable_checked="2026-01-01")))
    assert "reachable" in " | ".join(row_problems(_local(reachable=False, reachable_checked="2026-01-01")))
    assert "reachable_checked" in " | ".join(row_problems(_local(reachable_checked="2026-01-01")))


def test_the_built_in_checker_agrees_with_a_json_schema_library_on_local_rows():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(schema.ROW_SCHEMA)
    candidates = [
        _local(), _local(video_id="-x"), _local(video_id="a b"), _local(video_id="x" * 64), _local(video_id="x" * 65),
        _local(reachable=True, reachable_checked="2026-01-01"), _local(reachable_checked="2026-01-01"),
        _local(video_id=TWITCH_ID), row_doc(platform="local"), row_doc(platform="elsewhere"),
        row_doc(platform="youtube", video_id="my-recording_01"), row_doc(platform="twitch", video_id="my-recording_01"),
    ]
    for candidate in candidates:
        ours = not schema.schema_errors(candidate, schema.ROW_SCHEMA)
        assert ours == validator.is_valid(candidate), json.dumps(candidate)
    assert [not schema.schema_errors(candidate, schema.ROW_SCHEMA) for candidate in candidates] == [
        True, False, False, True, False, False, False, True, True, False, False, False]


# ------------------------------------------------------------- nothing after the end of an id
def _ends_in_a_line_break() -> dict[str, dict]:
    """A row for every kind of id and tag the format has, valid but for a line break at its end."""
    return {
        "a video id on the first platform": row_doc(video_id=VIDEO_ID + "\n"),
        "a video id on the second platform": row_doc(platform="twitch", video_id=TWITCH_ID + "\n"),
        "the name of a recording of your own": _local(video_id="my-recording_01\n"),
        "a group": row_doc(group="g-test-01\n"),
        "a source type": row_doc(source_type="physical/webcam\n"),
        "a date": row_doc(reachable=True, reachable_checked="2026-01-01\n"),
        "a credit": row_doc(credit="Example Labeler\n"),
        "a box id": row_doc(boxes=[box_doc(id="b1\n")]),
        "a point id": points_row_doc(points=[point_doc(id="p1\n")]),
        "the card id of a box": row_doc(boxes=[box_doc(card_id="SET-001\n")]),
        "the card id of a point": points_row_doc(points=[point_doc(card_id="SET-001\n")]),
        "an origin": row_doc(boxes=[box_doc(origin="manual\n")]),
        "an attribute key": row_doc(attributes={"camera\n": "overhead"}),
        "an attribute value": row_doc(attributes={"camera": "overhead\n"}),
        "a frame fingerprint": row_doc(attributes={"frame_dhash": "00ff00ff00ff00ff\n"}),
    }


def _patterns(node) -> list[str]:
    """Every ``pattern`` of a schema, at any depth."""
    if isinstance(node, dict):
        own = [node["pattern"]] if isinstance(node.get("pattern"), str) else []
        return own + [found for value in node.values() for found in _patterns(value)]
    return [found for value in node for found in _patterns(value)] if isinstance(node, list) else []


@pytest.mark.parametrize("what", sorted(_ends_in_a_line_break()))
def test_an_id_that_ends_in_a_line_break_is_refused(what):
    """Python's ``$`` matches before a line break that ends the text. The format's does not."""
    doc = _ends_in_a_line_break()[what]
    assert row_problems(doc), what
    with pytest.raises(ManifestError):
        schema.parse_row(doc)
    # The same row without the line break is a valid one, so the line break is what was refused.
    assert row_problems(json.loads(json.dumps(doc).replace("\\n", ""))) == []


def test_a_pattern_ends_at_the_end_of_the_text():
    assert schema.fits_pattern(r"^[0-9]{2}$", "12") and not schema.fits_pattern(r"^[0-9]{2}$", "12\n")
    assert not schema.fits_pattern(r"^[0-9]{2}$", "123") and not schema.fits_pattern(r"^[0-9]{2}$", "\n12")
    # A pattern may still match anywhere in the text, as a JSON Schema pattern does.
    assert schema.fits_pattern("b", "abc") and schema.fits_pattern("c$", "abc") and not schema.fits_pattern("c$", "abc\n")
    assert schema.fits_pattern(r"\$", "a$b")                 # an escaped dollar is a dollar
    # Every pattern of the schema file is tied to both ends of the text, so the rule covers them all.
    patterns = _patterns(json.loads((ROOT / "data" / "manifest.schema.json").read_text(encoding="utf-8")))
    assert len(patterns) >= 12 and all(pattern.startswith("^") and pattern.endswith("$") for pattern in patterns)
    # The source types of a game's own file and the frame fingerprint are checked the same way.
    from cardvision.data.dhash import is_dhash
    from cardvision.data.sources import SourceConfigError, parse_source_config

    assert is_dhash("00ff00ff00ff00ff") and not is_dhash("00ff00ff00ff00ff\n")
    assert parse_source_config({"sourceTypes": ["digital/client-a"]}).source_types[-1] == "digital/client-a"
    with pytest.raises(SourceConfigError):
        parse_source_config({"sourceTypes": ["digital/client-a\n"]})


def test_the_schema_file_refuses_a_final_line_break_as_the_checker_does():
    """In a JSON Schema pattern ``$`` is the end of the text. The library used here searches
    with Python's ``re``, whose ``$`` also matches before a final line break, so it is given
    the format's reading of ``$``. With that reading the schema file and the built-in checker
    agree on every row that ends an id with a line break."""
    jsonschema = pytest.importorskip("jsonschema")

    def pattern(validator, wanted, instance, _schema):
        if validator.is_type(instance, "string") and not re.search(re.sub(r"\$$", r"\\Z", wanted), instance):
            yield jsonschema.ValidationError(f"{instance!r} does not match {wanted!r}")

    strict = jsonschema.validators.extend(jsonschema.Draft202012Validator, {"pattern": pattern})
    validator = strict(json.loads((ROOT / "data" / "manifest.schema.json").read_text(encoding="utf-8")))
    for what, doc in _ends_in_a_line_break().items():
        assert not validator.is_valid(doc) and schema.schema_errors(doc, schema.ROW_SCHEMA), what
        clean = json.loads(json.dumps(doc).replace("\\n", ""))
        assert validator.is_valid(clean) and not schema.schema_errors(clean, schema.ROW_SCHEMA), what


def test_a_card_id_has_no_whitespace_and_no_control_character():
    """The rule is a loose one, because a card id is whatever its catalog calls the card."""
    for card_id in ("SET-001", "OGN-001", "UNL-082A", "SFD-R01", "set.001/a_b+c", "\u30ab\u30fc\u30c9-001", "x" * 64):
        assert row_problems(row_doc(boxes=[box_doc(card_id=card_id)])) == [], card_id
        assert row_problems(points_row_doc(points=[point_doc(card_id=card_id)])) == [], card_id
    for card_id in ("SET 001", " SET-001", "SET-001 ", "SET-001\n", "SET-001\r\n", "SET\t001", "SET\u00a0001", "SET\u2028001",
                    "SET\u0000001", "SET\u001f001", "SET\u007f001", "SET\u0085001", "\ufeffSET-001", "", "x" * 65):
        assert row_problems(row_doc(boxes=[box_doc(card_id=card_id)])), repr(card_id)
        assert row_problems(points_row_doc(points=[point_doc(card_id=card_id)])), repr(card_id)


def test_every_card_id_of_the_published_files_fits_the_rule():
    """The manifests under ``data/``, the catalogs under ``games/`` and the two toy games."""
    import sys

    import toy_game

    ids: dict[str, set[str]] = {}
    for path in sorted((ROOT / "data").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                doc = json.loads(line)
                ids.setdefault(path.name, set()).update(
                    label["card_id"] for label in [*doc.get("boxes", []), *doc.get("points", [])] if isinstance(label.get("card_id"), str))
    for path in sorted((ROOT / "games").glob("*/catalog.json")):
        cards = json.loads(path.read_text(encoding="utf-8"))["cards"]
        ids[f"{path.parent.name}/{path.name}"] = {card[key] for card in cards for key in ("id", "printingOf") if card.get(key) is not None}
    ids["the toy game of the tests"] = {value for card in toy_game.CARDS for value in (card[0], card[3]) if value is not None}
    sys.path.insert(0, str(ROOT / "labeler"))
    try:
        import make_toy_data
    finally:
        sys.path.remove(str(ROOT / "labeler"))
    ids["the toy game of the labeling tool"] = {value for card in make_toy_data.CARDS for value in (card[0], card[3]) if value is not None}
    assert {"riftbound-evaluation.manifest.jsonl", "riftbound-train.manifest.jsonl", "riftbound-extra.manifest.jsonl",
            "example.manifest.jsonl", "riftbound/catalog.json", "_template/catalog.json"} <= set(ids)
    assert all(found for found in ids.values())
    for where, found in ids.items():
        unfit = sorted(card_id for card_id in found
                       if not (1 <= len(card_id) <= 64 and schema.fits_pattern(schema.CARD_ID_PATTERN, card_id)))
        assert unfit == [], where


def test_a_row_that_is_not_an_object_is_refused():
    assert row_problems([1, 2, 3]) and row_problems("row") and row_problems(None)
    with pytest.raises(ManifestError):
        schema.parse_row({"platform": "youtube"})
    assert schema.parse_row(row_doc()).video_id == "EXAMPLE0001"


# ------------------------------------------------ the checker agrees with a full implementation
def test_the_built_in_checker_agrees_with_a_json_schema_library():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema.ROW_SCHEMA)
    validator = jsonschema.Draft202012Validator(schema.ROW_SCHEMA)
    candidates = [json.loads(line) for line in EXAMPLE.read_text(encoding="utf-8").splitlines()]
    candidates += [
        row_doc(), points_row_doc(), row_doc(boxes=[]), row_doc(t=12), row_doc(frame_width=1920.0),
        row_doc(role="evaluation", trained_on=True), row_doc(drop=("boxes",)), row_doc(drop=("credit",)),
        row_doc(platform="twitch"), row_doc(platform="twitch", video_id=TWITCH_ID), row_doc(video_id="x"),
        row_doc(reachable=True), row_doc(reachable_checked="2026-01-01"), row_doc(reachable=False, reachable_checked="2026-01-01"),
        row_doc(note="x"), row_doc(attributes={}), row_doc(attributes={"a": "b"}), row_doc(attributes={"a": [1]}),
        row_doc(attributes={"A": 1}), row_doc(credit=" x"), row_doc(credit="x\ty"), row_doc(source_type="digital/client-a"),
        row_doc(source_type="other"), row_doc(t=-0.5), row_doc(t=True), row_doc(manifest_version=1.0),
        row_doc(manifest_version=True), row_doc(frame_height=2.5), row_doc(group="x"),
        row_doc(boxes=[box_doc(identity="unknown")]), row_doc(boxes=[box_doc(identity="unknown", card_id=None)]),
        row_doc(boxes=[plain_box_doc("back", card_id=None)]), row_doc(boxes=[plain_box_doc("back")]),
        row_doc(boxes=[box_doc(quad=[[0, 0], [3, 0], [3, 3], [0, 3]])]), row_doc(boxes=[box_doc(readable=None)]),
        row_doc(boxes=[box_doc(extra=1)]), row_doc(boxes=[box_doc(ignore=1)]),
        row_doc(boxes=[box_doc(card_id="SET 001")]), row_doc(boxes=[box_doc(card_id="SET-001\t")]),
        row_doc(boxes=[box_doc(card_id="set.001/a_b+c")]), row_doc(boxes=[box_doc(card_id="\u30ab\u30fc\u30c9-001")]),
        row_doc(boxes=[box_doc(card_id="SET\u00a0001")]), points_row_doc(points=[point_doc(card_id="SET 001")]),
        points_row_doc(points=[]), points_row_doc(boxes=[box_doc()], drop=()), points_row_doc(points=[point_doc(x=2)]),
        points_row_doc(points=[{"x": 0.1, "y": 0.2, "kind": "back", "readable": "no"}]),
        [], "row", 3, None, {},
    ]
    for candidate in candidates:
        ours = not schema.schema_errors(candidate, schema.ROW_SCHEMA)
        theirs = validator.is_valid(candidate)
        assert ours == theirs, json.dumps(candidate)
