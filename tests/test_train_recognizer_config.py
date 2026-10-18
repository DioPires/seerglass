# SPDX-License-Identifier: Apache-2.0
"""The trainer's config: defaults, strict keys, the segment rules, and the Riftbound recipe
file. No PyTorch needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from cardvision.augment.components import DEFAULTS as COMPONENT_DEFAULTS
from cardvision.augment.physical import PHYSICAL
from cardvision.train_recognizer import config as C
from cardvision.train_recognizer.config import ConfigError, load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]
MINIMAL = {"data": {"catalog": "catalog.json", "art": "art"}, "epoch": {"base": 1000}}


def doc(**over) -> dict:
    return {**{key: dict(value) for key, value in MINIMAL.items()}, **over}


def test_a_minimal_config_gets_the_defaults(tmp_path):
    cfg = parse_config(doc(), tmp_path)
    assert (cfg["epochs"], cfg["batch"], cfg["contract"]) == (34, 256, "full128")
    assert cfg["data_seed"] == cfg["seed"] == 0
    assert cfg["optimizer"] == {"lr": 3e-4, "weight_decay": 1e-4, "warmup_epochs": 3}
    assert cfg["loss"]["arcface_scale"] == 32.0 and cfg["loss"]["contrastive_temperature"] == 0.07
    assert cfg["data"]["catalog"] == str((tmp_path / "catalog.json").resolve())
    assert cfg["data"]["margin"] == 0.4 and cfg["data"]["roles"] == ["train"]
    # The default epoch: card images, composites, real crops, real negatives.
    assert [(s["streams"], s["draw"]) for s in cfg["segment"]] == [
        (["art"], "balanced"), (["composite_positives"], "subset"), (["real_positives"], "balanced"),
        (["real_negatives"], "balanced")]
    assert cfg["segment"][0]["per_base"] == 1.0 and cfg["segment"][3]["share"] == 0.08 and cfg["segment"][3]["negative"]
    # In the public recipe the physical chain runs on card images and composites.
    assert {name: s["chain"] for name, s in cfg["streams"].items()} == {
        "art": "physical", "composite_positives": "physical", "real_positives": "plain", "real_negatives": "plain"}
    assert cfg["augment"]["physical"] == PHYSICAL
    assert cfg["augment"]["glare"] == COMPONENT_DEFAULTS["glare"]
    assert cfg["components"]["rotation"] is False and cfg["components"]["pairs"] is False


def test_the_composite_stream_states_which_faces_are_cut(tmp_path):
    limits = {"min_visible": 0.90, "min_short_side": 48, "max_tilt": 15.0}
    assert C.COMPOSITE_SELECTION == limits
    streams = parse_config(doc(), tmp_path)["streams"]
    assert {key: streams["composite_positives"][key] for key in limits} == limits
    assert not any(key in streams[name] for key in limits for name in ("art", "real_positives", "real_negatives"))
    # One of them stated: the other two keep their defaults, and so does the chain.
    stated = parse_config(doc(streams={"composite_positives": {"max_tilt": 20}}), tmp_path)["streams"]["composite_positives"]
    assert stated == {"chain": "physical", "negative": False, **limits, "max_tilt": 20}
    for key, bad in (("min_visible", 1.5), ("min_visible", -0.1), ("min_visible", "most"), ("min_short_side", -1),
                     ("max_tilt", 46), ("max_tilt", -1), ("max_tilt", True)):
        with pytest.raises(ConfigError, match=rf"\[streams.composite_positives\] {key} must be a number"):
            parse_config(doc(streams={"composite_positives": {key: bad}}), tmp_path)
    # They belong to the composite stream alone, and a misspelled one is refused.
    with pytest.raises(ConfigError, match=r"\[streams.real_positives\]: unknown key\(s\) \['max_tilt'\]"):
        parse_config(doc(streams={"real_positives": {"max_tilt": 20}}), tmp_path)
    with pytest.raises(ConfigError, match=r"\[streams.composite_positives\]: unknown key\(s\) \['min_visibility'\]"):
        parse_config(doc(streams={"composite_positives": {"min_visibility": 0.5}}), tmp_path)
    # The default file writes all three out.
    text = (ROOT / "cardvision" / "train_recognizer" / "default.toml").read_text(encoding="utf-8")
    assert all(f"\n{key} = " in text for key in limits)


def test_stream_tags_default_to_numbers_derived_from_their_names(tmp_path):
    import hashlib

    seeds = parse_config(doc(), tmp_path)["seeds"]
    assert set(seeds) == {*C.DRAW_TAGS, "sampler_offset"} == {
        "subset", "balanced", "negative", "per_class", "boost", "pairs", "sampler_offset"}
    for name in C.DRAW_TAGS:
        expected = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        assert seeds[name] == [expected] == [C.default_tag(name)]
    assert seeds["sampler_offset"] == C.default_tag("sampler_offset")
    # No two streams share a tag, so no two share a generator.
    assert len({tuple(seeds[name]) for name in C.DRAW_TAGS}) == len(C.DRAW_TAGS)
    assert C.DEFAULTS["seeds"] == seeds


def test_stream_tags_can_be_stated_to_repeat_another_run(tmp_path):
    stated = {"subset": 3, "balanced": [4], "negative": 5, "per_class": [6, 1], "boost": [6, 2], "pairs": [6, 3],
              "sampler_offset": 600}
    seeds = parse_config(doc(seeds=stated), tmp_path)["seeds"]
    assert seeds == {"subset": [3], "balanced": [4], "negative": [5], "per_class": [6, 1], "boost": [6, 2],
                     "pairs": [6, 3], "sampler_offset": 600}
    # Stating some leaves the others at their defaults.
    partly = parse_config(doc(seeds={"boost": [9, 9]}), tmp_path)["seeds"]
    assert partly["boost"] == [9, 9] and partly["subset"] == [C.default_tag("subset")]
    assert C.plain(parse_config(doc(seeds=stated), tmp_path))["seeds"]["per_class"] == [6, 1]   # recorded in run.json
    assert "[seeds]" in C.__doc__ and "salt" in C.__doc__


def test_init_is_one_of_two_words_or_the_path_of_a_checkpoint(tmp_path):
    assert C.INIT_WORDS == ("imagenet", "random") and C.CHECKPOINT_SUFFIXES == (".pt", ".safetensors")
    assert parse_config(doc(), tmp_path)["model"]["init"] == "imagenet"
    assert parse_config(doc(model={"init": "random"}), tmp_path)["model"]["init"] == "random"
    # A path is resolved against the config's folder, like every other path; nothing is opened here.
    for written in ("runs/other-game/best.pt", "exported/recognizer.safetensors"):
        cfg = parse_config(doc(model={"init": written}), tmp_path)
        assert cfg["model"]["init"] == str((tmp_path / written).resolve()) and C.is_checkpoint(cfg["model"]["init"])
    absolute = str((tmp_path / "elsewhere" / "last.pt").resolve())
    assert parse_config(doc(model={"init": absolute}), tmp_path / "configs")["model"]["init"] == absolute
    assert not C.is_checkpoint("imagenet") and not C.is_checkpoint("random") and not C.is_checkpoint(None)
    # Anything else is refused, so that a misspelled word is not taken for a file.
    for bad in ("pretrained", "Random", "", "best.pth", "weights.onnx", "runs/other-game", 3, None, ["random"]):
        with pytest.raises(ConfigError, match=r'\[model\] init must be "imagenet", "random" or the path of a checkpoint'):
            parse_config(doc(model={"init": bad}), tmp_path)
    # The default file says what the option is, and which start is the documented one.
    text = (ROOT / "cardvision" / "train_recognizer" / "default.toml").read_text(encoding="utf-8")
    assert 'init = "imagenet"' in text and '"imagenet" is the documented start' in text
    assert "has not been measured here" in text and "compare them on recordings held out from both" in text


def test_the_base_count_has_no_default():
    with pytest.raises(ConfigError, match=r"\[epoch\] base is required"):
        parse_config({"data": MINIMAL["data"]})
    with pytest.raises(ConfigError, match="catalog and art are required"):
        parse_config({"epoch": {"base": 10}})


@pytest.mark.parametrize("over, message", [
    ({"epochz": 3}, "unknown key 'epochz'"),
    ({"loss": {"arcface_scal": 1}}, "unknown key 'arcface_scal'"),
    ({"model": {"init": "pretrained"}}, "init must be"),
    ({"batch": 1}, "batch must be a whole number"),
    ({"data": {**MINIMAL["data"], "manifests": ["m.jsonl"]}}, "frames is required"),
    ({"validation": {"unseen_cards": ["A"], "unseen_pick": 2}}, "not both"),
    ({"streams": {"art": {"chain": "fancy"}}}, '"chain" must be one of'),
    ({"streams": {"art": {"color": "red"}}}, "unknown key"),
    ({"augment": {"glare": {"strength": 2}}}, r"\[augment.glare\]: unknown key"),
    ({"augment": {"sparkle": {}}}, "unknown table"),
    ({"components": {"glare": "yes"}}, "glare must be true or false"),
    ({"pairs": {"names": [["only one"]]}}, "pairs of card names"),
    ({"batch": 8, "components": {"pairs": True}, "pairs": {"per_batch": 4}}, "do not fit in a batch"),
    ({"seeds": {"subsets": 3}}, "unknown key 'subsets'"),
    ({"seeds": {"subset": -1}}, r"\[seeds\] subset must be a whole number"),
    ({"seeds": {"boost": [6, 2.5]}}, r"\[seeds\] boost must be a whole number"),
    ({"seeds": {"pairs": []}}, r"\[seeds\] pairs must be a whole number or a list"),
    ({"seeds": {"negative": "44"}}, r"\[seeds\] negative must be a whole number"),
    ({"seeds": {"sampler_offset": [1]}}, r"\[seeds\] sampler_offset must be a whole number"),
])
def test_a_config_that_is_wrong_says_where(over, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(doc(**over))


@pytest.mark.parametrize("segment, message", [
    ({"stream": "nowhere", "count": 5}, r"no \[streams.nowhere\] table"),
    ({"stream": "art", "streams": ["art"], "count": 5}, 'either "stream" or "streams"'),
    ({"stream": "art"}, "give exactly one of"),
    ({"stream": "art", "count": 5, "per_base": 1.0}, "give exactly one of"),
    ({"stream": "art", "count": 5, "draw": "shuffled"}, '"draw" must be one of'),
    ({"stream": "art", "per_class": 3, "draw": "balanced"}, "go together"),
    ({"stream": "art", "count": 3, "draw": "per_class"}, "go together"),
    ({"streams": ["art", "real_negatives"], "count": 4}, "cannot share a segment"),
    ({"streams": ["art", "real_positives"], "count": 4, "draw": "subset"}, 'only share a "balanced" segment'),
    ({"stream": "art", "share": 1.0}, "share must be a number"),
    ({"stream": "art", "rows": 5}, "unknown key"),
])
def test_segment_rules(segment, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(doc(segment=[segment]))


def test_segments_and_streams_as_a_plug_in_would_declare_them():
    cfg = parse_config(doc(
        streams={"my_crops": {"chain": "physical"}, "my_negatives": {"negative": True}},
        segment=[{"stream": "my_crops", "per_base": 1.0, "draw": "all"},
                 {"stream": "my_crops", "per_class": 4},
                 {"stream": "art", "draw": "all"},
                 {"streams": ["real_negatives", "my_negatives"], "share": 0.08}]))
    assert cfg["streams"]["my_crops"] == {"chain": "physical", "negative": False}
    assert cfg["streams"]["my_negatives"] == {"chain": "plain", "negative": True}
    assert cfg["streams"]["art"]["surface_margin"] == 0.4                         # the built-in streams stay
    assert [s["draw"] for s in cfg["segment"]] == ["all", "per_class", "all", "balanced"]
    assert cfg["segment"][3] == {"streams": ["real_negatives", "my_negatives"], "draw": "balanced", "negative": True, "share": 0.08}


def test_a_toml_file_is_read_and_its_ranges_become_tuples(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(
        'seed = 5\n[data]\ncatalog = "c.json"\nart = "/abs/art"\n[epoch]\nbase = 64\n'
        "[augment.crop]\njitter_scale = [0.9, 1.1]\n[augment.glare]\np = 0.5\n[augment.glare.types]\nbloom = 1.0\n",
        encoding="utf-8")
    cfg = load_config(path)
    assert cfg["seed"] == 5 and cfg["data"]["art"] == "/abs/art"
    assert cfg["augment"]["crop"]["jitter_scale"] == (0.9, 1.1) and cfg["augment"]["crop"]["jitter_translate"] == 0.15
    assert cfg["augment"]["glare"]["types"] == {"bloom": 1.0} and cfg["augment"]["glare"]["streak_amp"] == (0.25, 0.70)
    assert C.plain(cfg)["augment"]["crop"]["jitter_scale"] == [0.9, 1.1]
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
    path.write_text("seed = \n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(path)


def test_the_default_config_file_writes_out_the_defaults_of_the_code():
    written = load_config(ROOT / "cardvision" / "train_recognizer" / "default.toml")
    defaults = parse_config({"data": {"catalog": "catalog.json", "art": "art"}, "epoch": {"base": 20000}},
                            ROOT / "cardvision" / "train_recognizer")
    assert written == defaults


def test_the_riftbound_recipe_file():
    cfg = load_config(ROOT / "games" / "riftbound" / "train-recognizer.toml")
    assert (cfg["epochs"], cfg["batch"], cfg["workers"], cfg["contract"]) == (34, 256, 13, "full128")
    assert cfg["model"] == {"init": "imagenet", "dim": 128}
    assert cfg["optimizer"] == {"lr": 3e-4, "weight_decay": 1e-4, "warmup_epochs": 3}
    assert cfg["loss"] == {
        "arcface_scale": 32.0, "arcface_margin": 0.2, "arcface_weight": 1.0, "contrastive_temperature": 0.07,
        "contrastive_weight": 0.5, "negative_weight": 1.0, "negative_margin": 0.55, "back_weight": 4.0,
        "back_margin": 0.50, "back_labels": ["back"], "bank_refresh_steps": 50}
    assert cfg["components"] == {"occluders": True, "glare": True, "low_light": True, "rotation": False,
                                 "printing_mask": True, "pairs": True, "boost": True}
    assert cfg["validation"]["every"] == 5 and cfg["validation"]["unseen_count"] == 400
    assert cfg["data"]["margin"] == 0.4
    assert cfg["streams"]["composite_positives"] == {"chain": "physical", "negative": False, **C.COMPOSITE_SELECTION}
    assert (cfg["pairs"]["per_batch"], len(cfg["pairs"]["names"])) == (4, 20)
    assert cfg["boost"] == {"alternate_printings": 1.0, "share_kind": "Battlefield", "share": 0.081}
    assert [(s["streams"], s.get("per_base"), s.get("share")) for s in cfg["segment"]] == [
        (["art"], 1.0, None), (["composite_positives"], 0.30, None), (["real_positives"], None, 0.10),
        (["real_negatives"], None, 0.08)]
    # The file writes the augmentation settings out; they are the package's defaults.
    assert cfg["augment"]["physical"] == PHYSICAL
    assert {name: cfg["augment"][name] for name in COMPONENT_DEFAULTS} == COMPONENT_DEFAULTS
    # Its catalog is the one beside it, which carries the kind settings (no separate kinds
    # file), and every pair names catalog cards.
    from cardvision.composites.cards import KindSettings
    from cardvision.core.catalog import load_catalog

    catalog = load_catalog(cfg["data"]["catalog"])
    assert cfg["data"]["kinds"] is None and catalog.kinds
    kinds = KindSettings.from_kinds(catalog.kinds)
    assert kinds.dice and kinds.foil
    assert all(catalog.ids_named(name) for pair in cfg["pairs"]["names"] for name in pair)
    assert any(card.kind == cfg["boost"]["share_kind"] for card in catalog)
    assert set(kinds.dice) | set(kinds.foil) <= {card.kind for card in catalog}


def test_the_riftbound_recipe_names_the_cards_kept_out_and_states_the_epoch():
    from cardvision.core.catalog import load_catalog

    path = ROOT / "games" / "riftbound" / "train-recognizer.toml"
    cfg = load_config(path)
    kept_out = cfg["validation"]["unseen_cards"]
    # 37 cards, each named once, each in the catalog beside the file; no seeded pick in their place.
    assert len(kept_out) == 37 and len(set(kept_out)) == 37 and cfg["validation"]["unseen_pick"] == 0
    catalog = load_catalog(ROOT / "games" / "riftbound" / "catalog.json")
    assert [card_id for card_id in kept_out if card_id not in catalog] == []
    assert cfg["data"]["catalog"] == str(ROOT / "games" / "riftbound" / "catalog.json")
    by_set = {prefix: sum(card_id.startswith(prefix) for card_id in kept_out) for prefix in ("OGN-", "SFD-", "UNL-", "VEN-")}
    assert by_set == {"OGN-": 14, "SFD-": 11, "UNL-": 6, "VEN-": 6} and kept_out == sorted(kept_out)
    assert kept_out[0] == "OGN-017" and kept_out[-1] == "VEN-164"
    # The epoch: the stated base, and 800 steps of 256 rows over 34 epochs.
    assert (cfg["epoch"]["base"], cfg["batch"], cfg["epochs"]) == (113704, 256, 34)
    text = " ".join(" ".join(line.lstrip("# ").split()) for line in path.read_text(encoding="utf-8").splitlines())
    assert "113,704 crops from a private stream, covering 870 training cards" in text
    assert "800 steps at batch 256" in text and "for 34 epochs: 27,200 steps in all" in text
    assert "about" not in text[text.index("The unit of the sizes below"):text.index("base = 113704")]
    # The arithmetic of the released epoch, from the base up. The trainer's rule for a share F of the
    # rows so far is round(F / (1 - F) x rows so far); its module needs PyTorch, so it is written here.
    def share_rows(share, rows_so_far):
        return int(round(share / (1 - share) * rows_so_far))

    base = cfg["epoch"]["base"]
    pool, of_kept_out = 27943, 817
    assert round(0.30 * base) == 34111 > pool - of_kept_out == 27126
    composite = min(round(0.30 * base), pool - of_kept_out)
    base_so_far = base + composite
    real = share_rows(0.10, base_so_far)
    positives = base_so_far + real
    negatives = share_rows(0.08, positives)
    real_negatives, stream_negatives = negatives // 2, negatives - negatives // 2
    subtotal = positives + negatives
    boost, second_stream = 14700, 13728
    pairs = 2 * cfg["pairs"]["per_batch"] * 800
    assert 13728 % 96 == 0 and pairs == 6400 and boost + second_stream + 2 * 3200 == 34828
    total = subtotal + boost + second_stream + pairs
    assert (composite, base_so_far, real, positives, negatives) == (27126, 140830, 15648, 156478, 13607)
    assert (real_negatives, stream_negatives, subtotal, total) == (6803, 6804, 170085, 204913)
    # 800 steps of 256 rows, the last partial batch dropped; 34 epochs.
    assert total // cfg["batch"] == 800 and total / cfg["batch"] > 800 and 800 * cfg["epochs"] == 27200
    # The shares, to two decimal places by plain rounding, nothing adjusted; they add to exactly 100.00.
    rows = (base, composite, real, real_negatives, stream_negatives, second_stream, boost, pairs)
    assert sum(rows) == total
    shares = [round(100 * count / total, 2) for count in rows]
    assert shares == [55.49, 13.24, 7.64, 3.32, 3.32, 6.70, 7.17, 3.12]
    assert round(sum(shares), 2) == 100.00
    assert round(100 * (base + second_stream) / total, 2) == 62.19
    # The comment states every count and share, and the model card the same ones.
    steps = ("34,111", "27,126", "27,943", "817", "140,830", "15,648", "156,478", "13,607", "170,085", "34,828", "204,913")
    assert all(f"{count:,}" in text for count in rows) and all(number in text for number in steps)
    assert all(f"{share:.2f} %" in text for share in shares)
    assert "nothing is adjusted" in text and "exactly 100.00 %" in text
    card = " ".join((ROOT / "docs" / "model-card.md").read_text(encoding="utf-8").split())
    assert "113,704 crops from a private stream, covering 870 training cards" in card
    assert "204,913 rows per epoch" in card and "Share of 204,913" in card
    assert all(number in card for number in steps) and all(f"{count:,}" in card for count in rows)
    for count, share in zip(rows, shares):
        assert f"| {count:,} | {share:.2f} % |" in card
    assert "| Total | 204,913 | 100.00 % |" in card
    assert "Nothing is adjusted" in card and "exactly 100.00 %" in card and "62.19 %" in card
    assert "205,000" not in card and "205,000" not in text and "about 114,000" not in card
