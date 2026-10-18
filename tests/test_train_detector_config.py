# SPDX-License-Identifier: Apache-2.0
"""The detector trainer's config: the default file, the released recipe's file, and the checks."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from cardvision.train_detector.config import Config, ConfigError, load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "cardvision" / "train_detector" / "default.toml"
RELEASED = ROOT / "games" / "riftbound" / "train-detector.toml"
MARK = "# extra stream"


def completed(text: str) -> str:
    """The released recipe's file with its commented placeholder lines switched on."""
    lines = []
    for line in text.splitlines():
        if line.rstrip().endswith(MARK):
            assert line.lstrip().startswith("# "), line
            indent = line[:len(line) - len(line.lstrip())]
            line = indent + line.lstrip()[2:]
        lines.append(line)
    return "\n".join(lines) + "\n"


def test_the_default_file_states_the_defaults():
    config = load_config(DEFAULT)
    defaults = Config()
    for field in dataclasses.fields(Config):
        if field.name not in ("streams", "raw"):
            assert getattr(config, field.name) == getattr(defaults, field.name), field.name
    assert sorted(config.streams) == ["composites", "negatives", "real"]
    assert config.extra_sources == ()
    # Stated in counts and shares: 7 + 3 + 6 fill the batch, and the shares add up to one.
    assert config.primary == (("real", 7), ("negatives", 3), ("composites", 6))
    assert sum(tiles for _, tiles in config.primary) == config.batch_size == 16
    assert sum(share for _, share in config.mosaic_extra) == pytest.approx(1.0, abs=1e-6)
    assert config.no_augment_from == 450


def test_the_default_mix_is_the_released_one_shared_out_again():
    released = parse_config(_toml(completed(RELEASED.read_text(encoding="utf-8"))))
    public = [(name, tiles) for name, tiles in released.primary if name != "extra"]
    scaled = {name: tiles * 16 / sum(tiles for _, tiles in public) for name, tiles in public}
    assert {name: round(value) for name, value in scaled.items()} == dict(Config().primary)
    shares = [(name, share) for name, share in released.mosaic_extra if name != "extra"]
    total = sum(share for _, share in shares)
    assert [(name, round(share / total, 4)) for name, share in shares] == list(Config().mosaic_extra)


def _toml(text: str) -> dict:
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(text)


def test_the_released_recipe_does_not_run_without_its_extra_stream():
    with pytest.raises(ConfigError, match="adds up to 11 tiles, and batch_size is 16"):
        load_config(RELEASED)


def test_the_released_recipe_once_completed():
    text = RELEASED.read_text(encoding="utf-8")
    assert text.count(MARK) == 5
    config = parse_config(_toml(completed(text)))
    assert (config.seed, config.steps, config.batch_size, config.store_width) == (1337, 500, 16, 704)
    assert config.primary == (("real", 5), ("negatives", 2), ("composites", 4), ("extra", 5))
    assert config.primary_order() == ["real"] * 5 + ["negatives"] * 2 + ["composites"] * 4 + ["extra"] * 5
    assert config.mosaic_extra == (("real", 0.30), ("composites", 0.25), ("negatives", 0.125), ("extra", 0.325))
    assert config.streams["extra"] == {"degrade": 0.6}
    assert len(config.extra_sources) == 1
    a, o = config.augment, config.optimizer
    assert (a.mosaic, a.degrees, a.translate, a.scale, a.shear, a.flip) == (0.5, 10.0, 0.1, (0.5, 1.5), 2.0, 0.5)
    assert (a.no_augment_share, a.max_targets, a.max_ignore) == (0.10, 200, 480)
    assert config.no_augment_from == 450
    assert (o.lr, o.momentum, o.weight_decay, o.warmup_steps, o.final_lr_ratio, o.ema_decay) == \
        (0.001, 0.9, 0.0005, 30, 0.05, 0.9998)
    assert config.ignore_overlap == 0.5
    real = config.streams["real"]
    assert real["roles"] == ["train"] and real["trained_on"] is True and real["aspect_range"] == [0.55, 0.90]
    # Apart from the mix, the machine settings and the stream tables, it is the default recipe.
    defaults = Config()
    for name in ("seed", "steps", "batch_size", "store_width", "augment", "optimizer", "ignore_overlap"):
        assert getattr(config, name) == getattr(defaults, name), name


def test_init_names_the_starting_weights_and_is_left_out_by_default():
    base = {"streams": {"negatives": {"folder": "n"}}, "batch_size": 2,
            "mix": {"primary": [{"stream": "negatives", "tiles": 2}], "mosaic_extra": [{"stream": "negatives", "share": 1.0}]}}
    # Left out: the public pretrained checkpoint, and nothing about it in the recorded recipe.
    plain = parse_config(base)
    assert plain.init is None and Config().init is None and "init" not in plain.to_json()
    assert load_config(DEFAULT).init is None and load_config(DEFAULT).to_json().keys() == Config().to_json().keys()
    # Stated: a checkpoint's path, as written, or "random"; the recorded recipe carries it.
    for written in ("runs/other-game/last.pth", "exported/detector.safetensors", "random"):
        config = parse_config({**base, "init": written})
        assert config.init == written and config.to_json()["init"] == written
        assert list(config.to_json())[:5] == ["seed", "steps", "batch_size", "store_width", "init"]
    assert parse_config({**base, "init": "a.pth"}).with_changes(steps=3).init == "a.pth"
    for bad in ("", "  ", 3, True, ["last.pth"]):
        with pytest.raises(ConfigError, match='init must be the path of a checkpoint to start from, or "random"'):
            parse_config({**base, "init": bad})
    # The default file says what the option is, and which start is the documented one.
    text = DEFAULT.read_text(encoding="utf-8")
    assert "that is the documented start" in text and '#     init = "runs/other-game/last.pth"' in text
    assert "has not been measured here" in text and "compare them on recordings\n# held out from both" in text


def test_both_files_carry_the_license_identifier():
    for path in (DEFAULT, RELEASED):
        assert path.read_text(encoding="utf-8").splitlines()[0] == "# SPDX-License-Identifier: Apache-2.0"


def test_json_and_toml_read_alike(tmp_path):
    doc = {"seed": 5, "steps": 12, "batch_size": 2,
           "streams": {"negatives": {"folder": "n"}},
           "mix": {"primary": [{"stream": "negatives", "tiles": 2}],
                   "mosaic_extra": [{"stream": "negatives", "share": 1.0}]},
           "augment": {"scale": [0.8, 1.2]}, "loader": {"workers": 0}}
    (tmp_path / "c.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "c.toml").write_text(
        'seed = 5\nsteps = 12\nbatch_size = 2\n[streams.negatives]\nfolder = "n"\n[mix]\n'
        'primary = [ { stream = "negatives", tiles = 2 } ]\nmosaic_extra = [ { stream = "negatives", share = 1.0 } ]\n'
        "[augment]\nscale = [0.8, 1.2]\n[loader]\nworkers = 0\n", encoding="utf-8")
    from_json, from_toml = load_config(tmp_path / "c.json"), load_config(tmp_path / "c.toml")
    assert from_json == from_toml and from_json.to_json() == from_toml.to_json()
    assert from_json.augment.scale == (0.8, 1.2) and from_json.loader.workers == 0
    assert from_json.stream_names() == ["negatives"] and from_json.no_augment_from == 11
    assert from_json.raw == doc
    resolved = from_json.to_json()
    assert resolved["optimizer"]["lr"] == 0.001 and resolved["mix"]["primary"] == [{"stream": "negatives", "tiles": 2}]
    assert json.loads(json.dumps(resolved)) == resolved


def test_with_changes_checks_again():
    config = Config()
    assert config.with_changes(steps=40).no_augment_from == 36
    with pytest.raises(ConfigError, match="steps"):
        config.with_changes(steps=0)
    with pytest.raises(ConfigError, match="batch_size is 8"):
        config.with_changes(batch_size=8)


@pytest.mark.parametrize(("doc", "message"), [
    ({"stepz": 3}, "unknown key stepz"),
    ({"augment": {"mosiac": 0.5}}, r"\[augment\]: unknown key mosiac"),
    ({"optimizer": {"lr": "fast"}}, "lr must be a number"),
    ({"optimizer": {"warmup_steps": 0}}, "warmup_steps"),
    ({"augment": {"mosaic": 1.5}}, "between 0.0 and 1.0"),
    ({"augment": {"scale": [1.5]}}, "scale"),
    ({"steps": 2.5}, "steps must be a whole number"),
    ({"mix": {"primary": []}}, "primary must be a list"),
    ({"mix": {"primary": [{"stream": "real", "tiles": 8}, {"stream": "real", "tiles": 8}]}}, "listed twice"),
    ({"mix": {"primary": [{"stream": "real", "tiles": 8, "weight": 1}]}}, "each entry is"),
    ({"mix": {"primary": [{"stream": "real", "tiles": 15}]}}, "adds up to 15 tiles"),
    ({"mix": {"mosaic_extra": [{"stream": "real", "share": 0.5}]}}, "adds up to 0.5"),
    ({"mix": {"shuffle": True}}, r"\[mix\]: unknown key shuffle"),
    ({"streams": {"real": {"degrade": 2}}}, "degrade"),
    ({"streams": {"real": "frames/"}}, r"\[streams.real\] must be a table"),
    ({"extra_sources": ["no_colon"]}, "extra_sources"),
    ({"loader": {"salt": 3}}, "salt must be a string"),
])
def test_what_the_check_refuses(doc, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(doc)


def test_files_that_cannot_be_read(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.toml")
    (tmp_path / "c.yaml").write_text("steps: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="ends in .toml or .json"):
        load_config(tmp_path / "c.yaml")
    (tmp_path / "bad.toml").write_text("steps = = 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(tmp_path / "bad.toml")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(tmp_path / "bad.json")
