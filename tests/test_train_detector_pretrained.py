# SPDX-License-Identifier: Apache-2.0
"""``cardvision train-detector --download-pretrained`` fetches and verifies the public
pretrained weights before PyTorch is looked for. The download here is a stand-in: nothing
touches the network, and none of these tests needs PyTorch."""

from __future__ import annotations

import functools
import hashlib
import io
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from cardvision import cli
from cardvision.train_detector import model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cardvision" / "train_detector" / "default.toml"
CONTENT = b"stand-in for the pretrained weights"


@pytest.fixture()
def download(monkeypatch, tmp_path):
    """A stand-in download of ``CONTENT``, a cache folder of its own, and the pinned hash set
    to that content's. Returns the list of addresses asked for."""
    asked: list[str] = []

    def fake(url, timeout=None, context=None):
        asked.append(url)
        assert isinstance(context, ssl.SSLContext) and context.verify_mode == ssl.CERT_REQUIRED
        return io.BytesIO(CONTENT)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv(model.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(model, "fetch_pretrained",
                        functools.partial(model.fetch_pretrained, sha256=hashlib.sha256(CONTENT).hexdigest()))
    return asked


def test_the_weights_are_downloaded_and_verified_before_pytorch_is_needed(download, monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)      # as if the training environment were not installed
    target = tmp_path / "cache" / "yolox_nano.pth"
    arguments = ["train-detector", "--config", str(CONFIG), "--out", str(tmp_path / "run"), "--download-pretrained"]
    assert cli.main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == f"pretrained weights: {target} (SHA-256 verified)\n"
    assert "training needs torch" in captured.err and 'python -m pip install -e ".[train]"' in captured.err
    assert download == [model.PRETRAINED_URL] and target.read_bytes() == CONTENT
    assert not (tmp_path / "run").exists() and sorted(path.name for path in target.parent.iterdir()) == ["yolox_nano.pth"]
    # A second run finds the file on disk, checks it again and asks for nothing.
    assert cli.main(arguments) == 2
    assert f"pretrained weights: {target} (SHA-256 verified)" in capsys.readouterr().out and len(download) == 1


def test_without_the_flag_nothing_is_downloaded_first(download, monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert cli.main(["train-detector", "--config", str(CONFIG), "--out", str(tmp_path / "run")]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "training needs torch" in captured.err
    # Nor with a starting checkpoint of one's own, whatever the flag says.
    assert cli.main(["train-detector", "--config", str(CONFIG), "--out", str(tmp_path / "run"), "--init", "random",
                     "--download-pretrained"]) == 2
    assert capsys.readouterr().out == "" and download == [] and not (tmp_path / "cache").exists()


def test_a_download_that_fails_its_check_is_reported_and_deleted(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None, context=None: io.BytesIO(CONTENT))
    monkeypatch.setenv(model.CACHE_ENV, str(tmp_path / "cache"))
    # The pinned hash is the real file's, which these bytes do not have.
    assert cli.main(["train-detector", "--config", str(CONFIG), "--out", str(tmp_path / "run"), "--download-pretrained"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "not the expected cd28f55f" in captured.err and "training needs" not in captured.err
    assert list((tmp_path / "cache").iterdir()) == []


def test_a_certificate_failure_says_what_to_install(monkeypatch, tmp_path, capsys):
    def unverified(url, timeout=None, context=None):
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))

    monkeypatch.setattr(urllib.request, "urlopen", unverified)
    monkeypatch.setenv(model.CACHE_ENV, str(tmp_path / "cache"))
    assert cli.main(["train-detector", "--config", str(CONFIG), "--out", str(tmp_path / "run"), "--download-pretrained"]) == 2
    err = capsys.readouterr().err
    assert "could not be verified" in err and "certifi" in err and 'python -m pip install -e ".[download]"' in err
    assert list((tmp_path / "cache").iterdir()) == []
