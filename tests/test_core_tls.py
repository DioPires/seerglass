# SPDX-License-Identifier: Apache-2.0
"""Certificates for the commands that download: certifi's bundle is used when it is installed,
and a certificate that cannot be verified becomes a message that says what to install. No
connection is opened here."""

from __future__ import annotations

import json
import ssl
import sys
import types
import urllib.error
import urllib.request

import pytest

from cardvision.bank import build as bank_build
from cardvision.composites import fetch as textures
from cardvision.core import tls
from cardvision.infer.contact_sheet import ArtLibrary


def test_the_certifi_bundle_is_used_when_the_package_is_installed(monkeypatch):
    asked = []
    monkeypatch.setattr(ssl, "create_default_context", lambda **options: asked.append(options) or "context")
    monkeypatch.setitem(sys.modules, "certifi", types.SimpleNamespace(where=lambda: "/somewhere/cacert.pem"))
    assert tls.tls_context() == "context" and asked == [{"cafile": "/somewhere/cacert.pem"}]
    monkeypatch.setitem(sys.modules, "certifi", None)       # as if it were not installed
    assert tls.tls_context() == "context" and asked[1] == {}


def test_certificates_are_always_verified():
    context = tls.tls_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True


def test_a_certificate_failure_is_recognized_in_an_error_and_in_its_text():
    cause = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate")
    assert tls.is_certificate_failure(cause) and tls.is_certificate_failure(urllib.error.URLError(cause))
    assert tls.is_certificate_failure("ERROR: Unable to download API page: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    wrapped = RuntimeError("the download failed")
    wrapped.__cause__ = cause
    assert tls.is_certificate_failure(wrapped)
    for other in (urllib.error.URLError("timed out"), OSError("connection refused"), "HTTP Error 404: Not Found",
                  ssl.SSLError("handshake failure")):
        assert not tls.is_certificate_failure(other)


def test_the_help_says_what_to_install():
    assert 'python -m pip install -e ".[download]" in the repository folder' in tls.CERTIFICATE_HELP
    assert "certifi" in tls.CERTIFICATE_HELP and "no CA certificates" in tls.CERTIFICATE_HELP
    assert issubclass(tls.CertificatesMissing, RuntimeError)


def test_fetch_textures_stops_at_a_certificate_failure_and_says_what_to_install(tmp_path, monkeypatch):
    """Every request would fail the same way, so the first failure ends the run with the help,
    in place of one failed scan after another."""
    asked = []

    def unverified(request, timeout=None, context=None):
        asked.append(request.full_url)
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))

    monkeypatch.setattr(urllib.request, "urlopen", unverified)
    assets = [{"source": "toy", "assetId": name, "category": "wood", "license": "CC0 1.0", "sha256": "0" * 64,
               "url": f"https://scans.example/{name}.jpg"} for name in ("oak", "pine", "ash")]
    index = tmp_path / "assets.json"
    index.write_text(json.dumps(assets), encoding="utf-8")
    with pytest.raises(textures.FetchError) as caught:
        textures.fetch_textures(index, tmp_path / "scans", delay=0)
    assert asked == ["https://scans.example/oak.jpg"]
    assert "toy oak: the server's TLS certificate could not be verified" in str(caught.value) and "certifi" in str(caught.value)
    # An ordinary failure is still one failed scan, and the others are tried.
    asked.clear()

    def unreachable(request, timeout=None, context=None):
        asked.append(request.full_url)
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", unreachable)
    summary = textures.fetch_textures(index, tmp_path / "scans", delay=0)
    assert len(asked) == 3 and len(summary["failed"]) == 3


def unverified_urlopen(asked):
    def unverified(request, timeout=None, context=None):
        asked.append((request if isinstance(request, str) else request.full_url, context))
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))

    return unverified


def test_the_bank_build_reads_art_over_https_with_the_shared_context_and_says_what_to_install(monkeypatch):
    asked = []
    monkeypatch.setattr(urllib.request, "urlopen", unverified_urlopen(asked))
    with pytest.raises(tls.CertificatesMissing) as caught:
        bank_build.read_art("https://cards.example/art/a.png")
    assert str(caught.value) == f"https://cards.example/art/a.png: {tls.CERTIFICATE_HELP}"
    assert len(asked) == 1 and isinstance(asked[0][1], ssl.SSLContext) and asked[0][1].verify_mode == ssl.CERT_REQUIRED
    # An address that is not https gets no context, and a missing image is still only a missing image.
    asked.clear()
    with pytest.raises(tls.CertificatesMissing):
        bank_build.read_art("HTTPS://cards.example/art/a.png")
    assert isinstance(asked[0][1], ssl.SSLContext)

    def not_found(request, timeout=None, context=None):
        asked.append((request.full_url, context))
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    asked.clear()
    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    with pytest.raises(urllib.error.HTTPError) as missing:
        bank_build.read_art("http://cards.example/art/a.png")
    assert bank_build.art_is_missing(missing.value) and asked == [("http://cards.example/art/a.png", None)]


def test_the_bank_build_stops_at_a_certificate_failure_with_the_help(monkeypatch):
    asked = []
    monkeypatch.setattr(urllib.request, "urlopen", unverified_urlopen(asked))
    with pytest.raises(bank_build.BankBuildError) as caught:
        bank_build._read_card_art("TOY-001", "https://cards.example/art/TOY-001.png")
    assert tls.CERTIFICATE_HELP in str(caught.value) and "https://cards.example/art/TOY-001.png" in str(caught.value)


def test_the_contact_sheet_downloads_with_the_shared_context_and_gives_up_at_a_certificate_failure(monkeypatch):
    from cardvision.core.catalog import parse_catalog

    cards = [{"id": f"A-{i}", "name": f"Card {i}", "art": f"{i}.png"} for i in range(5)]
    catalog = parse_catalog({"game": "toy", "locale": "en", "cards": cards})
    asked = []
    monkeypatch.setattr(urllib.request, "urlopen", unverified_urlopen(asked))
    art = ArtLibrary(catalog, "https://cards.example/art/", give_up_after=10)
    assert [art.image(card["id"]) for card in cards] == [None] * 5
    assert len(asked) == 1 and isinstance(asked[0][1], ssl.SSLContext)         # every download would fail alike
    assert art.certificate_problem == f"https://cards.example/art/0.png: {tls.CERTIFICATE_HELP}"
    assert len(art.missing) == 5
    # An ordinary failure leaves no such note.
    ordinary = ArtLibrary(catalog, "https://cards.example/art/", fetch=lambda url, timeout: (_ for _ in ()).throw(OSError("refused")))
    assert ordinary.image("A-0") is None and ordinary.certificate_problem is None


def test_the_download_extra_brings_certifi():
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    from pathlib import Path

    extras = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    names = {requirement.split(">")[0].split("=")[0].strip() for requirement in extras["project"]["optional-dependencies"]["download"]}
    assert names == {"yt-dlp", "certifi"}
