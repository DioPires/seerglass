# SPDX-License-Identifier: Apache-2.0
"""Device choice, tested against a stand-in for PyTorch so the tests need no accelerator."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from cardvision.core import device as D

ROOT = Path(__file__).resolve().parents[1]


class FakeTorch(types.ModuleType):
    """The small part of the torch module that device.py touches."""

    def __init__(self, cuda: bool = False, mps: bool = False) -> None:
        super().__init__("torch")
        self.__version__ = "0.0"
        self.calls: list[str] = []
        self.cuda = types.SimpleNamespace(
            is_available=lambda: cuda, synchronize=lambda: self.calls.append("cuda"))
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
        self.mps = types.SimpleNamespace(synchronize=lambda: self.calls.append("mps"))


@pytest.fixture
def torch(monkeypatch):
    def install(cuda: bool = False, mps: bool = False) -> FakeTorch:
        fake = FakeTorch(cuda, mps)
        monkeypatch.setitem(sys.modules, "torch", fake)
        return fake

    monkeypatch.delenv(D.DEVICE_ENV, raising=False)
    return install


@pytest.mark.parametrize("cuda,mps,allow_mps,expected", [
    (True, True, True, "cuda"),
    (True, False, True, "cuda"),
    (True, True, False, "cuda"),
    (False, True, True, "mps"),
    (False, True, False, "cpu"),
    (False, False, True, "cpu"),
    (False, False, False, "cpu"),
])
def test_automatic_order_is_cuda_then_mps_then_cpu(torch, cuda, mps, allow_mps, expected):
    torch(cuda, mps)
    assert D.pick_device(allow_mps=allow_mps) == expected


def test_default_allows_mps(torch):
    torch(mps=True)
    assert D.pick_device() == "mps"


@pytest.mark.parametrize("value,expected", [("cpu", "cpu"), ("cuda", "cuda"), ("mps", "mps"), (" CUDA ", "cuda"), ("Mps", "mps")])
def test_the_environment_variable_decides(torch, monkeypatch, value, expected):
    torch(cuda=True, mps=True)
    monkeypatch.setenv("CARDVISION_DEVICE", value)
    assert D.DEVICE_ENV == "CARDVISION_DEVICE"
    assert D.pick_device() == expected


def test_the_environment_variable_can_force_mps_past_allow_mps(torch, monkeypatch):
    torch(cuda=True, mps=True)
    monkeypatch.setenv(D.DEVICE_ENV, "mps")
    assert D.pick_device(allow_mps=False) == "mps"


def test_an_empty_environment_variable_means_automatic(torch, monkeypatch):
    torch(mps=True)
    monkeypatch.setenv(D.DEVICE_ENV, "  ")
    assert D.pick_device() == "mps"


def test_forcing_an_unknown_device_is_an_error(torch, monkeypatch):
    torch(cuda=True)
    monkeypatch.setenv(D.DEVICE_ENV, "gpu")
    with pytest.raises(ValueError, match="CARDVISION_DEVICE='gpu'"):
        D.pick_device()


@pytest.mark.parametrize("value", ["cuda", "mps"])
def test_forcing_a_device_the_machine_lacks_is_an_error(torch, monkeypatch, value):
    torch(cuda=False, mps=False)
    monkeypatch.setenv(D.DEVICE_ENV, value)
    with pytest.raises(RuntimeError, match=f"{value} but"):
        D.pick_device()


def test_synchronize_waits_on_the_right_accelerator(torch):
    fake = torch(cuda=True, mps=True)
    D.synchronize("cpu")
    assert fake.calls == []
    D.synchronize("cuda")
    D.synchronize("cuda:1")
    D.synchronize("mps")
    assert fake.calls == ["cuda", "cuda", "mps"]


@pytest.mark.skipif(sys.platform == "win32", reason="reads resource.getrusage, which is POSIX only")
def test_rss_max_mb_is_a_plausible_number_of_megabytes():
    peak = D.rss_max_mb()
    assert isinstance(peak, float) and 1 < peak < 1_000_000
    assert peak == round(peak, 1)


def test_importing_the_core_does_not_import_torch_or_onnxruntime():
    code = (
        "import sys, cardvision.core.device, cardvision.core.onnx, cardvision.core.bank, "
        "cardvision.core.catalog, cardvision.core.preprocess; "
        "assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr
