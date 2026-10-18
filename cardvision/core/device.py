# SPDX-License-Identifier: Apache-2.0
"""Choose the PyTorch device for training and export, the same way on every machine.

PyTorch is not a dependency of the core package; it is imported here on first use, so only the
tools that train or export need it installed.
"""

from __future__ import annotations

import os
import sys

DEVICE_ENV = "CARDVISION_DEVICE"
DEVICES = ("cuda", "mps", "cpu")


def pick_device(allow_mps: bool = True) -> str:
    """Return ``"cuda"``, ``"mps"`` or ``"cpu"``.

    If the ``CARDVISION_DEVICE`` environment variable is set, it decides, and it is an error
    for it to name a device this machine does not have. Otherwise the order is CUDA, then MPS,
    then the CPU. A caller whose recipe is not validated on MPS passes ``allow_mps=False`` to
    skip it in the automatic order; the environment variable can still force it.
    """
    import torch

    forced = os.environ.get(DEVICE_ENV, "").strip().lower()
    if forced:
        if forced not in DEVICES:
            raise ValueError(f"{DEVICE_ENV}={forced!r}: expected one of {', '.join(DEVICES)}")
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"{DEVICE_ENV}=cuda but CUDA is not available (torch {torch.__version__})")
        if forced == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(f"{DEVICE_ENV}=mps but MPS is not available")
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: object) -> None:
    """Wait for work queued on an accelerator to finish, so that a timing taken next is honest.

    Does nothing for the CPU.
    """
    import torch

    name = str(device)
    if name.startswith("cuda"):
        torch.cuda.synchronize()
    elif name == "mps":
        torch.mps.synchronize()


def rss_max_mb() -> float:
    """Peak resident memory of this process in megabytes, to one decimal place.

    Available on POSIX systems only (it reads ``resource.getrusage``).
    """
    import resource

    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 1e6 if sys.platform == "darwin" else peak / 1e3, 1)
