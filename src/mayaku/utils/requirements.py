"""Install a deployment runtime the first time it is used.

mayaku has no install extras: ``pip install mayaku`` brings what training and
PyTorch inference need (ONNX included, since every export goes through it),
and each runtime is installed by the call that first needs it -- exporting to
its format or loading one of its artifacts (`pip_spec` says which package).
Every install pins the installed torch and torchvision, so a runtime can never
replace the user's torch build, and each install is printed. Set
``MAYAKU_AUTOINSTALL=0`` to turn it off: a missing runtime then raises with the
exact ``pip install`` command instead.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from types import ModuleType

__all__ = ["autoinstall_enabled", "pip_spec", "require"]

#: import name -> the oldest version mayaku runs with
_MIN = {"onnxruntime": "1.17", "openvino": "2024.3", "coremltools": "8.0",
        "tensorrt": "10.0", "modelopt": "0.44"}


def autoinstall_enabled() -> bool:
    return os.environ.get("MAYAKU_AUTOINSTALL", "1").strip().lower() not in ("0", "false", "no", "off")


def pip_spec(name: str) -> str:
    """The pip requirement that provides the runtime imported as `name` on
    this host."""
    import torch

    floor = _MIN[name]
    cuda = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else None
    if name == "onnxruntime":
        if not (cuda and torch.cuda.is_available() and platform.system() != "Darwin"):
            return f"onnxruntime>={floor}"
        # its PyPI GPU wheels are built for CUDA 13 from 1.27, for CUDA 12 before
        return "onnxruntime-gpu>=1.21,<1.27" if cuda <= 12 else "onnxruntime-gpu>=1.27"
    if name == "tensorrt":
        if cuda is None:
            raise RuntimeError("TensorRT needs a CUDA build of torch; this one is "
                               f"{torch.__version__} without CUDA")
        # 10.x: TensorRT 11 builds fp16/int8 only through ModelOpt, which
        # needs torch >= 2.8 and pins its own onnx; an installed 11 still runs
        return f"tensorrt-cu{cuda}>={floor},<11"
    if name == "modelopt":
        return f"nvidia-modelopt[onnx]>={floor}"
    return f"{name}>={floor}"


def _numbers(version: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", version)[:3])


def _satisfied(name: str) -> bool:
    try:
        module = sys.modules.get(name) or importlib.import_module(name)
    except ImportError:
        return False
    return _numbers(str(getattr(module, "__version__", "0"))) >= _numbers(_MIN[name])


def _pinned() -> list[str]:
    """``pkg==version`` for the installed torch and torchvision."""
    pins = []
    for dist in ("torch", "torchvision"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            pins.append(f"{dist}=={importlib.metadata.version(dist)}")
    return pins


def require(name: str) -> ModuleType:
    """Import the runtime `name` (a key of `_MIN`), installing it first when
    it is missing or older than mayaku supports."""
    if _satisfied(name):
        return importlib.import_module(name)
    spec = pip_spec(name)
    command = f"pip install '{spec}'"
    if not autoinstall_enabled():
        raise ImportError(f"{name} is not installed (or too old); run: {command} "
                          "(MAYAKU_AUTOINSTALL=0 turned off installing it automatically)")
    pins = _pinned()
    print(f"[mayaku] installing {spec} (keeping {', '.join(pins)}); "
          "set MAYAKU_AUTOINSTALL=0 to turn this off", file=sys.stderr, flush=True)
    # the pins ride along as requirements: satisfied as installed, so pip
    # refuses any resolution that would replace them
    result = subprocess.run([sys.executable, "-m", "pip", "install", spec, *pins])
    if result.returncode:
        raise ImportError(f"installing {spec} failed (pip exit {result.returncode}); "
                          f"install it yourself: {command}")
    importlib.invalidate_caches()
    # an older version imported before the install must not shadow the new one
    for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        del sys.modules[mod]
    return importlib.import_module(name)
