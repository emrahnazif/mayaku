"""Runtimes installed on first use: the right wheel per host, torch never
replaced, and a clear command when installing is turned off."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest
import torch

from mayaku.utils import requirements as req


@pytest.mark.parametrize(("cuda", "build", "system", "want"), [
    (True, "12.4", "Linux", "onnxruntime-gpu>=1.21,<1.27"), (True, "13.0", "Linux", "onnxruntime-gpu>=1.27"),
    (False, "12.4", "Linux", "onnxruntime>=1.17"), (True, "12.4", "Darwin", "onnxruntime>=1.17")])
def test_onnxruntime_wheel_follows_the_host(monkeypatch, cuda, build, system, want) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.version, "cuda", build)
    monkeypatch.setattr(req.platform, "system", lambda: system)
    assert req.pip_spec("onnxruntime") == want


def test_tensorrt_wheel_matches_the_cuda_build_of_torch(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.4")
    assert req.pip_spec("tensorrt") == "tensorrt-cu12>=10.0,<11"
    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match="CUDA build of torch"):
        req.pip_spec("tensorrt")


def test_turned_off_it_raises_with_the_command(monkeypatch) -> None:
    monkeypatch.setenv("MAYAKU_AUTOINSTALL", "0")
    monkeypatch.setattr(req, "_satisfied", lambda name: False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not install"))
    with pytest.raises(ImportError, match=r"pip install 'openvino>=2024\.3'"):
        req.require("openvino")


def test_an_install_pins_torch_and_imports_the_result(monkeypatch) -> None:
    monkeypatch.setenv("MAYAKU_AUTOINSTALL", "1")
    monkeypatch.setattr(req, "_satisfied", lambda name: False)
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **_: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(req.importlib, "import_module", lambda n: types.SimpleNamespace(__name__=n))
    assert req.require("openvino").__name__ == "openvino"
    (cmd,) = seen
    assert cmd[:5] == [sys.executable, "-m", "pip", "install", "openvino>=2024.3"]
    assert any(a.startswith("torch==") for a in cmd) and any(a.startswith("torchvision==") for a in cmd)


def test_versions_compare_numerically(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(__version__="1.9.0"))
    assert not req._satisfied("onnxruntime")             # 1.9 < 1.17, not "1.9" > "1.17"
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(__version__="1.23.2"))
    assert req._satisfied("onnxruntime")
