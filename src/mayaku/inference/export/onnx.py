"""ONNX: the canonical artifact, and the source the OpenVINO and TensorRT
builds start from. fp32 is the structural five-op graph; int8 (a
quantization-aware model) is the same graph with the trained ranges as
QuantizeLinear / DequantizeLinear around every conv."""

from __future__ import annotations

import contextlib
import io
import tempfile
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from mayaku.inference.export.metadata import embed_sidecar
from mayaku.inference.preprocess import Canvases, planar
from mayaku.model import Detector
from mayaku.model.contract import export_onnx
from mayaku.utils.requirements import install, pip_spec

PRECISIONS = ("fp32", "int8")
REGRID_INT8 = True    # one-sided int8 ranges get the whole grid (`mayaku.model.quant`)


def runnable() -> bool:
    return True


@contextlib.contextmanager
def onnx_graph(model: Detector, canvas: tuple[int, int], int8: bool, batch: int = 1,
               regrid: bool = False) -> Iterator[Path]:
    """The model as a contract-checked ONNX file in a temporary directory,
    for the runtimes that build from ONNX."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        export_onnx(model, str(path), canvas, batch=batch, int8=int8, regrid=regrid)
        yield path


def write(model: Detector, path: Path, canvas: tuple[int, int], precision: str,
          sidecar: Mapping[str, Any]) -> None:
    export_onnx(model, str(path), canvas, int8=precision == "int8", regrid=REGRID_INT8)
    embed_sidecar(path, "onnx", dict(sidecar))


def _session(ort: Any, path: Path, cuda: bool) -> Any:
    """An onnxruntime session on CUDA when `cuda`, else the CPU. CUDA first
    loads the CUDA and cuDNN libraries from their pip packages, which
    onnxruntime does not find by itself."""
    providers = ["CPUExecutionProvider"]
    if cuda:
        with contextlib.redirect_stdout(io.StringIO()):   # it prints each one it misses
            ort.preload_dlls()
        providers.insert(0, "CUDAExecutionProvider")
    options = ort.SessionOptions()
    # errors only: CUDA warns about every int8 graph, whose constant bias
    # dequantization stays on the host (a few small copies, by design)
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=providers)


class Session:
    """onnxruntime, on CUDA when asked for and available, else the CPU."""

    def __init__(self, path: Path, outputs: list[str], precision: str, device: str = "auto"):
        import onnxruntime as ort

        cuda = device.startswith(("cuda", "auto")) and \
            "CUDAExecutionProvider" in ort.get_available_providers()
        self._sess = _session(ort, path, cuda)
        if cuda and "CUDAExecutionProvider" not in self._sess.get_providers():
            # an onnxruntime-gpu installed without its CUDA libraries (before
            # mayaku asked for them) falls back to the CPU: install them once
            spec = pip_spec("onnxruntime")
            error = install(spec, "onnxruntime runs on the CPU: its CUDA libraries are missing")
            if error is None:
                self._sess = _session(ort, path, cuda)
            if "CUDAExecutionProvider" not in self._sess.get_providers():
                warnings.warn(error or "onnxruntime runs on the CPU: its CUDA provider did not "
                              f"load; reinstall it with: pip install '{spec}'", stacklevel=2)
        inp = self._sess.get_inputs()[0]
        self._input, self.outputs = inp.name, outputs
        self.input_hw = tuple(inp.shape[-2:])
        # traced at a fixed batch size (1 unless exported otherwise)
        self.batch = inp.shape[0] if isinstance(inp.shape[0], int) else None

    def canvases(self, n: int) -> Canvases | None:
        return None        # the host conversion reads any array

    def __call__(self, x: Canvases) -> list[torch.Tensor]:
        maps = self._sess.run(self.outputs, {self._input: planar(x)})
        return [torch.from_numpy(m) for m in maps]
