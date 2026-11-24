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
from mayaku.model.contract import assert_contract, export_onnx
from mayaku.utils.requirements import install, pip_spec

PRECISIONS = ("fp32", "int8")
REGRID_INT8 = True    # one-sided int8 ranges get the whole grid (`mayaku.model.quant`)
QUANT_SKIPS = True    # residual skips quantized, as trained


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
    if precision == "int8":
        store_int8(path)
        assert_contract(str(path), int8=True)
    embed_sidecar(path, "onnx", dict(sidecar))


def store_int8(path: Path) -> None:
    """Rewrite an int8 graph in the form ONNX Runtime runs as int8.

    The traced graph keeps each weight in float behind its QuantizeLinear,
    and its activations' zero points are int8. ONNX Runtime fuses
    DequantizeLinear -> Conv -> QuantizeLinear into one int8 convolution only
    for int8 weight initializers, int32 biases and uint8 activations;
    otherwise every conv runs in float between extra (de)quantize steps, and
    the file is as large as fp32. Same network: each weight is the int8 value
    its QuantizeLinear gave, each activation the same grid shifted by 128,
    each bias rounded to the conv's accumulator step (input scale x weight
    scale), as an int8 runtime adds it."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(path))
    g = m.graph
    init = {i.name: i for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}

    def value(name: str) -> Any:    # an initializer, else a Constant node's value
        t = init[name] if name in init else producer[name].attribute[0].t
        return numpy_helper.to_array(t)

    def add(array: Any, name: str) -> str:
        name = f"{name}_{len(g.initializer)}"
        g.initializer.append(numpy_helper.from_array(array, name))
        return name

    dropped, bias_dq, weight_dq = set(), [], set()
    for conv in [n for n in g.node if n.op_type == "Conv"]:
        x_dq, w_dq = producer[conv.input[0]], producer[conv.input[1]]
        w_q = producer[w_dq.input[0]]
        if not (x_dq.op_type == w_dq.op_type == "DequantizeLinear"
                and w_q.op_type == "QuantizeLinear"):
            continue    # a conv left in float (one that feeds a residual add)
        w, w_scale = value(w_q.input[0]), value(w_q.input[1])
        assert not np.any(value(w_q.input[2])), "weights are quantized symmetrically"
        per_channel = w_scale.reshape((-1,) + (1,) * (w.ndim - 1))
        w_dq.input[0] = add(np.clip(np.round(w / per_channel), -128, 127).astype(np.int8),
                            conv.input[1] + "_int8")
        dropped.add(w_q.output[0])
        weight_dq.add(w_dq.output[0])
        b_scale = value(x_dq.input[1]) * w_scale
        b = np.round(value(conv.input[2]) / b_scale).astype(np.int32)
        out = conv.input[2] + "_dq"     # zero point left out: 0
        bias_dq.append(helper.make_node(
            "DequantizeLinear",
            [add(b, conv.input[2] + "_int32"), add(b_scale, conv.input[2] + "_scale")],
            [out], axis=0))
        conv.input[2] = out

    nodes = [n for n in g.node if n.output[0] not in dropped]
    uint8: dict[str, str] = {}      # int8 zero point -> the same grid as uint8
    for n in nodes:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear") and len(n.input) > 2 \
                and n.output[0] not in weight_dq and value(n.input[2]).dtype == np.int8:
            z = n.input[2]
            if z not in uint8:
                uint8[z] = add((value(z).astype(np.int16) + 128).astype(np.uint8), z + "_u8")
            n.input[2] = uint8[z]

    nodes = bias_dq + nodes
    used = {i for n in nodes for i in n.input}   # Constant nodes have no inputs
    nodes = [n for n in nodes if n.op_type != "Constant" or n.output[0] in used]
    kept = [i for i in g.initializer if i.name in used]
    del g.node[:]
    g.node.extend(nodes)
    del g.initializer[:]
    g.initializer.extend(kept)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


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
