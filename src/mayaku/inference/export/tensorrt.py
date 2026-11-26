"""TensorRT: a serialized engine for NVIDIA GPUs, built from the ONNX graph.

fp16 (the default), int8 and fp32. An engine is specific to the GPU model and
the TensorRT version it was built with, so build it on the machine that
serves it. The engine format has no metadata slot, so the sidecar is
length-prefixed in front of the engine bytes (`metadata.strip_tensorrt_header`
removes it).

int8 is the explicit int8 graph (the trained ranges as Q/DQ pairs, with add
operands quantized and one scale per tensor, see `mayaku.model.quant`), so
TensorRT runs exactly what was trained; it needs symmetric activation ranges
(TensorRT has no zero point), the default since `model.qat_activations`.

TensorRT 10 builds reduced precision from builder flags. TensorRT 11 is
strongly typed: the flags are gone and the graph itself has to carry fp16, so
the ONNX is converted first with NVIDIA ModelOpt AutoCast (the
`nvidia-modelopt[onnx]` package), on a flat gray image, which converts every
node, as TensorRT 10's fp16 flag does.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mayaku.inference.decode import pack_boxes
from mayaku.inference.export.metadata import embed_sidecar, strip_tensorrt_header
from mayaku.inference.export.onnx import onnx_graph
from mayaku.inference.preprocess import Canvases, planar_tensor
from mayaku.model import Detector
from mayaku.utils.requirements import require

PRECISIONS = ("fp16", "int8", "fp32")
REGRID_INT8 = False   # no zero point: int8 keeps the trained symmetric grid (`mayaku.model.quant`)
QUANT_SKIPS = True    # quantized skips let TensorRT fuse each add into its conv


_WORKSPACE = 1 << 30
# Level 2 cuts cold build time 2.5-4x on our 5-op graph (RTX 3060, v4-n 768:
# fp16 71.8 -> 27.4s, int8 50.7 -> 13.6s) within <=0.07 ms E2E of level 3.
_DEFAULT_OPT_LEVEL = 2


def _opt_level() -> int:
    raw = os.environ.get("MAYAKU_TRT_OPT_LEVEL")
    if raw is None or not raw.strip():
        return _DEFAULT_OPT_LEVEL
    level = int(raw.strip())
    if not 0 <= level <= 5:
        raise ValueError(f"MAYAKU_TRT_OPT_LEVEL must be 0..5, got {level}")
    return level


def _timing_cache_enabled() -> bool:
    return os.environ.get("MAYAKU_TRT_TIMING_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _timing_cache_path(gpu_name: str, trt_version: str, opt_level: int) -> Path:
    """Where TensorRT's timing cache lives on this host: keyed by GPU,
    TensorRT major.minor and builder optimization level so repeat exports and
    fine-tunes reuse layer timings (3-5x faster builds) without polluting cwd."""
    override = os.environ.get("MAYAKU_CACHE_DIR")
    root = Path(override).expanduser() if override else Path.home() / ".cache" / "mayaku"
    gpu_slug = re.sub(r"[^a-z0-9]+", "_", gpu_name.lower()).strip("_") or "gpu"
    trt_mm = ".".join(trt_version.split(".")[:2])
    return root / f"trt_timing_{gpu_slug}_trt{trt_mm}_opt{opt_level}.bin"


def _load_timing_cache(config: Any, cache_path: Path) -> bool:
    """Attach the persistent timing cache (or an empty one on miss / corrupt
    file) to `config`; returns whether an existing cache was loaded."""
    raw = b""
    if cache_path.is_file():
        with contextlib.suppress(OSError):
            raw = cache_path.read_bytes()
    if raw:
        try:
            cache = config.create_timing_cache(raw)
            config.set_timing_cache(cache, ignore_mismatch=False)
            return True
        except Exception:
            pass
    cache = config.create_timing_cache(b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    return False


def _save_timing_cache(config: Any, cache_path: Path) -> None:
    """Persist `config`'s timing cache atomically; silently ignored if the
    cache directory is read-only."""
    try:
        cache = config.get_timing_cache()
        if cache is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        tmp.write_bytes(bytes(cache.serialize()))
        tmp.replace(cache_path)
    except OSError:
        pass


def runnable() -> bool:
    return torch.cuda.is_available()


def write(model: Detector, path: Path, canvas: tuple[int, int], precision: str,
          sidecar: Mapping[str, Any], batch: int = 1) -> None:
    """The engine at a fixed batch (1 for `export`, which checks it against
    the model; a larger fixed batch for throughput)."""
    import tensorrt as trt

    from mayaku.model.quant import activation_scheme

    if not torch.cuda.is_available():
        raise RuntimeError("building a TensorRT engine needs a CUDA GPU")
    int8 = precision == "int8"
    if int8 and activation_scheme(model) != "symmetric":
        raise ValueError("TensorRT runs int8 with symmetric activation ranges only; this model "
                         "was trained affine (train with model.qat_activations=symmetric)")
    strongly_typed = int(trt.__version__.split(".")[0]) >= 11
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with onnx_graph(model, canvas, int8=int8, batch=batch, regrid=REGRID_INT8) as onnx:
        if strongly_typed and precision != "fp32":
            onnx = _fp16_graph(onnx, canvas, batch)
        if not parser.parse(onnx.read_bytes()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse the ONNX graph:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE)
    opt_level = _opt_level()
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = opt_level
    gpu_name = torch.cuda.get_device_name(0)
    use_cache = _timing_cache_enabled() and hasattr(config, "create_timing_cache")
    cache_path = _timing_cache_path(gpu_name, trt.__version__, opt_level) if use_cache else None
    cache_hit = _load_timing_cache(config, cache_path) if cache_path is not None else False
    if not strongly_typed:
        if precision == "fp32":
            config.clear_flag(trt.BuilderFlag.TF32)   # fp32 means fp32
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)
    cache_note = ("cache hit" if cache_hit else "cache miss") if use_cache else "cache off"
    print(f"[mayaku] building TensorRT {precision} engine for {gpu_name} "
          f"({canvas[0]}x{canvas[1]}, opt={opt_level}, {cache_note})...",
          file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the engine; see its log above")
    elapsed = time.perf_counter() - t0
    if cache_path is not None:
        _save_timing_cache(config, cache_path)
    path.write_bytes(bytes(engine))
    embed_sidecar(path, "tensorrt", dict(sidecar))
    print(f"[mayaku] built TensorRT {precision} engine in {elapsed:.1f}s -> {path}",
          file=sys.stderr, flush=True)


def _fp16_graph(src: Path, canvas: tuple[int, int], batch: int = 1) -> Path:
    """The ONNX graph in fp16 with fp32 inputs and outputs, for a strongly
    typed build (TensorRT 11): ModelOpt AutoCast on a flat gray image, with
    `data_max` raised from AutoCast's 512 to fp16's max, so only a value fp16
    cannot hold on that image keeps a node in fp32 -- every node converts, as
    TensorRT 10's fp16 flag converts it (neither guards overflow on real
    images). Written next to `src`."""
    import onnx

    require("modelopt")              # TensorRT 11's fp16 graph needs NVIDIA ModelOpt
    from modelopt.onnx import autocast

    gray = np.full((batch, 3, *canvas), 114 / 255, np.float32)
    out = src.with_suffix(".fp16.onnx")
    onnx.save(autocast.convert_to_mixed_precision(str(src), low_precision_type="fp16", keep_io_types=True,
                                                  calibration_data={"images": gray},
                                                  data_max=float(np.finfo(np.float16).max)), str(out))
    return out


class Session:
    """The engine on the current CUDA device, with torch tensors as its I/O
    buffers. The BGR canvases cross as bytes through a reused pinned buffer;
    the planar RGB transpose, the 1/255 and the engine run as one CUDA graph.
    The maps stay on the GPU, so the decode runs there too and only the final
    detections come back."""

    def __init__(self, path: Path, outputs: list[str], precision: str, device: str = "auto"):
        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self._engine = runtime.deserialize_cuda_engine(strip_tensorrt_header(path))
        self._ctx = self._engine.create_execution_context()
        # an engine runs on a CUDA device whatever `device` says
        self._device = torch.device(device if device.startswith("cuda") else "cuda")
        name = self._engine.get_tensor_name(0)
        self._input = torch.empty(tuple(self._engine.get_tensor_shape(name)),
                                  dtype=torch.float32, device=self._device)
        self._ctx.set_tensor_address(name, self._input.data_ptr())
        b, _, h, w = self._input.shape
        self.batch, self.input_hw = int(b), (int(h), int(w))
        self._host = torch.empty((b, h, w, 3), dtype=torch.uint8).pin_memory()
        self._pixels = torch.empty((b, h, w, 3), dtype=torch.uint8, device=self._device)
        self._uploaded = torch.cuda.Event()      # recorded after each upload
        self._pending = False
        self._out = {n: torch.empty(tuple(self._engine.get_tensor_shape(n)),
                                    dtype=torch.float32, device=self._device)
                     for n in outputs}
        for n, t in self._out.items():
            self._ctx.set_tensor_address(n, t.data_ptr())
        self.outputs = outputs
        self._graph: torch.cuda.CUDAGraph | None = None
        self._box_graphs: dict[tuple[int, int, tuple[int, ...], int, bool],
                               tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self._pad_place: tuple[int, int] | None = None

    def canvases(self, n: int) -> Canvases | None:
        """The pinned upload buffer, once the last upload has left it; the
        caller letterboxes into it and `__call__` sends it without a copy."""
        if n > self.batch:
            return None
        self._pad_place = None
        self._wait()
        return self._host.numpy()[:n]

    def box_canvases(self) -> Canvases:
        """The batch-1 pinned upload buffer preserving `_pad_place` for `_predict_one`."""
        self._wait()
        return self._host.numpy()[:1]

    def need_pad(self, placement: tuple[int, int]) -> bool:
        """Whether the slot-0 canvas border needs filling before resizing to `placement`."""
        if self._pad_place == placement:
            return False
        self._pad_place = placement
        return True

    def _stage_input(self, x: Canvases) -> None:
        if x.ctypes.data != self._host.data_ptr():
            self._pad_place = None
            self._wait()
            self._host.numpy()[:len(x)] = x

    def forward_boxes(self, x: Canvases, nc: int, reg_max: int, strides: tuple[int, ...],
                      topk: int, multi_label: bool) -> torch.Tensor:
        """Batch-1 fused CUDA graph: upload, planar RGB, TensorRT inference,
        top-K + DFL box decode and async copy of ``(K, 6)`` candidates to
        pinned CPU memory."""
        key = (nc, reg_max, strides, topk, multi_label)
        pair = self._box_graphs.get(key)
        if pair is None:
            pair = self._capture_boxes(*key)
            self._box_graphs[key] = pair
        graph, host_out = pair
        self._stage_input(x)
        with torch.cuda.device(self._device):
            self._pixels.copy_(self._host, non_blocking=True)
            graph.replay()
            torch.cuda.current_stream(self._device).synchronize()
        self._pending = False
        return host_out

    def _run_boxes(self, nc: int, reg_max: int, strides: tuple[int, ...], k: int,
                   multi_label: bool, host_out: torch.Tensor) -> None:
        self._run()
        n_maps = 2 * len(strides)
        cls_maps = [self._out[n] for n in self.outputs[:n_maps:2]]
        box_maps = [self._out[n] for n in self.outputs[1:n_maps:2]]
        pack_boxes(cls_maps, box_maps, nc, reg_max, strides, k, multi_label, host_out)

    def _capture_boxes(self, nc: int, reg_max: int, strides: tuple[int, ...], topk: int,
                       multi_label: bool) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        n_anchors = sum(self._out[n].shape[2] * self._out[n].shape[3]
                        for n in self.outputs[:2 * len(strides):2])
        k = min(topk, n_anchors * nc if multi_label else n_anchors)
        host_out = torch.empty((k, 6), dtype=torch.float32).pin_memory()
        with torch.cuda.device(self._device):
            self._run_boxes(nc, reg_max, strides, k, multi_label, host_out)
            torch.cuda.current_stream(self._device).synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._run_boxes(nc, reg_max, strides, k, multi_label, host_out)
        return graph, host_out

    def __call__(self, x: Canvases) -> list[torch.Tensor]:
        # no sync after the upload: work on the maps is ordered after the
        # engine on this stream
        self._stage_input(x)
        self._pixels.copy_(self._host, non_blocking=True)
        self._uploaded.record(torch.cuda.current_stream(self._device))
        self._pending = True
        if self._graph is None:
            self._graph = self._capture()
        else:
            self._graph.replay()
        return [self._out[n] for n in self.outputs]

    def _wait(self) -> None:
        """Until the last upload has left the pinned buffer."""
        if self._pending:
            self._uploaded.synchronize()
            self._pending = False

    def _run(self) -> None:
        planar_tensor(self._pixels, out=self._input)
        self._ctx.execute_async_v3(torch.cuda.current_stream(self._device).cuda_stream)

    def _capture(self) -> torch.cuda.CUDAGraph:
        """Run once (TensorRT's lazy setup cannot be captured), then record the
        input conversion and the engine as one CUDA graph that later calls
        replay: one launch instead of one per layer, which at small sizes is
        what bounds the latency."""
        self._run()
        torch.cuda.current_stream(self._device).synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(self._device), torch.cuda.graph(graph):
            self._run()
        return graph
