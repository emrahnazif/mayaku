"""Run a trained detector from its checkpoint: `from_pretrained` and `Predictor`."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from mayaku.backends.device import Device
from mayaku.data.geometry import fitted
from mayaku.inference.decode import (
    Classification,
    Decode,
    Detections,
    decode_candidates,
    decode_sidecar,
    pack_boxes,
)
from mayaku.inference.export.metadata import SUFFIX_TO_TARGET
from mayaku.inference.preprocess import (
    Canvases,
    ImageInput,
    letterbox_batch,
    network_input,
    planar_tensor,
    read_bgr,
)
from mayaku.inference.runner import Runner
from mayaku.model.blocks import STRIDES
from mayaku.utils.checkpoint import read_deploy_checkpoint

if TYPE_CHECKING:
    from mayaku.model import Model

__all__ = ["Predictor", "from_pretrained"]


def _cuda_fp16_enabled(device: torch.device) -> bool:
    """Whether single-image CUDA predictions use the FP16 NHWC CUDA-graph fast
    path: Volta/Turing+ GPUs with Tensor Cores, unless MAYAKU_TORCH_FP16=0."""
    if device.type != "cuda":
        return False
    if os.environ.get("MAYAKU_TORCH_FP16", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool(torch.cuda.get_device_capability(device) >= (7, 0))


class _CudaFastRunner:
    """Batch-1 FP16 NHWC + pinned upload + CUDA-graph runner for a deployed
    PyTorch model on CUDA. Keeps the parent `Predictor.model` untouched in fp32
    so `evaluate()` and `export()` stay bit-exact."""

    def __init__(self, model: Model, canvas: tuple[int, int], device: torch.device):
        self._device = device
        self._model: Model = cast(Any, copy.deepcopy(model).eval().half()).to(
            device, memory_format=torch.channels_last,
        )
        h, w = canvas
        self._host = torch.empty((1, h, w, 3), dtype=torch.uint8).pin_memory()
        self._pixels = torch.empty((1, h, w, 3), dtype=torch.uint8, device=device)
        self._input = torch.empty((1, 3, h, w), dtype=torch.float16, device=device,
                                  memory_format=torch.channels_last)
        self._uploaded = torch.cuda.Event()
        self._pending = False
        self._pad_place: tuple[int, int] | None = None
        self._box_graphs: dict[tuple[int, int, tuple[int, ...], int, bool],
                               tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self._maps_graph: tuple[torch.cuda.CUDAGraph, list[torch.Tensor]] | None = None

    def _wait(self) -> None:
        if self._pending:
            self._uploaded.synchronize()
            self._pending = False

    def canvases(self) -> Canvases:
        self._wait()
        return self._host.numpy()[:1]

    def need_pad(self, placement: tuple[int, int]) -> bool:
        if self._pad_place == placement:
            return False
        self._pad_place = placement
        return True

    def _stage_input(self, x: Canvases) -> None:
        if x.ctypes.data != self._host.data_ptr():
            self._pad_place = None
            self._wait()
            self._host.numpy()[:len(x)] = x

    def _run_model(self) -> list[torch.Tensor]:
        planar_tensor(self._pixels, out=self._input)
        return cast(list[torch.Tensor], self._model(self._input))

    def forward_boxes(self, x: Canvases, nc: int, reg_max: int, strides: tuple[int, ...],
                      topk: int, multi_label: bool) -> torch.Tensor:
        """Fused CUDA graph: planar RGB fp16 NHWC, model forward, top-K + DFL
        decode and async copy of ``(K, 6)`` candidates to pinned CPU memory."""
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

    def _capture_boxes(self, nc: int, reg_max: int, strides: tuple[int, ...], topk: int,
                       multi_label: bool) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        n_maps = 2 * len(strides)
        with torch.cuda.device(self._device):
            warm = self._run_model()
            n_anchors = sum(m.shape[2] * m.shape[3] for m in warm[:n_maps:2])
            k = min(topk, n_anchors * nc if multi_label else n_anchors)
            host_out = torch.empty((k, 6), dtype=torch.float32).pin_memory()
            pack_boxes(warm[:n_maps:2], warm[1:n_maps:2], nc, reg_max, strides, k, multi_label,
                       host_out)
            torch.cuda.current_stream(self._device).synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = self._run_model()
                pack_boxes(outs[:n_maps:2], outs[1:n_maps:2], nc, reg_max, strides, k, multi_label,
                           host_out)
        return graph, host_out

    def forward_maps(self, x: Canvases) -> list[torch.Tensor]:
        """Fused CUDA graph returning fp32 output maps on CUDA (for seg,
        keypoints, OBB and classification single-image predictions)."""
        if self._maps_graph is None:
            with torch.cuda.device(self._device):
                self._run_model()
                torch.cuda.current_stream(self._device).synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outs = self._run_model()
            self._maps_graph = (graph, outs)
        graph, outs = self._maps_graph
        self._stage_input(x)
        with torch.cuda.device(self._device):
            self._pixels.copy_(self._host, non_blocking=True)
            self._uploaded.record(torch.cuda.current_stream(self._device))
            self._pending = True
            graph.replay()
        return [o.float() for o in outs]


class Predictor(Runner):
    """A trained detector's deploy graph (`Detector.for_deploy`) in Torch,
    with its sidecar: architecture, canvas, classes and decode all come from
    the checkpoint."""

    def __init__(self, model: Model, sidecar: Mapping[str, Any], device: str = "auto"):
        super().__init__(sidecar, "Predictor")
        self.device = torch.device(Device.resolve(device))
        self.model = model.for_deploy().to(self.device)
        self._cuda_fast: _CudaFastRunner | None = None

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> Predictor:
        """Rebuild the model its sidecar describes (quantization-aware when it
        was trained so) and load the weights strictly."""
        sidecar, cfg, state = read_deploy_checkpoint(Path(path))
        model = cfg.model.build(cfg.input.canvas, len(sidecar["class_names"]))
        model.load_state_dict(state, strict=True)
        return cls(model, sidecar, device)

    def _predict_one(self, image: ImageInput, d: Decode | None,
                     decode_mode: str) -> Detections | Classification | None:
        if decode_mode != "predict" or not _cuda_fp16_enabled(self.device):
            return None
        if self._cuda_fast is None:
            self._cuda_fast = _CudaFastRunner(self.model, self.canvas, self.device)
        fast = self._cuda_fast
        bgr = read_bgr(image)
        ch, cw = self.canvas
        h, w = bgr.shape[:2]
        nh, nw = fitted((h, w), min(ch / h, cw / w))
        need_pad = fast.need_pad((nh, nw))
        x, metas = letterbox_batch([bgr], self.canvas, fast.canvases(), pad=need_pad)
        if (d is not None and self.sidecar["mask"] is None
                and self.sidecar["keypoints"] is None and not self.sidecar.get("obb")
                and 0.0 < d.conf < 1.0):
            dec = self.sidecar["decode"]
            strides = tuple(dec.get("strides", STRIDES))
            packed = fast.forward_boxes(x, len(self.class_names), dec["reg_max"],
                                        strides, d.topk, d.multi_label)
            return decode_candidates(packed, metas[0], d)
        maps = fast.forward_maps(x)
        return decode_sidecar(maps, metas, self.sidecar, d.conf if d is not None else None,
                              None, "predict")[0]

    def _forward(self, x: Canvases) -> list[torch.Tensor]:
        return cast(list[torch.Tensor], self.model(network_input(x, self.device)))

    def export(self, target: str = "onnx", output: str | Path | None = None,
               precision: str | None = None) -> Path:
        """Write a deployable artifact with this model's sidecar embedded;
        `precision` defaults to the target's first. See
        `mayaku.inference.export.export`."""
        from mayaku.inference.export import export

        return export(self.model, self.sidecar, target, output or "model", precision)

def from_pretrained(source: str | Path, device: str = "auto") -> Runner:
    """Load a deployable detector: a checkpoint path or model name gives a
    `Predictor`; an exported artifact gives an `ArtifactPredictor` run by its
    own runtime. Both are called with an image and return `Detections`."""
    if Path(source).suffix.lower() in SUFFIX_TO_TARGET:
        from mayaku.inference.artifact import ArtifactPredictor

        return ArtifactPredictor(source, device)
    from mayaku.utils.download import resolve_weights

    return Predictor.from_checkpoint(resolve_weights(source), device)
