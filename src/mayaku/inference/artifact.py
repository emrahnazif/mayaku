"""Run an exported artifact with its own runtime: `ArtifactPredictor`.

The artifact carries everything in its embedded sidecar, so no checkpoint or
config is needed; the graph runs in the runtime and the decode on the host,
exactly as a deployment would.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mayaku.data.geometry import fitted
from mayaku.inference.decode import Decode, Detections, decode_candidates
from mayaku.inference.export import open_session
from mayaku.inference.export.metadata import read_sidecar, target_from_suffix
from mayaku.inference.preprocess import Canvases, ImageInput, letterbox_batch, read_bgr
from mayaku.inference.runner import Runner
from mayaku.model.blocks import STRIDES
from mayaku.utils.checkpoint import check_sidecar

__all__ = ["ArtifactPredictor"]


class ArtifactPredictor(Runner):
    """An exported artifact (ONNX, Core ML, OpenVINO or TensorRT) run by its
    runtime, called like a `Predictor`."""

    def __init__(self, path: str | Path, device: str = "auto"):
        path = Path(path)
        super().__init__(check_sidecar(read_sidecar(path, target_from_suffix(path)), str(path)),
                         str(path))
        self._session = open_session(path, self.sidecar, device)
        if self._session.input_hw != self.canvas:
            raise ValueError(f"{path}: graph input {self._session.input_hw} disagrees with the "
                             f"sidecar canvas {self.canvas}")

    def _canvases(self, n: int) -> Canvases | None:
        return self._session.canvases(n)

    def _predict_one(self, image: ImageInput, d: Decode) -> Detections | None:
        from typing import cast

        from mayaku.inference.export.tensorrt import Session as TensorRTSession

        if (self.sidecar["mask"] is not None or self.sidecar["keypoints"] is not None
                or self.sidecar.get("obb")
                or not isinstance(cast(object, self._session), TensorRTSession)
                or self._session.batch != 1 or not (0.0 < d.conf < 1.0)):
            return None
        sess = cast(TensorRTSession, self._session)
        dec = self.sidecar["decode"]
        strides = tuple(dec.get("strides", STRIDES))
        bgr = read_bgr(image)
        ch, cw = self.canvas
        h, w = bgr.shape[:2]
        nh, nw = fitted((h, w), min(ch / h, cw / w))
        need_pad = sess.need_pad((nh, nw))
        x, metas = letterbox_batch([bgr], self.canvas, sess.box_canvases(), pad=need_pad)
        packed = sess.forward_boxes(x, len(self.class_names), dec["reg_max"],
                                    strides, d.topk, d.multi_label)
        return decode_candidates(packed, metas[0], d)

    def _forward(self, x: Canvases) -> list[torch.Tensor]:
        """BGR canvases, run in chunks of the session's batch size."""
        step = self._session.batch or len(x)
        parts = [x[i:i + step] for i in range(0, len(x), step)]
        if len(parts) == 1:
            return self._session(parts[0])
        # a session may hand back its own buffers, overwritten by the next run
        runs = [[m.clone() for m in self._session(p)] for p in parts]
        return [torch.cat(m) for m in zip(*runs, strict=True)]
