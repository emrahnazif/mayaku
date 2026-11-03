"""Run an exported artifact with its own runtime: `ArtifactPredictor`.

The artifact carries everything in its embedded sidecar, so no checkpoint or
config is needed; the graph runs in the runtime and the decode on the host,
exactly as a deployment would.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch

from mayaku.inference.export import open_session
from mayaku.inference.export.metadata import read_sidecar, target_from_suffix
from mayaku.inference.preprocess import ImageInput, letterbox_rgb
from mayaku.inference.runner import Runner
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
        self._rgb = bool(getattr(self._session, "takes_rgb", False))

    def _prepare(self, images: Sequence[ImageInput]) -> tuple[Any, list[dict[str, Any]]]:
        """A runtime that takes interleaved RGB pixels (a Core ML image input)
        gets each letterboxed image as it is, skipping the planar batch the
        other runtimes need."""
        if not self._rgb:
            return super()._prepare(images)
        canvases, metas = zip(*(letterbox_rgb(image, self.canvas) for image in images), strict=True)
        return list(canvases), list(metas)

    def _forward(self, x: Any) -> list[torch.Tensor]:
        """A planar batch, run in chunks of the session's batch size, or a list
        of RGB canvases (`_prepare`), run one per call."""
        if self._rgb:
            run, parts = cast(Any, self._session).run_rgb, x
        else:
            step = self._session.batch or len(x)
            run, parts = self._session, [x[i:i + step] for i in range(0, len(x), step)]
        if len(parts) == 1:
            return cast(list[torch.Tensor], run(parts[0]))
        # a session may hand back its own buffers, overwritten by the next run
        runs = [[m.clone() for m in run(p)] for p in parts]
        return [torch.cat(m) for m in zip(*runs, strict=True)]
