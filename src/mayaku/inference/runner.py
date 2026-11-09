"""What every deployed detector shares: its sidecar, and images in ->
`Detections` out through the one preprocessing and the one decode."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from mayaku.inference.decode import Classification, Detections, decode_sidecar
from mayaku.inference.preprocess import Canvases, ImageInput, letterbox_batch
from mayaku.utils.checkpoint import check_sidecar


class Runner:
    """Call with an image path, a PIL image or an (H, W, 3) BGR array (OpenCV's
    order, as `cv2.imread` and `cv2.VideoCapture` give it) and get
    `Detections` in that image's pixels; `batch` takes several. A subclass
    supplies `_forward`: (B, H, W, 3) uint8 BGR canvases (`letterbox_batch`)
    -> the raw maps in `sidecar["outputs"]` order."""

    def __init__(self, sidecar: Mapping[str, Any], source: str):
        self.sidecar = dict(check_sidecar(sidecar, source))

    @property
    def canvas(self) -> tuple[int, int]:
        return tuple(self.sidecar["canvas_hw"])

    @property
    def class_names(self) -> list[str]:
        return list(self.sidecar["class_names"])

    def _forward(self, x: Canvases) -> list[torch.Tensor]:
        raise NotImplementedError

    def _canvases(self, n: int) -> Canvases | None:
        """Where `n` images are letterboxed: a runtime's own input buffer, or
        None for a new array."""
        return None

    def __call__(self, image: ImageInput,
                 conf: float | None = None) -> Detections | Classification:
        return self.batch([image], conf)[0]

    @torch.no_grad()
    def batch(self, images: Sequence[ImageInput], conf: float | None = None,
              encode_mask: Callable[..., list[Any]] | None = None
              ) -> list[Detections] | list[Classification]:
        """Several images in one pass. `conf` overrides the recorded score
        threshold; `encode_mask` hands masks over encoded (see
        `mayaku.inference.decode.decode`), as the evaluator's RLE."""
        x, metas = letterbox_batch(images, self.canvas, self._canvases(len(images)))
        return decode_sidecar(self._forward(x), metas, self.sidecar, conf, encode_mask)
