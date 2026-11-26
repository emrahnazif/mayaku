"""What every deployed detector shares: its sidecar, and images in ->
`Detections` out through the one preprocessing and the one decode."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from mayaku.inference.decode import (
    Classification,
    Decode,
    Detections,
    decode_sidecar,
    predict_decode,
)
from mayaku.inference.preprocess import Canvases, ImageInput, letterbox_batch
from mayaku.utils.checkpoint import check_sidecar


class Runner:
    """Call with an image path, a PIL image or an (H, W, 3) BGR array (OpenCV's
    order, as `cv2.imread` and `cv2.VideoCapture` give it) and get
    `Detections` in that image's pixels; `batch` takes several.

    A call decodes for an application with the sidecar's "predict" block,
    stated once as a warning; `decode="eval"` gives its evaluation decode,
    the one COCO AP is measured with (AP computed from predictions is lower:
    they leave out the low scores AP counts). A subclass
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

    def __call__(self, image: ImageInput, conf: float | None = None,
                 decode: str = "predict") -> Detections | Classification:
        return self._run([image], conf, None, decode)[0]

    def batch(self, images: Sequence[ImageInput], conf: float | None = None,
              encode_mask: Callable[..., list[Any]] | None = None, decode: str = "predict"
              ) -> list[Detections] | list[Classification]:
        """Several images in one pass. `decode` "predict" or "eval" (see the
        class); `conf` overrides the score threshold either way. `encode_mask`
        hands masks over encoded (see `mayaku.inference.decode.decode`), as
        the evaluator's RLE."""
        return self._run(images, conf, encode_mask, decode)

    def _predict_one(self, image: ImageInput, d: Decode) -> Detections | None:
        """Optional single-image fast path bypassing full raw-map decode."""
        return None

    def _run(self, images: Sequence[ImageInput], conf: float | None,
             encode_mask: Callable[..., list[Any]] | None, decode: str
             ) -> list[Detections] | list[Classification]:
        """`__call__` and `batch`, one frame below the caller, which the
        warning names (`stacklevel`)."""
        if decode not in ("predict", "eval"):
            raise ValueError(f'decode is "predict" or "eval", not {decode!r}')
        task = self.sidecar.get("task")
        if decode == "predict" and task != "classify":
            d = predict_decode(self.sidecar)
            warnings.warn(f"predictions use conf {d.conf:g}, NMS IoU {d.iou:g}"
                          f"{', multi-label' if d.multi_label else ''}; decode=\"eval\" gives the "
                          "COCO-AP decode", stacklevel=3)
        if len(images) == 1 and encode_mask is None and task != "classify":
            dec = Decode.from_sidecar(self.sidecar["decode"]) if decode == "eval" else predict_decode(self.sidecar)
            if conf is not None:
                dec = dataclasses.replace(dec, conf=conf)
            with torch.no_grad():
                fast = self._predict_one(images[0], dec)
            if fast is not None:
                return [fast]
        x, metas = letterbox_batch(images, self.canvas, self._canvases(len(images)))
        with torch.no_grad():
            return decode_sidecar(self._forward(x), metas, self.sidecar, conf, encode_mask, decode)
