"""Images in, letterboxed uint8 batches out: the preprocessing half of the
input contract (`mayaku.data.geometry.PREPROCESS`), shared by the Predictor
and the exported-artifact runner and pixel-identical to training's -- the same
OpenCV decode, the same `letterbox`, the same `to_tensor`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from mayaku.data.batch import to_tensor
from mayaku.data.geometry import letterbox

ImageInput = str | Path | npt.NDArray[Any]


def _imread(path: str | Path) -> npt.NDArray[Any]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return bgr


def _rgb_uint8(image: ImageInput) -> npt.NDArray[Any]:
    """An (H, W, 3) RGB array, checked, as uint8."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB array or an image path; got shape {arr.shape}")
    return arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)


def read_bgr(image: ImageInput) -> npt.NDArray[Any]:
    """A path (decoded with OpenCV, as in training) or an (H, W, 3) RGB array
    -> (H, W, 3) uint8 BGR, the layout training letterboxes."""
    if isinstance(image, str | Path):
        return _imread(image)
    return np.ascontiguousarray(_rgb_uint8(image)[:, :, ::-1])


def letterbox_batch(images: Sequence[ImageInput], canvas: tuple[int, int]
                    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Images -> ((B, 3, H, W) uint8 RGB, letterbox metas {"ratio", "pad",
    "shape"})."""
    out, metas = [], []
    for image in images:
        bgr = read_bgr(image)
        img, r, pad = letterbox(bgr, canvas)
        out.append(to_tensor(img))
        metas.append({"ratio": r, "pad": pad, "shape": bgr.shape[:2]})
    return (out[0][None] if len(out) == 1 else torch.stack(out)), metas


def letterbox_rgb(image: ImageInput, canvas: tuple[int, int]
                  ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """One image -> ((H, W, 3) uint8 RGB on the canvas, its letterbox meta),
    for a runtime that takes interleaved RGB pixels (a Core ML image input).
    The same pixels as `letterbox_batch`: resizing treats each channel alike
    and the padding is the same grey in all three, so letterboxing the RGB
    array directly skips the BGR copy and the planar round trip."""
    if isinstance(image, str | Path):
        src, bgr = _imread(image), True
    else:
        src, bgr = np.ascontiguousarray(_rgb_uint8(image)), False
    img, r, pad = letterbox(src, canvas)
    if bgr:      # converted on the canvas, not the (usually larger) source
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, {"ratio": r, "pad": pad, "shape": src.shape[:2]}
