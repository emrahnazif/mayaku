"""Images in, letterboxed canvases out: the one place an image enters
inference, shared by the Predictor and every exported-artifact runtime.

The pipeline is BGR, OpenCV's order, end to end: an image path is decoded with
OpenCV as in training, a numpy array is taken to be BGR as `cv2.imread` and
`cv2.VideoCapture` give it, and a PIL image (RGB by its type) is converted
once. The letterbox is training's own (`mayaku.data.geometry.letterbox`), so
the canvas holds the pixels training saw. The network itself reads planar RGB
in [0, 1] (`mayaku.data.geometry.PREPROCESS`, recorded in every sidecar);
each runtime makes that conversion where it is cheapest, with the channel
reversal folded into the transpose it needs anyway: `planar` on the host,
`planar_tensor` on a device.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from mayaku.data.geometry import letterbox

ImageInput: TypeAlias = str | Path | npt.NDArray[Any] | Image.Image
#: (B, H, W, 3) uint8 BGR letterboxed canvases, what every runtime session takes
Canvases: TypeAlias = npt.NDArray[np.uint8]


def read_bgr(image: ImageInput) -> npt.NDArray[np.uint8]:
    """A path, a PIL image or an (H, W, 3) BGR array -> (H, W, 3) uint8 BGR."""
    if isinstance(image, str | Path):
        bgr = cv2.imread(str(image))          # IMREAD_COLOR: (H, W, 3) uint8 BGR
        if bgr is None:
            raise FileNotFoundError(f"cannot read image {image}")
        return cast(npt.NDArray[np.uint8], bgr)
    if isinstance(image, Image.Image):        # RGB by its type
        rgb = np.asarray(image if image.mode == "RGB" else image.convert("RGB"))
        return cast(npt.NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("expected an image path, a PIL image or an (H, W, 3) BGR array; "
                         f"got an array of shape {arr.shape}")
    return arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)


def letterbox_batch(images: Sequence[ImageInput], canvas: tuple[int, int]
                    ) -> tuple[Canvases, list[dict[str, Any]]]:
    """Images -> (BGR canvases, letterbox metas {"ratio", "pad", "shape"})."""
    out, metas = [], []
    for image in images:
        bgr = read_bgr(image)
        img, r, pad = letterbox(bgr, canvas)
        out.append(img)
        metas.append({"ratio": r, "pad": pad, "shape": bgr.shape[:2]})
    # one image, the common case, is a view: no copy into a batch array
    return (out[0][None] if len(out) == 1 else np.stack(out)), metas


def network_input(canvases: Canvases, device: torch.device) -> torch.Tensor:
    """BGR canvases -> the network's planar RGB float input on `device`: on
    the host in one pass (`planar`), or sent as bytes and converted where the
    network runs (`planar_tensor`)."""
    if device.type == "cpu":
        return torch.from_numpy(planar(canvases))
    return planar_tensor(torch.from_numpy(canvases).to(device, non_blocking=True))


def planar(canvases: Canvases) -> npt.NDArray[np.float32]:
    """BGR canvases -> the network's (B, 3, H, W) float32 RGB in [0, 1], in
    one pass: the transpose and the channel reversal are a strided view the
    division reads through."""
    b, h, w, _ = canvases.shape
    out = np.empty((b, 3, h, w), np.float32)
    # divided, not multiplied by 1/255: bit-identical to training's `batch_to`
    np.divide(canvases[..., ::-1].transpose(0, 3, 1, 2), np.float32(255), out=out)
    return out


def planar_tensor(px: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """`planar` for a uint8 BGR canvas tensor already on its device, into
    `out` when given (a TensorRT engine's input buffer)."""
    return torch.div(px.permute(0, 3, 1, 2).flip(1), 255, out=out)
