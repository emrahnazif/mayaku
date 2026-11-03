"""The inference input boundary: BGR in, and exactly the network input
training produces from the same image."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mayaku.data.batch import batch_to, to_tensor
from mayaku.data.geometry import letterbox
from mayaku.inference.preprocess import letterbox_batch, network_input, planar, read_bgr


@pytest.fixture
def bgr():
    return np.random.default_rng(0).integers(0, 256, (97, 131, 3), dtype=np.uint8)


def test_the_runtime_input_is_trainings(bgr) -> None:
    """letterbox -> to_tensor -> batch_to is how training feeds an image; the
    runtime's BGR canvas through `planar` / `network_input` gives the same
    floats, on the host and on the device."""
    want = batch_to(to_tensor(letterbox(bgr, (128, 160))[0])[None], "cpu")
    canvases, metas = letterbox_batch([bgr], (128, 160))
    assert canvases.shape == (1, 128, 160, 3) and canvases.dtype == np.uint8
    assert torch.equal(torch.from_numpy(planar(canvases)), want)
    assert torch.equal(network_input(canvases, torch.device("cpu")), want)
    if torch.cuda.is_available():
        assert torch.allclose(network_input(canvases, torch.device("cuda")).cpu(), want)
    assert metas[0]["shape"] == (97, 131)


def test_inputs_are_read_as_bgr(bgr, tmp_path) -> None:
    import cv2
    from PIL import Image

    path = tmp_path / "x.png"
    cv2.imwrite(str(path), bgr)
    assert np.array_equal(read_bgr(path), bgr)                                  # a path
    assert np.array_equal(read_bgr(bgr), bgr)                                   # a BGR array
    assert np.array_equal(read_bgr(Image.fromarray(bgr[:, :, ::-1])), bgr)     # PIL: RGB by type
    assert read_bgr(bgr.astype(np.float32)).dtype == np.uint8
    with pytest.raises(ValueError, match="BGR array"):
        read_bgr(bgr[:, :, 0])
