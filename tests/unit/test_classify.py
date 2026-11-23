"""The classifier: class-folder data, the five-op graph, warm starts from a
detector, and the whole train -> predict -> evaluate -> export path."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
import torch

from mayaku.data import CLEAN_AUG, DEFAULT_AUG, collate
from mayaku.data.folder import FolderDataset, load_folder
from mayaku.engine.loss import ClassifyLoss
from mayaku.inference.decode import Classification
from mayaku.model import TINY, Detector, enable_qat
from mayaku.model.classify import Classifier, load_backbone, logits
from mayaku.model.contract import DEPLOY_OPS, export_onnx

COLOURS = ((40, 40, 220), (40, 200, 40), (220, 60, 40))


def class_folders(root, n=6, split="train"):
    """Three classes told apart by colour and a shape, `n` images each, of
    assorted sizes."""
    rng = np.random.default_rng(0 if split == "train" else 1)
    for c, colour in enumerate(COLOURS):
        d = os.path.join(root, split, "class_%d" % c)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            h, w = int(rng.integers(90, 200)), int(rng.integers(90, 200))
            img = np.full((h, w, 3), 30, np.uint8)
            cv2.circle(img, (w // 2, h // 2), min(h, w) // 3, colour, -1) if c != 1 else \
                cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), colour, -1)
            cv2.imwrite(os.path.join(d, "%02d.jpg" % i), img)
    return os.path.join(root, split)


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("cls"))
    return class_folders(root, 6, "train"), class_folders(root, 3, "val")


def test_folders_parse_in_class_order(folders) -> None:
    train, val = folders
    data = load_folder(train)
    assert data.class_names == ["class_0", "class_1", "class_2"] and len(data.files) == 18
    assert load_folder(val, data.class_names).labels.tolist() == [0] * 3 + [1] * 3 + [2] * 3
    with pytest.raises(ValueError):
        load_folder(train, ["class_0", "class_1"])     # a class the model lacks


@pytest.mark.parametrize("aug", [None, DEFAULT_AUG, CLEAN_AUG])
def test_items_collate_like_detection_samples(folders, aug) -> None:
    ds = FolderDataset(folders[0], 96, aug=aug)
    imgs, targets, metas, masks = collate([ds[i] for i in (0, 7, 15)])
    assert imgs.shape == (3, 3, 96, 96) and imgs.dtype == torch.uint8 and masks is None
    assert targets.tolist() == [[0, 0], [1, 1], [2, 2]]
    assert (metas[0] is None) == (aug is not None)


def test_graph_is_in_the_contract_and_pools_on_the_host(tmp_path) -> None:
    m = Classifier(TINY, 5).eval()
    x = torch.rand(2, 3, 96, 128)
    out = m(x)
    assert len(out) == 1 and out[0].shape == (2, 5, 3, 4)
    inv = export_onnx(m.for_deploy(), str(tmp_path / "cls.onnx"), (96, 128))
    assert set(inv) <= DEPLOY_OPS
    with torch.no_grad():
        assert torch.allclose(logits(m.for_deploy()(x)), logits(out), atol=1e-4)
    enable_qat(Classifier(TINY, 5))           # the library's default trains int8-aware


def test_backbone_warm_starts_from_a_detector() -> None:
    det = Detector(TINY, 7)
    m = Classifier(TINY, 3)
    info = load_backbone(m, det.state_dict())
    assert not info["head"]
    assert all(torch.equal(v, det.state_dict()[k]) for k, v in m.state_dict().items()
               if k.startswith("backbone."))
    # a classifier of the same class count carries its head over
    src = Classifier(TINY, 3)
    assert load_backbone(m, src.state_dict())["head"]
    assert torch.equal(m.logits.weight, src.logits.weight)


@pytest.mark.slow
def test_classifier_fits_its_training_images(folders) -> None:
    torch.manual_seed(0)
    ds = FolderDataset(folders[0], 96)
    imgs, targets, _, _ = collate([ds[i] for i in range(len(ds))])
    x = imgs.float() / 255
    m = Classifier(TINY, 3)
    crit = ClassifyLoss(3)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(60):
        loss, _ = crit(m(x), targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        acc = (logits(m(x)).argmax(1) == targets[:, 1].long()).float().mean()
    assert acc == 1.0, float(acc)


@pytest.mark.slow
def test_train_predict_evaluate_export(folders, tmp_path) -> None:
    """The library path a user takes: train from class folders, predict one
    image, score the val folders through the checkpoint and through its
    ONNX export, the two agreeing."""
    from mayaku import evaluate, train
    from mayaku.inference import Predictor, from_pretrained

    train_dir, val_dir = folders
    res = train(None, train_images=train_dir, val_images=val_dir, output_dir=tmp_path / "run",
                num_epochs=2, device="cpu",
                overrides={"model": {"task": "classify", "tier": "n", "qat": False},
                           "input": {"size_budget": 96},
                           "train": {"batch": 6, "recalibrate_images": 12, "warmup_epochs": 0.0},
                           "dataloader": {"num_workers": 0}}, log=lambda *_: None)
    assert set(res["metrics"]) == {"top1", "top5"} and res["metrics"]["top5"] == 1.0
    p = Predictor.from_checkpoint(res["final_weights"], device="cpu")
    one = p(os.path.join(val_dir, "class_1", "00.jpg"))
    assert isinstance(one, Classification) and len(one.scores) == 3
    assert abs(float(one.scores.sum()) - 1) < 1e-4
    onnx = p.export("onnx", tmp_path / "cls.onnx")
    assert from_pretrained(str(onnx), device="cpu").sidecar["task"] == "classify"
    a = evaluate(res["final_weights"], images=val_dir, device="cpu", log=lambda *_: None)
    b = evaluate(str(onnx), images=val_dir, device="cpu", log=lambda *_: None)
    assert a == b == res["metrics"]
