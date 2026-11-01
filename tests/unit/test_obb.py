"""Oriented boxes: the maths against OpenCV, the targets the dataset emits,
the evaluator on exact answers, the graph contract, and a one-image overfit."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random

import cv2
import numpy as np
import pytest
import torch

from mayaku.data import DEFAULT_AUG, CocoDetection
from mayaku.engine.evaluation import coco_ap, score, to_coco
from mayaku.inference.decode import Detections
from mayaku.model import TINY, Detector, load_pretrained, obb
from mayaku.model.contract import DEPLOY_OPS, export_onnx

OBB = dataclasses.replace(TINY, obb=True)


def random_rboxes(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cat((torch.rand(n, 2, generator=g) * 200, torch.rand(n, 2, generator=g) * 80 + 5,
                      (torch.rand(n, 1, generator=g) - 0.5) * 2 * math.pi), 1)


def cv2_iou(a, b):
    ra = ((a[0], a[1]), (a[2], a[3]), math.degrees(a[4]))
    rb = ((b[0], b[1]), (b[2], b[3]), math.degrees(b[4]))
    _, pts = cv2.rotatedRectangleIntersection(ra, rb)
    inter = cv2.contourArea(cv2.convexHull(pts)) if pts is not None and len(pts) > 2 else 0.0
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def test_rotated_iou_matches_opencv() -> None:
    a = random_rboxes(30, 0)
    b = random_rboxes(30, 1)
    b[:, :2] = a[:, :2] + torch.randn(30, 2, generator=torch.Generator().manual_seed(2)) * 20
    ours = obb.rotated_iou(a, b)
    ref = torch.tensor([[cv2_iou(x, y) for y in b.tolist()] for x in a.tolist()])
    assert (ours - ref).abs().max() < 1e-3 and (ref > 0).sum() > 50


def canonical(r):
    """xywhr -> the same boxes with w >= h and theta in the readout's range,
    the form `obb.from_points` emits."""
    w, h, t = r[..., 2], r[..., 3], r[..., 4]
    swap = w < h
    t = torch.where(swap, t + math.pi / 2, t)
    return torch.stack((r[..., 0], r[..., 1], torch.where(swap, h, w), torch.where(swap, w, h),
                        torch.remainder(t - obb.ANGLE_LO, math.pi) + obb.ANGLE_LO), -1)


def test_canonical_form_is_the_same_box() -> None:
    r = random_rboxes(50)
    c = canonical(r)
    assert (c[:, 2] >= c[:, 3]).all()
    assert ((c[:, 4] >= obb.ANGLE_LO) & (c[:, 4] < obb.ANGLE_LO + math.pi)).all()
    assert (obb.rotated_iou(r, c).diag() - 1).abs().max() < 1e-4
    assert (obb.probiou(r, c) - 1).abs().max() < 2e-3


def test_side_distances_round_trip() -> None:
    r = canonical(random_rboxes(50))
    points = r[:, :2] + torch.rand(50, 2) - 0.5          # inside every box
    stride = torch.full((50, 1), 8.0)
    back = obb.decode(obb.encode(r, points, stride, 64) * stride, points, r[:, 4])
    assert (back - r).abs().max() < 1e-3


def test_polygon_fit_and_hull() -> None:
    wide, tall = obb.from_points([np.float32([[0, 0], [10, 0], [10, 4], [0, 4]]),
                                  np.float32([[0, 0], [4, 0], [4, 10], [0, 10]])])
    assert np.allclose(wide, [5, 2, 10, 4, 0], atol=1e-5)
    assert np.allclose(tall, [2, 5, 10, 4, math.pi / 2], atol=1e-5)
    r = random_rboxes(20)
    c = obb.corners(r)
    assert torch.allclose(torch.cat((c.amin(1), c.amax(1)), 1), obb.to_xyxy(r), atol=1e-3)


def test_nms_keeps_one_per_overlapping_pair_and_class() -> None:
    r = torch.tensor([[50, 50, 40, 10, 0.0], [51, 50, 40, 10, 0.05],
                      [51, 50, 40, 10, 0.05], [150, 50, 40, 10, 0.0]])
    keep = obb.nms(r, torch.tensor([0.9, 0.8, 0.85, 0.7]), torch.tensor([0, 0, 1, 0]), 0.7)
    assert sorted(keep.tolist()) == [0, 2, 3]


def rotated_coco(root, n=6, nc=3, seed=0):
    """`n` images of filled rotated rectangles; each annotation's polygon is
    the rectangle's corners and its bbox their hull."""
    rng = random.Random(seed)
    os.makedirs(root, exist_ok=True)
    images, anns = [], []
    for i in range(n):
        h, w = 300, 400
        img = np.full((h, w, 3), 30, np.uint8)
        for _ in range(3):
            r = torch.tensor([rng.uniform(80, w - 80), rng.uniform(80, h - 80),
                              rng.uniform(30, 90), rng.uniform(10, 30), rng.uniform(-1.5, 1.5)])
            c = rng.randrange(nc)
            pts = obb.corners(r).numpy()
            cv2.fillPoly(img, [np.round(pts).astype(np.int32)], (80 * c + 60, 200 - 60 * c, 40 * c))
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c + 1,
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "area": float(r[2] * r[3]), "iscrowd": 0,
                         "segmentation": [pts.reshape(-1).tolist()]})
        cv2.imwrite(os.path.join(root, "%03d.jpg" % i), img)
        images.append({"id": i, "file_name": "%03d.jpg" % i, "height": h, "width": w})
    path = os.path.join(root, "rot.json")
    with open(path, "w") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": c + 1, "name": "c%d" % c} for c in range(nc)]}, f)
    return path


@pytest.fixture(scope="module")
def rot(tmp_path_factory):
    root = tmp_path_factory.mktemp("rot")
    return str(root), rotated_coco(str(root))


def test_ground_truth_through_the_letterbox_scores_ap_1(rot) -> None:
    """The eval path's oriented boxes, mapped back to the image as the host
    decode maps predictions, score AP 1 under the rotated evaluator."""
    root, ann = rot
    ds = CocoDetection(root, ann, 256, obb=True)
    dets = []
    for i in range(len(ds)):
        it = ds[i]
        lab, m = it["labels"], it["meta"]
        assert lab.shape[1] == 10
        rb = lab[:, 5:].clone()
        rb[:, :2] = (rb[:, :2] - torch.tensor(m["pad"], dtype=rb.dtype)) / m["ratio"]
        rb[:, 2:4] /= m["ratio"]
        d = Detections(obb.to_xyxy(rb), torch.ones(len(lab)), lab[:, 0].long(), rboxes=rb)
        dets.append((m["id"], d))
    assert score(ann, dets, ds.cat_ids)["AP"] > 0.99
    # and an angle that is off by 20 degrees is not a match at IoU 0.75
    for _, d in dets:
        d.rboxes[:, 4] += math.radians(20)
    assert score(ann, dets, ds.cat_ids)["AP75"] < 0.5


@pytest.mark.parametrize("aug", [DEFAULT_AUG, dataclasses.replace(DEFAULT_AUG, vflip=0.5)])
def test_training_targets_follow_the_polygons(rot, aug) -> None:
    """Under mosaic, scale and flips every oriented box stays a
    canonical box centred on the canvas, and the part of it on the canvas
    lies inside the row's (warped, clipped) box. A box cut by the canvas
    edge is the minimum rectangle of the clipped polygon, which may reach a
    little past the edge."""
    root, ann = rot
    ds = CocoDetection(root, ann, 256, aug=aug, obb=True, seed=3)
    n = 0
    for i in range(12):
        lab = ds[i % len(ds)]["labels"]
        if not len(lab):
            continue
        rb = lab[:, 5:]
        assert (rb[:, 2] >= rb[:, 3] - 1e-4).all()
        assert ((rb[:, :2] >= 0) & (rb[:, :2] <= 256)).all()
        hull = obb.to_xyxy(rb).clamp(0, 256)
        assert (hull[:, :2] >= lab[:, 1:3] - 2).all() and (hull[:, 2:] <= lab[:, 3:5] + 2).all()
        n += len(lab)
    assert n > 10


def test_graph_stays_in_the_contract_and_warm_starts(tmp_path) -> None:
    m = Detector(OBB, 4)
    names = m.out_names
    assert names[6:9] == ["ang0", "ang1", "ang2"]
    inv = export_onnx(m.fuse(), str(tmp_path / "obb.onnx"), 128)
    assert set(inv) <= DEPLOY_OPS
    # a plain detector's weights warm-start an oriented one, and back
    info = load_pretrained(Detector(OBB, 4), Detector(TINY, 4).state_dict())
    assert all(k.startswith("head.ang") for k in info["missing"])
    load_pretrained(Detector(TINY, 4), Detector(OBB, 4).state_dict())


def test_oriented_records_carry_their_box() -> None:
    d = Detections(torch.tensor([[0.0, 0, 10, 4]]), torch.tensor([0.5]), torch.tensor([0]),
                   rboxes=torch.tensor([[5.0, 2, 10, 4, 0]]))
    rec = to_coco(d, 1, [7])
    assert rec[0]["rbox"] == [5.0, 2.0, 10.0, 4.0, 0.0] and rec[0]["category_id"] == 7


def test_rotated_ap_is_order_and_threshold_aware(tmp_path) -> None:
    """Two objects, two predictions: the right one and one shifted by half
    its length (rotated IoU 1/3): AP50 counts one of two objects."""
    path = tmp_path / "g.json"
    seg = obb.corners(torch.tensor([[50.0, 50, 40, 10, 0.6], [150.0, 120, 30, 12, -0.4]]))
    anns = [{"id": k + 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 400.0,
             "iscrowd": 0, "segmentation": [seg[k].reshape(-1).tolist()]} for k in range(2)]
    path.write_text(json.dumps({"images": [{"id": 0, "file_name": "x", "height": 200, "width": 200}],
                                "annotations": anns, "categories": [{"id": 1, "name": "a"}]}))
    res = [{"image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.9,
            "rbox": [50.0, 50, 40, 10, 0.6]},
           {"image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.8,
            "rbox": [150.0 + 15 * math.cos(-0.4), 120 + 15 * math.sin(-0.4), 30, 12, -0.4]}]
    ap50 = coco_ap(str(path), res)["AP50"]
    assert 0.45 < ap50 < 0.55


@pytest.mark.slow
def test_oriented_branch_learns() -> None:
    """One image of three rotated objects: after training, each object's
    best prediction overlaps it by rotated IoU above 0.5."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)            # one thread: the same run every time
    try:
        _fit_three_rotated_objects()
    finally:
        torch.set_num_threads(threads)


def _fit_three_rotated_objects():
    from mayaku.engine.loss import DetectionLoss
    from mayaku.inference.decode import Decode, postprocess

    torch.manual_seed(0)
    gt = torch.tensor([[60.0, 60, 60, 16, 0.5], [150, 150, 90, 30, -0.7], [190, 60, 50, 20, 1.2]])
    gt = canonical(gt)
    x = torch.zeros(1, 3, 256, 256)
    img = np.zeros((256, 256, 3), np.float32)
    for k, r in enumerate(gt):
        ch = np.zeros((256, 256), np.float32)
        cv2.fillPoly(ch, [np.round(obb.corners(r).numpy()).astype(np.int32)], 1.0)
        img[..., k] = ch
    x[0] = torch.from_numpy(img).permute(2, 0, 1)
    targets = torch.cat((torch.zeros(3, 1), torch.arange(3.0)[:, None], obb.to_xyxy(gt), gt), 1)
    model = Detector(OBB, 4)
    crit = DetectionLoss(nc=4, reg_max=OBB.reg_max, warmup=3, obb=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for step in range(300):
        total, _ = crit(model(x), targets, epoch=step // 20)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
    # scored with the batch statistics it trained on: a network fitted to
    # one image is so sensitive that eval mode's unbiased variance (64/63 on
    # the stride-32 grid) moves its coarse logits by tens, and what is under
    # test is the oriented branch, not BatchNorm
    with torch.no_grad():
        preds = model(x)
    g = model.split(preds)
    dets, _, _, _ = postprocess(preds, 4, OBB.reg_max, Decode(conf=0.05), ang=g["ang"])
    d = dets[0]
    for k in range(3):
        mine = d[d[:, 5] == k]
        assert len(mine), "no prediction for object %d" % k
        best = obb.rotated_iou(mine[:1, 6:11], gt[k:k + 1])[0, 0]
        assert best > 0.5, (k, float(best))
