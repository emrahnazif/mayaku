"""The host decode: raw output maps -> detections in original-image pixels.

Everything after the five-op graph lives here -- the sigmoid, the DFL
expectation, the score threshold, top-k, NMS (rotated, for oriented boxes),
the per-instance mask network and the keypoint snap -- so the Predictor, the exported-artifact runner and the
evaluator all run the same code, and evaluation measures exactly what a
deployment gets. Reading `postprocess` and `decode` with a `Decode` tells a
runtime author what to implement.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mayaku.backends.ops.nms import batched_nms
from mayaku.data.geometry import unletterbox
from mayaku.model import kpt as kptlib
from mayaku.model import mask as masklib
from mayaku.model import obb as obblib
from mayaku.model.blocks import STRIDES
from mayaku.model.box import (
    anchor_grid,
    decode_boxes,
    dfl_expectation,
    flatten_head,
    flatten_levels,
    ltrb_to_xyxy,
)
from mayaku.model.detector import split_outputs

__all__ = ["DEPLOY", "PREDICT", "Classification", "Decode", "Detections", "PredictDecode", "decode",
           "decode_candidates", "decode_sidecar", "host_side", "pack_boxes", "postprocess",
           "predict_decode"]


@dataclasses.dataclass(frozen=True)
class Decode:
    """What the host does after the graph. These numbers change AP without
    changing a weight, so they are recorded with every run and every artifact.

    `multi_label` True (the default) keeps every (anchor, class) pair above
    the threshold and takes the top-k over those pairs, as YOLO26's head
    does, so a second-choice class still earns recall: +0.7 COCO AP on a
    nano checkpoint (AR100 +4.3) for the same weights. False takes the best
    class per anchor, one comparison per anchor.
    """

    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    topk: int = 1000
    multi_label: bool = True

    @classmethod
    def from_sidecar(cls, block):
        """From a sidecar's "decode" or "predict" block (the former also
        carries strides and reg_max)."""
        return cls(**{f.name: block[f.name] for f in dataclasses.fields(cls)})


DEPLOY = Decode()

# A prediction decodes with the checkpoint's "predict" block, evaluation
# with its "decode" block (`DEPLOY`, for COCO AP). NMS 0.5 removes 94% of
# the second boxes on one object on the street nano's val, at -0.5 AP if
# evaluation used it, so evaluation keeps 0.7.
@dataclasses.dataclass(frozen=True)
class PredictDecode(Decode):
    """`Decode` with the prediction defaults, so a config that sets only some
    of `train.predict` keeps these for the rest."""

    conf: float = 0.4
    iou: float = 0.5
    multi_label: bool = False


PREDICT = PredictDecode()


def predict_decode(sidecar) -> Decode:
    """The `Decode` a prediction uses: the sidecar's "predict" block, or
    `PREDICT` for a checkpoint written before it existed."""
    return PredictDecode.from_sidecar(sidecar["predict"]) if sidecar.get("predict") else PREDICT


def host_side(preds: list[torch.Tensor]) -> list[torch.Tensor]:
    """Where the decode runs: on the GPU for CUDA, else on the CPU. The
    decode is many small ops with host syncs between them (thresholds,
    top-k, NMS, per-chunk mask sizes), which CUDA runs well and MPS does
    not: on an M1 Max, decoding a 1080p frame of an 800-canvas seg model
    takes 107 ms on MPS and 8.5 ms on the CPU, with identical results."""
    return [p if p.is_cuda else p.cpu() for p in preds]


# Most mask elements one chunk of `masklib.box_masks` computes at once (a
# float each, 512 MB): masks are computed inside their boxes, but a chunk
# of large boxes on a large image (Objects365 has 42-megapixel ones) would
# otherwise ask for tens of GB at once.
MASK_BUDGET = 1 << 27


@dataclasses.dataclass
class Detections:
    """One image's detections in its original pixels, on the CPU.

    boxes (n, 4) xyxy float32, scores (n,), labels (n,) int64 class indices;
    masks (n, h, w) bool when the model segments (or, when `decode` got an
    `encode_mask`, a list of its encoding of each mask); keypoints (n, K, 3) as
    (x, y, visibility probability) when it has keypoints; rboxes (n, 5) as
    (cx, cy, w, h, theta) when it predicts oriented boxes, `boxes` then
    being their axis-aligned hulls.
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    masks: torch.Tensor | None = None
    keypoints: torch.Tensor | None = None
    rboxes: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.scores)


@dataclasses.dataclass
class Classification:
    """One image's class probabilities (nc,), from a classifier; `label` is
    the most likely class index, `top(k)` the k best as (index, probability)
    pairs."""

    scores: torch.Tensor

    @property
    def label(self) -> int:
        return int(self.scores.argmax())

    def top(self, k: int = 5) -> list[tuple[int, float]]:
        p, i = self.scores.topk(min(k, len(self.scores)))
        return list(zip(i.tolist(), p.tolist(), strict=True))


def _nms_sorted_numpy(arr: npt.NDArray[np.float32], iou: float, max_det: int) -> list[int]:
    """Greedy class-aware NMS over rows ``(n, 6)`` already sorted descending by score."""
    n = len(arr)
    if n <= 1:
        return list(range(n)) if max_det > 0 else []
    x1, y1, x2, y2, lb = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 5]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []
    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        if len(keep) == max_det:
            break
        if i + 1 < n:
            j = slice(i + 1, n)
            ix1 = np.maximum(x1[i], x1[j])
            iy1 = np.maximum(y1[i], y1[j])
            ix2 = np.minimum(x2[i], x2[j])
            iy2 = np.minimum(y2[i], y2[j])
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            union = areas[i] + areas[j] - inter
            suppressed[j] |= (lb[j] == lb[i]) & (inter > iou * union)
    return keep


def pack_boxes(cls_maps: list[torch.Tensor], box_maps: list[torch.Tensor],
               nc: int, reg_max: int, strides: tuple[int, ...], k: int,
               multi_label: bool, host_out: torch.Tensor) -> None:
    """Batch-1 device top-K + DFL box decode -> async copy of ``(k, 6)``
    ``[x1, y1, x2, y2, logit, label]`` float32 candidates into pinned CPU
    ``host_out``. Safe to capture inside a CUDA graph."""
    cls_f = [m.float() for m in cls_maps]
    box_f = [m.float() for m in box_maps]
    points, stride = anchor_grid([tuple(c.shape[2:]) for c in cls_f], strides,
                                 device=cls_f[0].device, dtype=torch.float32)
    logits = torch.cat([m.reshape(nc, -1) for m in cls_f], 1)
    n_anchors = logits.shape[1]
    if multi_label:
        top, idx = logits.reshape(-1).topk(k)
        lab, anc = idx // n_anchors, idx % n_anchors
    else:
        best, labs = logits.max(0)
        top, anc = best.topk(k)
        lab = labs[anc]
    dist = torch.cat([m.reshape(4 * reg_max, -1) for m in box_f], 1)[:, anc].t()
    ltrb = dfl_expectation(dist, reg_max) * stride[anc]
    xyxy = ltrb_to_xyxy(ltrb, points[anc])
    host_out.copy_(torch.cat((xyxy, top[:, None], lab[:, None].float()), 1), non_blocking=True)


def decode_candidates(packed: torch.Tensor, meta: Mapping[str, Any], d: Decode) -> Detections:
    """Sorted ``(K, 6)`` ``[x1, y1, x2, y2, logit, label]`` CPU candidates
    (from a fused device top-K + DFL pass) -> ``Detections`` in original pixels."""
    arr = packed.numpy()
    floor = math.log(d.conf / (1.0 - d.conf))
    # arr[:, 4] is descending, so -arr[:, 4] is ascending; "left" enforces strictly > floor
    n = int(np.searchsorted(-arr[:, 4], -floor, side="left"))
    if n == 0 or d.max_det <= 0:
        return Detections(torch.zeros((0, 4), dtype=torch.float32),
                          torch.zeros((0,), dtype=torch.float32),
                          torch.zeros((0,), dtype=torch.int64))
    ratio, (left, top), (oh, ow) = meta["ratio"], meta["pad"], meta["shape"]
    if n <= 128:
        keep = _nms_sorted_numpy(arr[:n], d.iou, d.max_det)
        out = arr[keep]
        bx = out[:, :4].copy()
        bx[:, 0::2] = np.clip((bx[:, 0::2] - left) / ratio, 0.0, float(ow))
        bx[:, 1::2] = np.clip((bx[:, 1::2] - top) / ratio, 0.0, float(oh))
        logits = out[:, 4]
        scores = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)
        labels = out[:, 5].astype(np.int64)
        return Detections(torch.from_numpy(bx), torch.from_numpy(scores), torch.from_numpy(labels))
    sub = packed[:n]
    boxes_c = sub[:, :4]
    scores_t = sub[:, 4].sigmoid()
    labels_t = sub[:, 5].long()
    keep_t = batched_nms(boxes_c, scores_t, labels_t, d.iou)[:d.max_det]
    return Detections(unletterbox(boxes_c[keep_t], ratio, (left, top), (oh, ow)),
                      scores_t[keep_t], labels_t[keep_t])


def postprocess(preds, nc, reg_max, d=DEPLOY, ang=None, strides=STRIDES):
    """Head outputs -> (dets, idxs, points, stride): per image an (n, 6)
    [x1, y1, x2, y2, score, class] tensor in canvas pixels and the anchor
    index each surviving detection came from, plus the shared anchor points
    and strides. The indices are how the mask and keypoint heads find a
    detection's kernels / offsets without decoding the head a second time.
    `ang`, the angle maps of an oriented-box model, makes the boxes oriented: NMS runs
    on ProbIoU, and each row carries the box's xywhr after its class,
    [x1, y1, x2, y2, score, class, cx, cy, w, h, theta], the first four
    being the hull.

    Axis-aligned boxes are decoded for the surviving anchors only: the score
    threshold runs on the class logits first, so the DFL expectation and the
    box arithmetic touch tens of anchors instead of every one (8400 at
    640x640), through the same `decode_boxes` training uses.
    """
    if ang is not None:
        cls, dist, shapes = flatten_head(preds[:2 * len(strides)], nc, reg_max, contiguous=True)
    else:
        cls_maps = preds[:2 * len(strides):2]
        shapes = [tuple(c.shape[2:]) for c in cls_maps]
        cls = flatten_levels(cls_maps, nc)
        dist = None
    points, stride = anchor_grid(shapes, strides, device=cls.device, dtype=cls.dtype)
    rboxes = None if ang is None else obblib.decode_head(ang, dist, points, stride, reg_max)
    floor = math.log(d.conf / (1 - d.conf))
    out, idxs = [], []
    for b, logits in enumerate(cls):
        # the survivors' indices once: each boolean-mask index syncs the host
        if d.multi_label:
            anchor, lab = (logits > floor).nonzero(as_tuple=True)
            score = logits[anchor, lab].sigmoid()
        else:
            best, lab = logits.max(1)
            anchor = (best > floor).nonzero().squeeze(1)
            score, lab = best[anchor].sigmoid(), lab[anchor]
        if len(score) == 0:
            cols = 11 if rboxes is not None else 6
            out.append(torch.zeros((0, cols), device=cls.device, dtype=torch.float32))
            idxs.append(anchor)
            continue
        if len(score) > d.topk:
            keep = score.topk(d.topk).indices
            score, lab, anchor = score[keep], lab[keep], anchor[keep]
        if rboxes is not None:
            box = rboxes[b][anchor]
            keep = obblib.nms(box, score, lab, d.iou)[:d.max_det]
            out.append(torch.cat((obblib.to_xyxy(box[keep]), score[keep, None],
                                  lab[keep, None].float(), box[keep]), 1))
        else:
            if dist is None:
                dist = flatten_levels(preds[1:2 * len(strides):2], 4 * reg_max)
            box = decode_boxes(dist[b, anchor], points[anchor], stride[anchor], reg_max)
            keep = batched_nms(box, score, lab, d.iou)[:d.max_det]
            out.append(torch.cat((box[keep], score[keep, None], lab[keep, None].float()), 1))
        idxs.append(anchor[keep])
    return out, idxs, points, stride


@torch.no_grad()
def decode(preds, metas, nc, reg_max, seg=False, kpt=0, d=DEPLOY, encode_mask=None, obb=False,
           strides=STRIDES):
    """The model's flat output list for a batch -> one `Detections` per image.

    `metas` are the letterbox metas ({"ratio", "pad", "shape"}) of the batch's
    images; everything returned is mapped back to those original frames.
    Masks are computed only inside the rectangle each can be set in
    (`mayaku.model.mask.box_masks`). `encode_mask`, given, turns each chunk
    of them -- `encode_mask(masks, rects, shapes)`, on the device the decode
    runs on (`host_side`) -- into a list of what `masks` holds (the
    evaluator's RLE, `rle_crops`), so full-frame masks never exist.
    """
    preds = host_side(preds)
    grp = split_outputs(preds, seg=seg, kpt=kpt, obb=obb, strides=strides)
    dets, idxs, points, stride = postprocess(preds, nc, reg_max, d, grp.get("ang"), strides=strides)
    # flatten the aux maps once for the whole batch, not per image
    kflat = masklib.flatten_ker(grp["ker"]) if seg else None
    ms = masklib.stride_of(grp["mask"], grp["head"][0]) if seg else None
    kpflat = kptlib.flatten_kpt(grp["kpt"], kpt) if kpt else None
    masks = _decode_masks(grp, dets, idxs, points, stride, kflat, ms, metas, encode_mask) \
        if seg else None
    out = []
    for b, meta in enumerate(metas):
        det = dets[b].float().cpu()
        r = Detections(unletterbox(det[:, :4], meta["ratio"], meta["pad"], meta["shape"]),
                       det[:, 4], det[:, 5].long())
        if obb:
            # the centre and sides map back like a box, unclipped; the angle
            # is untouched (the letterbox scales both axes alike)
            rb = det[:, 6:11].clone()
            rb[:, :2] = (rb[:, :2] - torch.tensor(meta["pad"], dtype=rb.dtype)) / meta["ratio"]
            rb[:, 2:4] /= meta["ratio"]
            r.rboxes = rb
        if seg:
            r.masks = masks[b]
        if kpt:
            if len(det):
                ai, box = idxs[b], dets[b][:, :4].float()     # box stays on the device
                heat, off = grp["heat"][b, :kpt].float(), grp["heat"][b, kpt:].float()
                pxy, pv = kptlib.decode(kpflat[b][ai].float(), points[ai], stride[ai])
                vis = pv.sigmoid()
                pxy = kptlib.snap(pxy, vis, heat, off, box)
                pxy, vis = torch.cat((pxy, vis[..., None]), -1).cpu().split((2, 1), -1)
                xy = unletterbox(pxy.reshape(len(det), -1), meta["ratio"], meta["pad"],
                                 meta["shape"]).reshape(len(det), kpt, 2)
                r.keypoints = torch.cat((xy, vis), -1)
            else:
                r.keypoints = torch.zeros(0, kpt, 3)
        out.append(r)
    return out


def _decode_masks(grp, dets, idxs, points, stride, kflat, ms, metas, encode_mask):
    """Every image's masks, the whole batch at once (`masklib.box_masks`):
    per image a list of `encode_mask`'s encodings, or a (n, h, w) bool
    tensor on the CPU."""
    dev = dets[0].device
    counts = [len(d) for d in dets]
    owner = [b for b, n in enumerate(counts) for _ in range(n)]   # each detection's image
    base = [0, *itertools.accumulate(counts)]
    img = torch.tensor(owner, device=dev, dtype=torch.long)
    ai = torch.cat(idxs)
    if encode_mask:
        flat = [None] * base[-1]
    else:
        out = [torch.zeros(n, *m["shape"], dtype=torch.bool) for n, m in zip(counts, metas, strict=True)]
    chunks = masklib.box_masks(grp["mask"].float(), img, kflat[img, ai].float(), points[ai],
                               stride[ai], torch.cat([d[:, :4] for d in dets]).float(), ms, metas,
                               MASK_BUDGET)
    for idx, m, rects in chunks:
        if encode_mask:
            shapes = [metas[owner[i]]["shape"] for i in idx]
            for i, e in zip(idx, encode_mask(m, rects, shapes), strict=True):
                flat[i] = e
        else:
            for i, mi, (y0, y1, x0, x1) in zip(idx, m.cpu(), rects.tolist(), strict=True):
                out[owner[i]][i - base[owner[i]], y0:y1, x0:x1] = mi[:y1 - y0, :x1 - x0]
    return [flat[base[b]:base[b + 1]] for b in range(len(dets))] if encode_mask else out


def decode_sidecar(preds, metas, sidecar, conf=None, encode_mask=None,
                   mode="eval") -> list[Detections] | list[Classification]:
    """`decode` with every parameter read from a model's sidecar (see
    `mayaku.utils.checkpoint.build_sidecar`): its "decode" block for `mode`
    "eval", its "predict" block (`predict_decode`) for "predict"; `conf`
    replaces the score threshold, `encode_mask` is `decode`'s. A classifier's sidecar (`task: classify`) gives one
    `Classification` per image instead: the softmax of the class map's
    spatial mean."""
    if sidecar.get("task") == "classify":
        from mayaku.model.classify import logits

        return [Classification(p) for p in logits(preds).softmax(1).cpu()]
    dec = sidecar["decode"]
    d = predict_decode(sidecar) if mode == "predict" else Decode.from_sidecar(dec)
    if conf is not None:
        d = dataclasses.replace(d, conf=conf)
    kp = sidecar["keypoints"]
    return decode(preds, metas, len(sidecar["class_names"]), dec["reg_max"],
                  seg=sidecar["mask"] is not None, kpt=kp["num"] if kp else 0, d=d,
                  encode_mask=encode_mask, obb=bool(sidecar.get("obb")),
                  strides=tuple(dec.get("strides", STRIDES)))
