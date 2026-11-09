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

import torch

from mayaku.backends.ops.nms import batched_nms
from mayaku.data.geometry import unletterbox
from mayaku.model import kpt as kptlib
from mayaku.model import mask as masklib
from mayaku.model import obb as obblib
from mayaku.model.blocks import STRIDES
from mayaku.model.box import anchor_grid, decode_boxes, flatten_head
from mayaku.model.detector import split_outputs

__all__ = ["DEPLOY", "Classification", "Decode", "Detections", "decode", "decode_sidecar",
           "postprocess"]


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
    def from_sidecar(cls, block, **override):
        """From a sidecar's "decode" block (which also carries strides and
        reg_max); `override` replaces fields, e.g. conf."""
        return cls(**{f.name: block[f.name] for f in dataclasses.fields(cls)} | override)


DEPLOY = Decode()

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


def postprocess(preds, nc, reg_max, d=DEPLOY, ang=None):
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
    cls, dist, shapes = flatten_head(preds[:2 * len(STRIDES)], nc, reg_max, contiguous=ang is not None)
    points, stride = anchor_grid(shapes, STRIDES, device=cls.device, dtype=cls.dtype)
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
        if len(score) > d.topk:
            keep = score.topk(d.topk).indices
            score, lab, anchor = score[keep], lab[keep], anchor[keep]
        if rboxes is not None:
            box = rboxes[b][anchor]
            keep = obblib.nms(box, score, lab, d.iou)[:d.max_det]
            out.append(torch.cat((obblib.to_xyxy(box[keep]), score[keep, None],
                                  lab[keep, None].float(), box[keep]), 1))
        else:
            box = decode_boxes(dist[b, anchor], points[anchor], stride[anchor], reg_max)
            keep = batched_nms(box, score, lab, d.iou)[:d.max_det]
            out.append(torch.cat((box[keep], score[keep, None], lab[keep, None].float()), 1))
        idxs.append(anchor[keep])
    return out, idxs, points, stride


@torch.no_grad()
def decode(preds, metas, nc, reg_max, seg=False, kpt=0, d=DEPLOY, encode_mask=None, obb=False):
    """The model's flat output list for a batch -> one `Detections` per image.

    `metas` are the letterbox metas ({"ratio", "pad", "shape"}) of the batch's
    images; everything returned is mapped back to those original frames.
    Masks are computed only inside the rectangle each can be set in
    (`mayaku.model.mask.box_masks`). `encode_mask`, given, turns each chunk
    of them -- `encode_mask(masks, rects, shapes)`, still on the device --
    into a list of what `masks` holds (the evaluator's RLE, `rle_crops`), so
    full-frame masks never exist or cross to the host.
    """
    grp = split_outputs(preds, seg=seg, kpt=kpt, obb=obb)
    dets, idxs, points, stride = postprocess(preds, nc, reg_max, d, grp.get("ang"))
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


def decode_sidecar(preds, metas, sidecar, conf=None,
                   encode_mask=None) -> list[Detections] | list[Classification]:
    """`decode` with every parameter read from a model's sidecar (see
    `mayaku.utils.checkpoint.build_sidecar`); `conf` overrides the recorded
    score threshold, `encode_mask` is `decode`'s. A classifier's sidecar (`task: classify`) gives one
    `Classification` per image instead: the softmax of the class map's
    spatial mean."""
    if sidecar.get("task") == "classify":
        from mayaku.model.classify import logits

        return [Classification(p) for p in logits(preds).softmax(1).cpu()]
    dec = sidecar["decode"]
    d = Decode.from_sidecar(dec, **({} if conf is None else {"conf": conf}))
    kp = sidecar["keypoints"]
    return decode(preds, metas, len(sidecar["class_names"]), dec["reg_max"],
                  seg=sidecar["mask"] is not None, kpt=kp["num"] if kp else 0, d=d,
                  encode_mask=encode_mask, obb=bool(sidecar.get("obb")))
