"""COCO metrics for a model on a dataset.

Detections come from `mayaku.inference.decode`, the same host decode a
deployment runs, so evaluation measures exactly what ships. `coco_ap` is
measurement only and deliberately delegates to `pycocotools`, so a
disagreement is never ambiguous between the model and the ruler. AP-S / AP-M
/ AP-L come out of the same call and are always reported alongside AP.
"""

import contextlib
import functools
import io
import types

import numpy as np
import torch
import torch.nn.functional as F

from mayaku.data.batch import batch_to, collate
from mayaku.inference.decode import DEPLOY, decode
from mayaku.model import kpt as kptlib
from mayaku.model import obb as obblib

# The order pycocotools returns its twelve numbers in. A size bucket with no
# ground-truth objects comes back as -1, meaning not applicable, not zero.
STATS = ("AP", "AP50", "AP75", "AP-S", "AP-M", "AP-L",
         "AR1", "AR10", "AR100", "AR-S", "AR-M", "AR-L")
# keypoint eval has no small bucket and returns ten numbers
KPT_STATS = ("AP", "AP50", "AP75", "AP-M", "AP-L",
             "AR", "AR50", "AR75", "AR-M", "AR-L")


# a classifier's scores: top-1 and top-5 accuracy
CLS_STATS = ("top1", "top5")


def to_coco(det, image_id, cat_ids):
    """One image's `Detections` -> COCO bbox result records: xywh, and the
    dense class index mapped back to the annotation file's category id."""
    box = det.boxes.clone()
    box[:, 2:] -= box[:, :2]
    # oriented: each record also carries its xywhr
    rb = det.rboxes.tolist() if det.rboxes is not None else [None] * len(det)
    return [{"image_id": image_id, "category_id": cat_ids[int(c)], "bbox": b, "score": s,
             **({} if r is None else {"rbox": r})}
            for b, s, c, r in zip(box.tolist(), det.scores.tolist(), det.labels.tolist(), rb,
                                  strict=True)]


def rle(mask):
    """An (h, w) bool mask -> COCO run-length encoding, JSON-ready."""
    from pycocotools import mask as maskutil
    r = maskutil.encode(np.asfortranarray(mask.numpy()))
    r["counts"] = r["counts"].decode("ascii")
    return r


def rle_crops(masks, rects, shapes):
    """Masks that are False outside a rectangle each -> COCO RLEs of their
    whole frames: masks (n, H, W) bool, mask k at the top-left of its rect
    k = [y0, y1, x0, x1] in a frame of shapes[k] = (h, w) pixels (a host
    sequence; `mayaku.model.mask.box_masks`'s chunks).

    COCO's runs are column-major. A run boundary falls at pixel p where p
    differs from p - 1 (and at pixel 0 when it is set); outside the
    rectangle every pixel is False, so the boundaries are the crop's own
    column transitions with a False row above and below it, shifted to the
    frame. They are found where the masks live and only the change points
    cross to the host. Where a column ends set at the frame's bottom row and
    the next starts set at the top, that pixel pair is one run: the end of
    one and the start of the other land on the same pixel and cancel."""
    from pycocotools import mask as maskutil

    n = len(masks)
    hw = np.array([h * w for h, w in shapes], np.int64).reshape(n)
    span = int(hw.max()) + 1 if n else 1
    dev = masks.device
    pad = F.pad(masks.to(torch.uint8), (0, 0, 1, 1))
    k, r, c = (pad[:, 1:] != pad[:, :-1]).nonzero(as_tuple=True)
    frame_h = torch.tensor([h for h, _ in shapes], device=dev, dtype=torch.long)
    pos = (rects[k, 2] + c) * frame_h[k] + rects[k, 0] + r
    key = (k * span + pos).sort().values
    twin = key[1:] == key[:-1]
    drop = torch.zeros_like(key, dtype=torch.bool)    # F.pad would fail on an empty `twin`
    drop[1:] |= twin
    drop[:-1] |= twin
    row, pos = np.divmod(key[~drop].cpu().numpy(), span)
    keep = pos < hw[row]                              # a boundary at the frame's end is no boundary
    row, pos = row[keep], pos[keep]
    # every mask's counts in one pass: its boundaries between a 0 and its
    # frame's size, differenced, then sliced out per mask
    cnt = np.bincount(row, minlength=n)
    seq = np.empty(len(pos) + 2 * n, np.int64)
    seq[np.arange(len(pos)) + 2 * row + 1] = pos
    head = np.concatenate(([0], np.cumsum(cnt + 2)[:-1]))
    seq[head], seq[head + cnt + 1] = 0, hw
    diffs = np.diff(seq).tolist()
    out = [None] * n
    by_shape = {}
    for i, hw_ in enumerate(map(tuple, shapes)):
        by_shape.setdefault(hw_, []).append(i)
    for (h, w), ids in by_shape.items():      # one compression call per frame size
        rles = maskutil.frPyObjects([{"size": [h, w], "counts": diffs[head[i]:head[i] + cnt[i] + 1]}
                                     for i in ids], h, w)
        for i, r in zip(ids, rles, strict=True):
            r["counts"] = r["counts"].decode("ascii")
            out[i] = r
    return out


def to_coco_segm(det, image_id, cat_ids):
    """One image's `Detections` with masks -> COCO segm result records (RLE)."""
    return [{"image_id": image_id, "category_id": cat_ids[int(det.labels[k])],
             "segmentation": m if isinstance(m := det.masks[k], dict) else rle(m),
             "score": float(det.scores[k])}
            for k in range(len(det))]


def to_coco_kpt(det, image_id, cat_ids):
    """One image's `Detections` with keypoints -> COCO keypoint result records."""
    return [{"image_id": image_id, "category_id": cat_ids[int(det.labels[k])],
             "keypoints": det.keypoints[k].reshape(-1).tolist(), "score": float(det.scores[k])}
            for k in range(len(det))]


def _gt_rbox(ann):
    """A ground-truth annotation's oriented box, fitted to its polygon (or
    its box where it has none) once, by the same `obb.from_points` the
    training targets use."""
    if "rbox" not in ann:
        from mayaku.data.polygons import Polys

        x, y, w, h = ann["bbox"]
        parts = Polys.from_coco([ann.get("segmentation")]).parts(0)
        ann["rbox"] = obblib.from_points([obblib.outline(parts, (x, y, x + w, y + h))])[0].tolist()
    return ann["rbox"]


@functools.cache
def coco_api():
    """The COCO API, resolved once: ``.COCO`` and ``.COCOeval`` from
    faster-coco-eval when installed -- the same numbers as pycocotools to the
    last digit, its per-image matching and accumulation in C++, several
    times faster on large splits -- else from pycocotools."""
    try:
        from faster_coco_eval import COCO
        from faster_coco_eval import COCOeval_faster as COCOeval
    except ImportError:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    return types.SimpleNamespace(COCO=COCO, COCOeval=COCOeval)


def rotated_eval(gt, dt):
    """COCOeval with the exact IoU of oriented boxes (`obb.rotated_iou`) in
    place of the axis-aligned one; matching, area ranges and the AP / AR
    summaries are COCO's own. Crowd regions are matched as ordinary
    objects."""

    class RotatedEval(coco_api().COCOeval):
        def instances(self, imgId, catId):
            p = self.params
            cats = [catId] if p.useCats else p.catIds
            if hasattr(self, "gt_dataset"):     # faster-coco-eval's C++ index
                return (self.gt_dataset.get_instances([imgId], cats, bool(p.useCats))[0][0],
                        self.dt_dataset.get_instances([imgId], cats, bool(p.useCats))[0][0])
            return ([a for c in cats for a in self._gts[imgId, c]],
                    [a for c in cats for a in self._dts[imgId, c]])

        def computeIoU(self, imgId, catId):
            p = self.params
            g, d = self.instances(imgId, catId)
            if not g or not d:
                return []
            order = np.argsort([-a["score"] for a in d], kind="mergesort")
            d = [d[i] for i in order[:p.maxDets[-1]]]
            return obblib.rotated_iou(torch.tensor([a["rbox"] for a in d]),
                                      torch.tensor([_gt_rbox(a) for a in g])).numpy()

    return RotatedEval(gt, dt, "bbox")


@functools.cache
def ground_truth(ann_path):
    """The parsed annotations, once per path. Large annotation files cost
    seconds to parse and index and the trainer evaluates every epoch; they
    are read-only, so one parse serves the whole run."""
    with contextlib.redirect_stdout(io.StringIO()):
        return coco_api().COCO(ann_path)


@functools.cache
def has_masks(ann_path):
    """Whether any annotation in the file carries a segmentation; box-only
    data (RF100-VL writes ``"segmentation": []``) has no mask AP to score."""
    return any(a.get("segmentation") for a in ground_truth(ann_path).dataset["annotations"])


def coco_ap(ann_path, results, iou_type="bbox", k=0):
    """COCO metrics for a list of result records; `iou_type` is bbox, segm or
    keypoints (with `k` keypoints per instance). Keypoints report ten
    numbers, not twelve. Box records
    that carry an "rbox" are oriented boxes, scored by their exact rotated
    IoU against the ground truth's polygons (`rotated_eval`)."""
    stats = KPT_STATS if iou_type == "keypoints" else STATS
    if not results:
        return dict.fromkeys(stats, 0.0)
    gt = ground_truth(ann_path)
    rotated = iou_type == "bbox" and "rbox" in results[0]
    with contextlib.redirect_stdout(io.StringIO()):
        e = (rotated_eval if rotated else lambda g, d: coco_api().COCOeval(g, d, iou_type))(gt, gt.loadRes(results))
        if iou_type == "keypoints":
            # COCOeval defaults its OKS sigmas to the 17 COCO person
            # keypoints; a model with a different K needs its own, the same
            # table the loss uses
            e.params.kpt_oks_sigmas = kptlib.sigmas(k).numpy()
            # only the categories that carry keypoints are scored: in a
            # multi-class file the others have no keypoint fields at all
            e.params.catIds = [c["id"] for c in gt.dataset["categories"]
                               if len(c.get("keypoints", ())) == k] or e.params.catIds
        e.evaluate()
        e.accumulate()
        e.summarize()
    # faster-coco-eval keeps extra numbers after COCO's own
    return dict(zip(stats, [float(v) for v in e.stats[:len(stats)]], strict=True))


def score(ann_path, detections, cat_ids, seg=False, kpt=0, kpt_ann_path=None):
    """COCO metrics for `(image_id, Detections)` pairs: box AP, plus mask AP
    under "segm_" and keypoint AP under "kpt_" when the model has those heads
    (mask AP only when the ground truth has masks).
    Keypoints are scored against `kpt_ann_path` when the file is separate.
    Oriented boxes (detections with `rboxes`) are scored by their rotated
    IoU."""
    seg = seg and has_masks(ann_path)
    results, segm, kpts = [], [], []
    for image_id, det in detections:
        results += to_coco(det, image_id, cat_ids)
        if seg:
            segm += to_coco_segm(det, image_id, cat_ids)
        if kpt:
            kpts += to_coco_kpt(det, image_id, cat_ids)
    out = coco_ap(ann_path, results)
    if seg:
        out.update({"segm_" + k: v for k, v in coco_ap(ann_path, segm, "segm").items()})
    if kpt:
        out.update({"kpt_" + k: v for k, v in
                    coco_ap(kpt_ann_path or ann_path, kpts, "keypoints", k=kpt).items()})
    return out


def accuracy(probs, labels):
    """(n, nc) scores and (n,) labels -> {"top1", "top5"} accuracy."""
    top = probs.topk(min(5, probs.shape[1]), 1).indices
    hit = top == labels[:, None]
    return {"top1": float(hit[:, 0].float().mean()), "top5": float(hit.any(1).float().mean())}


@torch.no_grad()
def evaluate_classifier(model, dataset, device="cpu", batch=16, workers=0):
    """Top-1 / top-5 accuracy of a classifier on a class-folder split
    (`mayaku.data.folder.FolderDataset`)."""
    from torch.utils.data import DataLoader

    from mayaku.model.classify import logits

    assert model.nc == dataset.nc, (model.nc, dataset.nc)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
                        collate_fn=collate, pin_memory=device.startswith("cuda"))
    was_training = model.training
    model.eval()
    probs, labels = [], []
    try:
        for imgs, targets, _, _ in loader:
            probs.append(logits(model(batch_to(imgs, device))).softmax(1).cpu())
            labels.append(targets[:, 1].long())
    finally:
        model.train(was_training)
    return accuracy(torch.cat(probs), torch.cat(labels))


@torch.no_grad()
def evaluate(model, dataset, device="cpu", batch=16, workers=0, d=DEPLOY):
    """Run a model over a dataset and return the COCO metrics.

    The class count and `reg_max` come from the model, because the model is
    what fixed the head's channel split. The dataset is asserted against it,
    so a head and an annotation file that disagree fail here instead of
    decoding plausible nonsense.
    """
    from torch.utils.data import DataLoader

    if model.task == "classify":
        return evaluate_classifier(model, dataset, device, batch, workers)
    assert model.nc == dataset.nc, (model.nc, dataset.nc)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate,
                        pin_memory=device.startswith("cuda"))
    was_training = model.training
    model.eval()
    seg, kpt, obb = model.cfg.seg, model.cfg.kpt, model.cfg.obb

    def detections():
        for imgs, _, metas, _ in loader:
            preds = model(batch_to(imgs, device))
            # masks RLE-encoded chunk by chunk: never a whole image's at once
            dets = decode(preds, metas, model.nc, model.cfg.reg_max, seg, kpt, d,
                          encode_mask=rle_crops if seg else None, obb=obb)
            yield from ((meta["id"], det) for meta, det in zip(metas, dets, strict=True))

    try:
        return score(dataset.ann_path, detections(), dataset.cat_ids, seg, kpt,
                     dataset.kpt_ann_path)
    finally:
        model.train(was_training)


def evaluate_runner(runner, images, annotations, batch=16, log=None) -> dict[str, float]:
    """COCO metrics for a deployed detector -- a `mayaku.inference.Runner`,
    so a checkpoint and its exported artifacts score through their own
    preprocessing, graph and precision -- on the split `annotations` /
    `images`. `log` receives a progress line every 100 batches."""
    import os

    ann = str(annotations)
    gt = ground_truth(ann)             # the index scoring reads, parsed once
    cat_ids = sorted(gt.getCatIds())   # dense class order, as `load_coco` has it
    if len(runner.class_names) != len(cat_ids):
        raise ValueError(f"the model has {len(runner.class_names)} classes and "
                         f"{annotations} has {len(cat_ids)}")
    ids = list(gt.imgs)
    files = [os.path.join(images, gt.imgs[i]["file_name"]) for i in ids]
    kp = runner.sidecar["keypoints"]

    def detections():
        for i in range(0, len(files), batch):
            if log and i and i % (100 * batch) == 0:
                log("eval %d/%d" % (i, len(files)))
            yield from zip(ids[i:i + batch], runner.batch(files[i:i + batch]), strict=True)

    return score(ann, detections(), cat_ids,
                 seg=runner.sidecar["mask"] is not None, kpt=kp["num"] if kp else 0)


def evaluate_runner_classifier(runner, images, batch=16, log=None) -> dict[str, float]:
    """Top-1 / top-5 accuracy of a deployed classifier (a `Runner` whose
    sidecar says `task: classify`) on the class-folder split `images`,
    through its own preprocessing, graph and precision."""
    from mayaku.data.folder import load_folder

    data = load_folder(str(images), runner.class_names)
    probs = []
    for i in range(0, len(data.files), batch):
        if log and i and i % (100 * batch) == 0:
            log("eval %d/%d" % (i, len(data.files)))
        probs += [c.scores for c in runner.batch(data.files[i:i + batch])]
    return accuracy(torch.stack(probs), torch.from_numpy(data.labels))


def ranked_metric(stats):
    """What a model is trained for, the metric training picks `best.pt` on:
    a classifier's top-1, keypoint AP, mask AP (scored only when the ground
    truth has masks), else box AP."""
    return next(k for k in ("top1", "kpt_AP", "segm_AP", "AP") if k in stats)


def summary(stats, prefix=""):
    """One line, AP first, size split always present, then mask and
    keypoint AP when scored; a classifier's top-1 / top-5."""
    if "top1" in stats:
        return "%stop1 %.4f  top5 %.4f" % (prefix, stats["top1"], stats["top5"])
    return "%sAP %.4f  AP50 %.4f  AP75 %.4f  S %.4f  M %.4f  L %.4f" % (
        prefix, stats["AP"], stats["AP50"], stats["AP75"],
        stats["AP-S"], stats["AP-M"], stats["AP-L"]) + "".join(
        "  %s %.4f" % (k, stats[k]) for k in ("segm_AP", "kpt_AP") if k in stats)
