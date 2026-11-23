"""The per-instance mask head, owned once.

The graph emits an 8-channel mask feature at stride s (8, or 4 with
`Tier.mask_stride`; `stride_of` reads it off the tensors) and, per anchor, 169
numbers that are the weights and biases of a three-layer 1x1 network
(8+2 -> 8 -> 8 -> 1, ReLU between). The host prepends two relative-coordinate
channels and runs that network over the detection's box. Nothing here is in
the deploy graph; this file is what the loss, the evaluator and any host
decoder must agree on.

Coordinates: cell (i, j) of the stride-s map covers image pixels
[sj, sj+s) x [si, si+s) and its centre is at ((j + .5) s, (i + .5) s), the
same convention as `mayaku.model.box.anchor_grid`.
"""

import functools
import itertools
import math

import torch
import torch.nn.functional as F

from mayaku.model.aux import KERNEL_LAYOUT, KERNEL_PARAMS, MASK_CH
from mayaku.model.blocks import STRIDES
from mayaku.model.box import anchor_grid, flatten_levels

# Relative coordinates are scaled by this many strides of the anchor's level,
# so a stride-8 anchor sees +-1 at 64 px and a stride-32 anchor at 256 px: the
# size range each level owns (CondInst's size of interest).
COORD_SCALE = 8


def flatten_ker(kers):
    """[ker0, ker1, ...] (B, 169, h, w) -> (B, N, 169), N fine to coarse."""
    return flatten_levels(kers, KERNEL_PARAMS)


def split_kernels(k):
    """(P, 169) -> [(W1 (P,8,10), b1 (P,8)), (W2 (P,8,8), b2), (W3 (P,1,8), b3)]."""
    assert k.shape[-1] == KERNEL_PARAMS, k.shape
    out, o = [], 0
    for cin, cout in KERNEL_LAYOUT:
        w = k[:, o:o + cin * cout].reshape(-1, cout, cin)
        o += cin * cout
        b = k[:, o:o + cout]
        o += cout
        out.append((w, b))
    return out


def stride_of(mask_feat, head0):
    """The mask feature's stride, from its size and the finest head map's."""
    return head0.shape[-2] * STRIDES[0] // mask_feat.shape[-2]


def rel_coords(points, stride, h, w, mask_stride, dtype=None, cells=None):
    """(P, 2) anchor points, (P, 1) strides -> (P, 2, h*w) relative coordinates
    of every cell centre of the stride-`mask_stride` map, in units of
    COORD_SCALE * stride; with `cells` (P, Q) flat cell indices, of those
    cells only -> (P, 2, Q)."""
    dtype = dtype or points.dtype
    grid = anchor_grid([(h, w)], [mask_stride], device=points.device, dtype=dtype)[0].T  # (2, h*w)
    grid = grid[None] if cells is None else grid[:, cells].transpose(0, 1)
    scale = (COORD_SCALE * stride).to(dtype)                       # (P, 1)
    return (grid - points.to(dtype)[:, :, None]) / scale[:, :, None]


def dyn_conv(feat, coords, kernels):
    """Apply each instance's kernels to its feature.

    feat (P, 8, Q), coords (P, 2, Q), kernels (P, 169) -> logits (P, Q). Every
    layer is 1x1, so Q can be a full map or any subset of points; the
    arithmetic is the same.
    """
    x = torch.cat((coords.to(feat.dtype), feat), 1)                # (P, 10, Q)
    layers = split_kernels(kernels.to(feat.dtype))
    for i, (w, b) in enumerate(layers):
        x = torch.baddbmm(b[:, :, None], w, x)                     # (P, cout, Q)
        if i + 1 < len(layers):
            x = F.relu(x)
    return x[:, 0]


def dice(logits, target, eps=5e-6):
    """Per-instance Dice loss on logits against hard or soft targets, in
    fp32; zero exactly when sigmoid(logits) equals the target. (P, Q) -> (P,)."""
    p = logits.float().sigmoid()
    t = target.float()
    inter = (p * t).sum(1)
    return 1 - (2 * inter + eps) / ((p * p).sum(1) + (t * t).sum(1) + eps)


def coverage(inst, img, ids, stride, chunk=64):
    """Each instance's exact share of every mask cell: its soft target.

    inst (B, H, W) uint8 full-resolution instance map (row + 1, from
    `mayaku.data.polygons.Polys.paint`), img (P,) image index, ids (P,) the
    instance's value in that image -> (P, H/s * W/s) float in [0, 1]. Two
    instances sharing a cell each get their own fraction. Chunked: a chunk
    holds `chunk` full-resolution boolean maps.
    """
    _, h, w = inst.shape
    assert h % stride == 0 and w % stride == 0, (inst.shape, stride)
    out = [F.avg_pool2d((inst[img[c:c + chunk]] == ids[c:c + chunk, None, None].to(inst.dtype))
                        [:, None].float(), stride).flatten(1)
           for c in range(0, len(img), chunk)]
    return torch.cat(out) if out else inst.new_zeros(0, (h // stride) * (w // stride), dtype=torch.float)


def crop_cells(boxes, h, w, mask_stride):
    """xyxy pixel boxes -> inclusive-exclusive cell rectangles on the
    stride-`mask_stride` map, as (x0, y0, x1, y1) long tensors clamped to it."""
    s = mask_stride
    x0 = (boxes[:, 0] / s).floor().long().clamp(0, w - 1)
    y0 = (boxes[:, 1] / s).floor().long().clamp(0, h - 1)
    x1 = torch.maximum((boxes[:, 2] / s).ceil().long(), x0 + 1).clamp(max=w)
    y1 = torch.maximum((boxes[:, 3] / s).ceil().long(), y0 + 1).clamp(max=h)
    return x0, y0, x1, y1


@torch.no_grad()
def assemble(mask_feat, kernels, points, stride, boxes, mask_stride):
    """Full-map logits per detection, -inf outside the detection's box
    (cropped at cell granularity), so a threshold at 0 never fires there.
    The definition `box_masks` reproduces without full maps (with
    `to_canvas` and `unletterbox_maps`); decoding uses `box_masks`.

    mask_feat (8, Hs, Ws), kernels (n, 169), points (n, 2), stride (n, 1),
    boxes (n, 4) xyxy letterbox pixels -> logits (n, Hs, Ws).
    """
    n = len(kernels)
    _, h, w = mask_feat.shape
    if n == 0:
        return mask_feat.new_zeros(0, h, w)
    coords = rel_coords(points, stride, h, w, mask_stride, dtype=mask_feat.dtype)
    logits = dyn_conv(mask_feat[None].expand(n, -1, -1, -1).reshape(n, MASK_CH, -1),
                      coords, kernels).view(n, h, w)
    x0, y0, x1, y1 = crop_cells(boxes, h, w, mask_stride)
    ys = torch.arange(h, device=logits.device)[None, :, None]
    xs = torch.arange(w, device=logits.device)[None, None, :]
    inside = ((ys >= y0[:, None, None]) & (ys < y1[:, None, None])
              & (xs >= x0[:, None, None]) & (xs < x1[:, None, None]))
    return logits.masked_fill(~inside, float("-inf"))


def to_canvas(logits, mask_stride):
    """(n, Hs, Ws) crop logits -> (n, 1, H, W) logits over the whole canvas,
    upsampled bilinearly, so the boundary is interpolated rather than
    blocky. Undo the letterbox with `mayaku.data.geometry.unletterbox_maps`
    and threshold at 0 for masks in the original image."""
    canvas = (logits.shape[-2] * mask_stride, logits.shape[-1] * mask_stride)
    return F.interpolate(logits[:, None].float(), size=canvas, mode="bilinear", align_corners=False)


@functools.lru_cache(maxsize=64)
def _axis(cells, mask_stride, offset, fit, out, device):
    """One axis of `to_canvas` then `unletterbox_maps` -- cells to canvas
    (x `mask_stride`), the letterbox cut [offset, offset + fit), canvas to
    the `out` original pixels -- run over an identity and over -inf masks,
    so everything is those interpolations' own:

    weights (out, cells): each output pixel's weight on each cell;
    lo, hi (out,): the first and last cell an output pixel reads, zero-weight
    reads included. A pixel is set in the full-map path only when every cell
    it reads is inside the box (outside, -inf turns it -inf or NaN), so
    these give the rectangle a box's mask can occupy."""
    def interp(x):  # x (cells, n): each column resampled alone -> (out, n)
        x = F.interpolate(x.T[:, None, :, None], size=(cells * mask_stride, 1), mode="bilinear",
                          align_corners=False)
        return F.interpolate(x[:, :, offset:offset + fit], size=(out, 1), mode="bilinear",
                             align_corners=False)[:, 0, :, 0].T
    ar = torch.arange(cells, device=device)
    weights = interp(torch.eye(cells, device=device))
    inf = torch.tensor(float("-inf"), device=device)
    # column c of `below` is -inf on cells < c, of `above` on cells >= c + 1
    below = torch.where(ar[:, None] < ar[None, :], inf, 0.0)
    above = torch.where(ar[:, None] > ar[None, :], inf, 0.0)
    lo = interp(below).isfinite().sum(1) - 1
    hi = cells - interp(above).isfinite().sum(1)
    return weights, lo, hi


class _Axis:
    """One axis (rows or columns) of a batch's box masks: the images' `_axis`
    tables padded to the largest (past an image's edge `lo` is -1, which
    fails every box), the boxes' cell ranges [c0, c1), and the pixel ranges
    [r0, r1) their masks can occupy."""

    def __init__(self, metas, dim, cells, mask_stride, img, c0, c1):
        from mayaku.data.geometry import fitted

        dev, nb = c0.device, len(metas)
        n = max(m["shape"][dim] for m in metas)
        self.cells, self.n, self.img, self.c0, self.c1 = cells, n, img, c0, c1
        self.weights = torch.zeros(nb, n, cells, device=dev)
        lo = torch.full((nb, n), -1, device=dev)
        hi = torch.zeros(nb, n, dtype=torch.long, device=dev)
        for b, m in enumerate(metas):
            size, offset = m["shape"][dim], m["pad"][1 - dim]     # pad is (left, top)
            fit = fitted(m["shape"], m["ratio"])[dim]
            self.weights[b, :size], lo[b, :size], hi[b, :size] = _axis(cells, mask_stride, offset,
                                                                       fit, size, dev)
        ok = (lo[img] >= c0[:, None]) & (hi[img] < c1[:, None])
        self.r0 = ok.to(torch.uint8).argmax(1)
        self.r1 = self.r0 + ok.sum(1)

    def chunk(self, k, n_cells, n_px):
        """For detections `k` padded to `n_cells` cells and `n_px` pixels:
        their cell indices (g, n_cells), clamped to the map, with which are
        inside the box; which pixels are inside the rectangle (g, n_px); and
        the weights (g, n_px, n_cells), zero outside both."""
        c = self.c0[k, None] + torch.arange(n_cells, device=k.device)
        inc = c < self.c1[k, None]
        c = c.clamp(max=self.cells - 1)
        r = self.r0[k, None] + torch.arange(n_px, device=k.device)
        okr = r < self.r1[k, None]
        w = self.weights[self.img[k, None, None], r.clamp(max=self.n - 1)[:, :, None], c[:, None, :]]
        return c, inc, okr, w * (okr[:, :, None] & inc[:, None, :])


def _chunks(rh, rw, budget):
    """Host lists of detection indices with similar rectangles: grouped by
    height and width band (powers of 4), each group split so its count
    times its padded area stays within `budget`. A chunk pads to its
    largest, and few chunks mean few kernel launches."""
    def band(i):
        return math.ceil(math.log2(rh[i]) / 2), math.ceil(math.log2(rw[i]) / 2)

    for _, grp in itertools.groupby(sorted(range(len(rh)), key=band), key=band):
        chunk, mh, mw = [], 0, 0
        for i in grp:
            nh, nw = max(mh, rh[i]), max(mw, rw[i])
            if chunk and (len(chunk) + 1) * nh * nw > budget:
                yield chunk
                chunk, nh, nw = [], rh[i], rw[i]
            chunk.append(i)
            mh, mw = nh, nw
        yield chunk


@torch.no_grad()
def box_masks(mask_feat, img, kernels, points, stride, boxes, mask_stride, metas, budget):
    """Each detection's mask in its original frame, computed only inside the
    rectangle it can be set in: what thresholding `unletterbox_maps(
    to_canvas(assemble(...)))` at 0 gives, without the full-frame maps.

    The network runs on the box's cells only, both resizes are folded into
    two small weight matrices per image size, and the rectangle is where the
    full path's -inf outside the box reaches no pixel (`_axis`). A whole
    batch runs together, in chunks of similar rectangles (`_chunks`), so the
    work is a few large kernels rather than many per detection.

    mask_feat (B, 8, Hs, Ws); img (n,) each detection's image; kernels
    (n, 169), points (n, 2), stride (n, 1), boxes (n, 4) xyxy letterbox
    pixels; `metas` the images' letterbox metas ({"ratio", "pad", "shape"}).
    Yields chunks of at most `budget` mask elements as (idx, a host list of
    indices into the n detections, masks (g, H, W) bool, rects (g, 4) long
    [y0, y1, x0, x1] in original pixels): mask k covers rect k from its
    top-left corner and is False elsewhere.
    """
    _, _, hs, ws = mask_feat.shape
    x0, y0, x1, y1 = crop_cells(boxes, hs, ws, mask_stride)
    ys = _Axis(metas, 0, hs, mask_stride, img, y0, y1)
    xs = _Axis(metas, 1, ws, mask_stride, img, x0, x1)
    rects = torch.stack((ys.r0, ys.r1, xs.r0, xs.r1), 1)
    feat = mask_feat.flatten(2).transpose(1, 2)                   # (B, Hs*Ws, 8)
    # every size crosses to the host once
    rh, rw, ch, cw = torch.stack(((ys.r1 - ys.r0).clamp(min=1), (xs.r1 - xs.r0).clamp(min=1),
                                  y1 - y0, x1 - x0)).tolist()
    for idx in _chunks(rh, rw, budget):
        k = torch.tensor(idx, device=mask_feat.device)
        cy, iny, oky, a = ys.chunk(k, max(ch[i] for i in idx), max(rh[i] for i in idx))
        cx, inx, okx, b = xs.chunk(k, max(cw[i] for i in idx), max(rw[i] for i in idx))
        cell = (cy[:, :, None] * ws + cx[:, None, :]).flatten(1)   # (g, cells)
        coords = rel_coords(points[k], stride[k], hs, ws, mask_stride, mask_feat.dtype, cell)
        lg = dyn_conv(feat[img[k, None], cell].transpose(1, 2), coords, kernels[k])
        lg = lg.view(len(idx), cy.shape[1], cx.shape[1]).masked_fill(~(iny[:, :, None] & inx[:, None, :]), 0)
        m = (a @ lg @ b.transpose(1, 2) > 0) & oky[:, :, None] & okx[:, None, :]
        yield idx, m, rects[k]
