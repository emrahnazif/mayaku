"""Oriented boxes: the angle's encoding, ProbIoU, exact rotated IoU and
rotated NMS, owned once for the loss, the assigner, the host decode and the
evaluator.

An oriented box is (cx, cy, w, h, theta) in pixels and radians, "xywhr",
theta turning the box's x axis from the image's towards its y axis (down).
Per anchor the graph emits the usual four DFL side distances, measured along
the box's own axes, and one angle logit on the box tower:
theta = (sigmoid(logit) - 1/4) * pi, in [-pi/4, 3pi/4). A box is the same box
under theta + pi, and under (h, w, theta + pi/2); ground truth is put in one
canonical form, w >= h and theta wrapped into the readout's range, so its side
distances are unique. ProbIoU is blind to these equivalences, so the overlap
terms never fight the canonical form.

ProbIoU (Llerena et al. 2021, arXiv 2106.06072) models each box as a 2-D
Gaussian with the box's covariance (w^2/12, h^2/12, rotated) and turns the
Bhattacharyya distance into an overlap: 1 - sqrt(1 - exp(-B)). It is smooth,
closed-form and differentiable everywhere, so it ranks candidates in the
assigner, trains the box and runs the host NMS; the evaluator scores with the
exact polygon IoU instead.
"""

import math

import numpy as np
import torch

from mayaku.model.blocks import STRIDES

ANGLE_LO = -math.pi / 4      # the readout's range is [ANGLE_LO, ANGLE_LO + pi)


def angle(logits):
    """Angle logits -> theta in radians, in [-pi/4, 3pi/4)."""
    return (logits.sigmoid() - 0.25) * math.pi


def _local(px, py, r):
    """Points (px, py) in the frame of boxes r: (u, v) from the centre along
    the box's own x and y axes."""
    dx, dy = px - r[..., 0], py - r[..., 1]
    c, s = torch.cos(r[..., 4]), torch.sin(r[..., 4])
    return dx * c + dy * s, dy * c - dx * s


def decode(dist, points, theta):
    """ltrb distances in pixels along the box's axes, anchor points (.., 2)
    and theta -> xywhr. The centre sits ((r - l) / 2, (b - t) / 2) from the
    point in the box's frame."""
    lf, t, r, b = dist.unbind(-1)
    ox, oy = (r - lf) / 2, (b - t) / 2
    c, s = torch.cos(theta), torch.sin(theta)
    cx = points[..., 0] + ox * c - oy * s
    cy = points[..., 1] + ox * s + oy * c
    return torch.stack((cx, cy, lf + r, t + b, theta), -1)


def decode_head(ang, dist, points, stride, reg_max):
    """The angle maps and the flat DFL bin logits (`box.decode_head`) ->
    (B, N, 5) xywhr in pixels: the one decode of a prediction, for the loss,
    the assigner and the host alike."""
    from mayaku.model.box import dfl_expectation, flatten_levels

    theta = angle(flatten_levels(ang, 1)[..., 0].float())
    return decode(dfl_expectation(dist.float(), reg_max) * stride, points, theta)


def encode(rbox, points, stride, reg_max):
    """xywhr -> ltrb in stride units along the box's axes, clamped into the
    representable range like `box.xyxy_to_ltrb`."""
    u, v = _local(points[..., 0], points[..., 1], rbox)
    hw, hh = rbox[..., 2] / 2, rbox[..., 3] / 2
    ltrb = torch.stack((hw + u, hh + v, hw - u, hh - v), -1) / stride
    return ltrb.clamp_(0, reg_max - 1 - 0.01)


def to_xyxy(r):
    """xywhr -> the axis-aligned hull, xyxy."""
    c, s = torch.cos(r[..., 4]).abs(), torch.sin(r[..., 4]).abs()
    ex = r[..., 2] / 2 * c + r[..., 3] / 2 * s
    ey = r[..., 2] / 2 * s + r[..., 3] / 2 * c
    return torch.stack((r[..., 0] - ex, r[..., 1] - ey, r[..., 0] + ex, r[..., 1] + ey), -1)


def corners(r):
    """xywhr (.., 5) -> the four corners (.., 4, 2), in order around the box."""
    c, s = torch.cos(r[..., 4]), torch.sin(r[..., 4])
    hw, hh = r[..., 2] / 2, r[..., 3] / 2
    u = torch.stack((-hw, hw, hw, -hw), -1)
    v = torch.stack((-hh, -hh, hh, hh), -1)
    x = r[..., 0, None] + u * c[..., None] - v * s[..., None]
    y = r[..., 1, None] + u * s[..., None] + v * c[..., None]
    return torch.stack((x, y), -1)


def _cov(r):
    a, b = r[..., 2].pow(2) / 12, r[..., 3].pow(2) / 12
    c, s = torch.cos(r[..., 4]), torch.sin(r[..., 4])
    return a * c * c + b * s * s, a * s * s + b * c * c, (a - b) * c * s


def probiou(a, b, eps=1e-7):
    """ProbIoU between broadcastable xywhr tensors, in [0, 1]."""
    a11, a22, a12 = _cov(a)
    b11, b22, b12 = _cov(b)
    s11, s22, s12 = (a11 + b11) / 2, (a22 + b22) / 2, (a12 + b12) / 2
    det = (s11 * s22 - s12 * s12).clamp(min=eps)
    dx, dy = a[..., 0] - b[..., 0], a[..., 1] - b[..., 1]
    maha = (s22 * dx * dx + s11 * dy * dy - 2 * s12 * dx * dy) / det
    det_a = (a11 * a22 - a12 * a12).clamp(min=eps)
    det_b = (b11 * b22 - b12 * b12).clamp(min=eps)
    bd = maha / 8 + 0.5 * torch.log(det / (det_a * det_b).sqrt() + eps)
    hd = (1 - torch.exp(-bd.clamp(eps, 100)) + eps).sqrt()
    return 1 - hd


def inside(points, rbox, eps=1e-9, finest=STRIDES[0], floor=STRIDES[1]):
    """(B, M, N) bool: is this anchor centre inside this oriented box? The
    oriented form of `assign.candidates_in_gt`, with the same widening of a
    side under the finest stride to the middle stride."""
    r = rbox[:, :, None, :]
    u, v = _local(points[None, None, :, 0], points[None, None, :, 1], r)
    w, h = r[..., 2], r[..., 3]
    w = torch.where(w < finest, torch.full_like(w, float(floor)), w)
    h = torch.where(h < finest, torch.full_like(h, float(floor)), h)
    return (w / 2 - u.abs() > eps) & (h / 2 - v.abs() > eps)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def rotated_iou(a, b, chunk=1 << 16):
    """Exact IoU between every xywhr box of `a` (N, 5) and of `b` (M, 5) ->
    (N, M). The intersection of two convex quadrilaterals is the convex
    polygon on the corners of each inside the other and the crossings of
    their edges; those (at most 24) candidates are ordered by angle around
    their centroid and measured with the shoelace formula, all as batched
    tensor ops."""
    n, m = len(a), len(b)
    out = torch.zeros(n, m, dtype=torch.float32)
    if n == 0 or m == 0:
        return out
    a, b = a.float(), b.float()
    ia, ib = torch.meshgrid(torch.arange(n), torch.arange(m), indexing="ij")
    ia, ib = ia.reshape(-1), ib.reshape(-1)
    area_a, area_b = a[:, 2] * a[:, 3], b[:, 2] * b[:, 3]
    flat = out.view(-1)
    for s in range(0, len(ia), chunk):
        pa, pb = a[ia[s:s + chunk]], b[ib[s:s + chunk]]
        ca, cb = corners(pa), corners(pb)                              # (P, 4, 2)
        # corners of each box inside the other (a small tolerance keeps shared corners)
        ua, va = _local(ca[..., 0], ca[..., 1], pb[:, None])
        in_a = (ua.abs() <= pb[:, None, 2] / 2 + 1e-4) & (va.abs() <= pb[:, None, 3] / 2 + 1e-4)
        ub, vb = _local(cb[..., 0], cb[..., 1], pa[:, None])
        in_b = (ub.abs() <= pa[:, None, 2] / 2 + 1e-4) & (vb.abs() <= pa[:, None, 3] / 2 + 1e-4)
        # edge crossings: a's edge i (p, p + r) against b's edge j (q, q + d)
        p, r = ca[:, :, None], (ca.roll(-1, 1) - ca)[:, :, None]       # (P, 4, 1, 2)
        q, d = cb[:, None], (cb.roll(-1, 1) - cb)[:, None]             # (P, 1, 4, 2)
        den = _cross(r, d)
        ok = den.abs() > 1e-9
        den = torch.where(ok, den, torch.ones_like(den))
        t = _cross(q - p, d) / den
        u = _cross(q - p, r) / den
        hit = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        cross = (p + t[..., None] * r).reshape(-1, 16, 2)
        pts = torch.cat((ca, cb, cross), 1)                            # (P, 24, 2)
        valid = torch.cat((in_a, in_b, hit.reshape(-1, 16)), 1)
        cnt = valid.sum(1)
        centre = (pts * valid[..., None]).sum(1) / cnt.clamp(min=1)[:, None]
        ang = torch.atan2(pts[..., 1] - centre[:, None, 1], pts[..., 0] - centre[:, None, 0])
        order = torch.where(valid, ang, torch.full_like(ang, 10.0)).argsort(1)
        pts = pts.gather(1, order[..., None].expand(-1, -1, 2))
        # invalid slots, sorted last, repeat the first point: zero-area edges
        live = torch.arange(24)[None] < cnt[:, None]
        pts = torch.where(live[..., None], pts, pts[:, :1])
        inter = 0.5 * _cross(pts, pts.roll(-1, 1)).sum(1).abs()
        inter = torch.where(cnt >= 3, inter, torch.zeros_like(inter))
        union = area_a[ia[s:s + chunk]] + area_b[ib[s:s + chunk]] - inter
        flat[s:s + chunk] = inter / union.clamp(min=1e-9)
    return out


def nms(rbox, scores, labels, iou):
    """Greedy per-class NMS on ProbIoU: indices of the kept boxes, by
    descending score. Classes are kept apart by moving each class's centres
    far from every other's, the offset trick `batched_nms` uses. Runs on
    the device as Cluster-NMS (Zheng et al. 2020, arXiv 2005.03572): a box
    is kept when no kept higher-scoring box overlaps it, iterated to its
    fixed point, which is exactly greedy NMS, in a few passes."""
    if len(scores) == 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)
    order = scores.argsort(descending=True)
    r = rbox[order].float().clone()
    big = float(r[:, :4].abs().max()) * 4 + 4096
    r[:, :2] += labels[order].float()[:, None] * big
    over = (probiou(r[:, None], r[None]) > iou).triu(1)
    keep = torch.ones(len(r), dtype=torch.bool, device=r.device)
    while True:
        new = ~(over & keep[:, None]).any(0)
        if torch.equal(new, keep):
            return order[keep]
        keep = new


def from_points(objects):
    """A list of (n, 2) point arrays, one per object -> (len, 5) float32
    xywhr, each object's minimum-area oriented box (OpenCV's
    `minAreaRect`) in canonical form. NumPy, for the data pipeline."""
    import cv2

    out = np.zeros((len(objects), 5), np.float32)
    for i, xy in enumerate(objects):
        (cx, cy), (w, h), deg = cv2.minAreaRect(np.asarray(xy, np.float32))
        t = math.radians(deg)
        if w < h:
            w, h, t = h, w, t + math.pi / 2
        out[i] = cx, cy, w, h, (t - ANGLE_LO) % math.pi + ANGLE_LO
    return out


def outline(parts, xyxy):
    """An object's points for `from_points`: its polygon parts, or the four
    corners of its axis-aligned (x1, y1, x2, y2) box when it has none."""
    if len(parts):
        return np.concatenate(parts)
    x1, y1, x2, y2 = xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
