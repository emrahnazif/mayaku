"""Aspect-aware input sizing under a compute budget.

The canvas is the one fixed (H, W) every image is letterboxed onto, for
training, evaluation, export and deployment alike. ``size_budget`` is the
budget dial -- the square-equivalent side, so the compute budget is
``size_budget ** 2`` pixels -- and :func:`snap_max_content` resolves the canvas
that holds the most real image content for the data's aspect while staying
under that budget.

Why max-content-under-budget: it is a strict Pareto win over a square
letterbox -- equal-or-more real resolution at equal-or-less compute on every
aspect -- and the never-exceed ceiling gives a hard compute / memory bound.

Every canvas is aligned to the detector's coarsest stride (``Tier.align``:
:data:`CANVAS_ALIGN`, 32, or 64 for a tier with a stride-64 level): both
sides must divide by it for that level's grid and for the exact 2x upsamples
between levels. Nothing coarser is needed, and a coarser grid wastes budget:
at 800² a 128 grid leaves a 16:9 canvas holding 73% real content where the
32 grid holds 92%.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from mayaku.model.blocks import CANVAS_ALIGN

#: Image (height, width) pairs: a sequence of pairs or an (n, 2) array.
Shapes = Sequence[tuple[int, int]] | npt.NDArray[np.integer[Any]]

__all__ = [
    "ASPECT_UNIFORMITY_THRESHOLD",
    "CANVAS_ALIGN",
    "aspect_spread",
    "canvas_for_data",
    "data_aspect",
    "multi_scale_canvases",
    "render_canvas",
    "snap_max_content",
]

#: Robust aspect spread (p90 / p10) at or below this: the dataset is "one
#: aspect", and a canvas at that aspect beats a square letterbox.
ASPECT_UNIFORMITY_THRESHOLD = 1.10


def aspect_spread(aspects: Sequence[float]) -> float:
    """Robust aspect spread ``p90 / p10`` (1.0 for fewer than 10 samples)."""
    n = len(aspects)
    if n < 10:
        return 1.0
    s = sorted(aspects)
    return s[(n * 9) // 10] / max(s[n // 10], 1e-9)


def data_aspect(shapes: Shapes) -> tuple[float, bool]:
    """Median image aspect ``W / H`` and whether the data is one aspect, from
    (height, width) image shapes. Uniform means the robust spread is within
    :data:`ASPECT_UNIFORMITY_THRESHOLD`, so a few outliers never flip it."""
    aspects = [w / h for h, w in shapes]
    if not aspects:
        return 1.0, False
    return statistics.median(aspects), aspect_spread(aspects) <= ASPECT_UNIFORMITY_THRESHOLD


def snap_max_content(budget: int, aspect: float, align: int = CANVAS_ALIGN) -> tuple[int, int]:
    """The aligned ``(H, W)`` canvas that maximises letterbox content under a
    budget.

    Args:
        budget: Max canvas area in pixels (``size_budget ** 2``). The result
            never exceeds it (``H * W <= budget``).
        aspect: Data aspect ``W / H`` (>1 landscape, <1 portrait, 1 square).
        align: What both sides divide by, the model's coarsest stride.

    Returns:
        ``(H, W)`` maximising real content ``min(W**2 / aspect, aspect * H**2)``
        -- the binding dimension's content after an aspect-preserving
        letterbox. For a square aspect this is the largest aligned square.
    """
    if budget <= 0:
        raise ValueError(f"budget must be > 0; got {budget}")
    if aspect <= 0:
        raise ValueError(f"aspect must be > 0; got {aspect}")
    # A side never needs to exceed the long edge of the most extreme fit,
    # sqrt(budget * max(a, 1/a)); round up to the grid for the search bound.
    reach = math.isqrt(int(budget * max(aspect, 1.0 / aspect)))
    sides = range(align, ((reach // align) + 1) * align + 1, align)

    best_content = -1.0
    best_hw = (align, align)
    for w in sides:
        for h in sides:
            if w * h > budget:
                continue
            content = min(w * w / aspect, aspect * h * h)
            if content > best_content:
                best_content = content
                best_hw = (h, w)
    return best_hw


def canvas_for_data(shapes: Shapes, size_budget: int,
                    align: int = CANVAS_ALIGN) -> tuple[int, int]:
    """The canvas for a dataset of (height, width) image shapes under
    ``size_budget ** 2`` pixels: a rectangle at the data's aspect when it has
    one (no padding waste), else a square, which is robust to any shape.
    Sides are multiples of ``align``, so a stride-64 model under budget 768
    trains at 768 x 768."""
    aspect, uniform = data_aspect(shapes)
    return snap_max_content(size_budget * size_budget, aspect if uniform else 1.0, align)


def render_canvas(deploy_canvas: tuple[int, int], max_budget: int,
                  align: int = CANVAS_ALIGN) -> tuple[int, int]:
    """The canvas training batches render at: the canvas ``max_budget`` gives
    at the deploy aspect when that is larger than ``deploy_canvas``, else the
    deploy canvas itself (``max_budget`` 0 or below it)."""
    h, w = deploy_canvas
    if max_budget <= 0:
        return deploy_canvas
    big = snap_max_content(max_budget * max_budget, w / h, align)
    return big if big[0] * big[1] > h * w else deploy_canvas


def multi_scale_canvases(deploy_canvas: tuple[int, int], min_budget: int,
                         align: int = CANVAS_ALIGN,
                         max_budget: int = 0) -> list[tuple[int, int]]:
    """Training canvases for multi-scale, anchored on the deploy canvas.

    ``min_budget`` is a size budget like ``size_budget`` (0 starts at
    ``deploy_canvas``), and ``max_budget`` optionally lifts the top rung above
    ``deploy_canvas`` (:func:`render_canvas`). Each rung is the canvas
    :func:`snap_max_content` resolves at the deploy aspect for a budget
    stepping by ``align``. Both ``deploy_canvas`` and the top rendered rung are
    always included. Ascending, de-duplicated.
    """
    h, w = deploy_canvas
    deploy_area, aspect = h * w, w / h
    top = render_canvas(deploy_canvas, max_budget, align)
    top_area = top[0] * top[1]
    canvases = {deploy_canvas, top}
    min_area = 0 if min_budget > 0 else deploy_area
    budget = max(align, min_budget if min_budget > 0 else align)
    while True:
        rung = snap_max_content(budget * budget, aspect, align)
        r_area = rung[0] * rung[1]
        if r_area >= top_area:
            break
        if r_area >= min_area:
            canvases.add(rung)
        budget += align
    return sorted(canvases, key=lambda c: c[0] * c[1])
