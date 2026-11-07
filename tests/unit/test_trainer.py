"""The training loop: parameter groups, schedule, EMA, and a real tiny run."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from mayaku.data import DEFAULT_AUG
from mayaku.engine.trainer import (
    BASE,
    EMA,
    build_optimizer,
    cosine,
    param_groups,
    set_lr,
    train,
)
from mayaku.model import TINY, Detector, enable_qat

from ._coco_fixture import fixture


def test_param_groups_cover_every_parameter_once() -> None:
    model = Detector(TINY, 4)
    groups = param_groups(model, 5e-4)
    sizes = [sum(p.numel() for p in g["params"]) for g in groups]
    assert sum(sizes) == sum(p.numel() for p in model.parameters())
    assert groups[1]["weight_decay"] == groups[2]["weight_decay"] == 0.0
    assert [g["bias"] for g in groups] == [False, False, True]


def test_cosine() -> None:
    assert abs(cosine(BASE, 0) - 1.0) < 1e-9
    assert abs(cosine(BASE, BASE.epochs) - BASE.lr_final_frac) < 1e-9
    assert cosine(BASE, 62) < cosine(BASE, 0)


def test_warmup_hands_off_to_the_cosine() -> None:
    opt = build_optimizer(Detector(TINY, 4), BASE, accumulate=4)
    set_lr(opt, BASE, 0, 0, 100)
    # only the bias group starts above zero; a BatchNorm gain must not
    assert [g["lr"] for g in opt.param_groups] == [0.0, 0.0, BASE.warmup_bias_lr_start]
    set_lr(opt, BASE, 3, 100, 100)
    assert all(abs(g["lr"] - BASE.lr * cosine(BASE, 3)) < 1e-12 for g in opt.param_groups)


def test_adamw_biases_warm_up_from_zero() -> None:
    """The SGD bias warmup's high start would be a huge AdamW step."""
    r = dataclasses.replace(BASE, optimizer="adamw", lr=0.001)
    opt = build_optimizer(Detector(TINY, 4), r, accumulate=4)
    set_lr(opt, r, 0, 0, 100)
    assert [g["lr"] for g in opt.param_groups] == [0.0, 0.0, 0.0]


def test_ema_lags_the_model() -> None:
    model = Detector(TINY, 4)
    ema = EMA(model, decay=0.9, tau=1.0)
    ref = next(model.parameters()).clone()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    ema.update()
    moved = (next(ema.model.parameters()) - ref).abs().mean().item()
    assert 0.0 < moved < 1.0


@pytest.mark.slow
@pytest.mark.parametrize("qat", [False, True], ids=["fp32", "qat"])
def test_tiny_run_learns(tmp_path, qat) -> None:
    """32 synthetic images through a tiny network: AP has to leave zero. The
    gate is loose on purpose -- CPU convolution is not bit-deterministic --
    so it asserts that training happened, not that it reached a number.

    `lr_ref_batch` is 4, not 64: at batch 4 the default would accumulate 16
    batches per update and 36 epochs on 32 images would be 18 steps."""
    r = dataclasses.replace(
        BASE, epochs=36, batch=4, lr_ref_batch=4, lr=0.02,
        final_frac=0.22, warmup_epochs=2.5,
        assigner_warmup=2, amp=False, recalibrate_images=32)
    tr = fixture(tmp_path / "train", n=32, canvas=192, aug=DEFAULT_AUG, seed=3)
    va = fixture(tmp_path / "val", n=32, canvas=192, seed=3)
    model = Detector(TINY, tr.nc, tr.canvas)
    if qat:
        enable_qat(model)
    best, _, records = train(model, tr, va, r, eval_every=12, log_every=0,
                             out=str(tmp_path / "run"), log=lambda *_: None)
    assert best["AP50"] > 0.08
    assert records[-1]["cls"] < records[0]["cls"] / 1.5
    assert records[-1]["box"] < records[0]["box"] / 1.5
    assert (tmp_path / "run" / "best.pt").exists()
    # the int8 score is logged and ranked on with QAT and only then
    assert ("AP_int8" in records[-1]) == qat
    assert ("AP_int8" in best) == qat


def test_ema_int8_ranges_count_as_observed_with_the_live_ones() -> None:
    """The EMA copies the live range buffers; it must also take over whether
    they were observed, or its first int8 forward would re-observe them."""
    from mayaku.engine.trainer import EMA
    from mayaku.model.quant import observers

    model = enable_qat(Detector(TINY, 2, 64))
    ema = EMA(model)
    model.train()(torch.rand(2, 3, 64, 64))
    assert all(o.observed for o in observers(model))
    ema.update()
    assert all(o.observed for o in observers(ema.model))


def test_a_step_is_the_same_size_at_any_micro_batch(tmp_path, monkeypatch) -> None:
    """A step's gradient sums its micro-batches, so its size does not shrink
    with `batch` (averaging them made it 4x smaller here); BatchNorm makes
    the two agree only roughly."""
    norms = {}

    def first_norm(key):
        real = torch.nn.utils.clip_grad_norm_

        def clip(params, max_norm):
            n = real(params, max_norm)
            if key not in norms:
                norms[key] = float(n)
            return n
        return clip

    ds = fixture(tmp_path / "d", n=16, canvas=96, seed=3)
    for batch in (2, 8):
        torch.manual_seed(0)
        model = Detector(TINY, ds.nc, ds.canvas)
        r = dataclasses.replace(BASE, epochs=1, batch=batch, lr_ref_batch=8, warmup_epochs=0,
                                amp=False, recalibrate_images=0)
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", first_norm(batch))
        train(model, ds, None, r, log_every=0, log=lambda *_: None)
    assert 0.5 < norms[2] / norms[8] < 2.0


def test_rank_score_is_the_lower_of_fp32_and_int8() -> None:
    """A QAT epoch ranks on its weaker precision, so int8 beating float by
    chance cannot pick it; a record without an int8 score (a float model,
    or a `best` resumed from an older state.pt) ranks on fp32."""
    from mayaku.engine.trainer import rank_score
    assert rank_score({"AP": 0.50, "AP_int8": 0.47}, "AP") == 0.47
    assert rank_score({"AP": 0.48, "AP_int8": 0.51}, "AP") == 0.48
    assert rank_score({"AP": 0.50}, "AP") == 0.50
    # no best yet: ranks last
    assert rank_score({}, "AP") == -1.0

