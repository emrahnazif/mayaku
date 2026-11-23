"""Deployment: Predictor from a checkpoint, ONNX export with the sidecar
embedded, and the exported artifact run by onnxruntime -- all through the one
host decode the evaluator uses."""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import warnings

import pytest
import torch
from torchvision.ops import box_iou

from mayaku.config import InputConfig, KeypointConfig, MayakuConfig, ModelConfig
from mayaku.data.batch import batch_to
from mayaku.inference import ArtifactPredictor, Predictor, from_pretrained
from mayaku.inference.decode import (
    DEPLOY,
    PREDICT,
    PredictDecode,
    decode,
    decode_sidecar,
    host_side,
    predict_decode,
)
from mayaku.inference.export import TARGETS, _with_suffix, export, open_session
from mayaku.inference.export.metadata import embed_sidecar, read_sidecar, strip_tensorrt_header
from mayaku.inference.preprocess import letterbox_batch
from mayaku.model.quant import qdq_export
from mayaku.utils.checkpoint import build_sidecar, check_sidecar, save_checkpoint

from ._coco_fixture import fixture

ort = pytest.importorskip("onnxruntime")


def _trained_like(model):
    """Random but non-trivial weights with enough confidence to detect."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for c in model.head.cls:
            c.bias.add_(8.0)
    return model


@pytest.fixture(scope="module", params=[(128, 192), (160, 160)], ids=["rect", "square"])
def run(request, tmp_path_factory):
    """A tiny tier-n-shaped checkpoint with masks and keypoints, and its data."""
    root = tmp_path_factory.mktemp("run")
    canvas = request.param
    ds = fixture(root / "data", n=6, canvas=canvas, masks=True, kpt=3)
    cfg = MayakuConfig(
        model=ModelConfig(tier="n", num_classes=ds.nc, seg=True,
                          keypoints=KeypointConfig(num=3, names=("tl", "ctr", "br"))),
        input=InputConfig(canvas_hw=canvas),
        train=dataclasses.replace(MayakuConfig().train,
                                  decode=dataclasses.replace(MayakuConfig().train.decode, conf=0.3)))
    torch.manual_seed(0)   # the init too, so the weights do not depend on test order
    model = _trained_like(cfg.model.build(canvas))
    model.train()          # tier n is quantization-aware: give it int8 ranges
    with torch.no_grad():
        model(torch.rand(2, 3, *canvas))
    model.eval()
    ckpt = root / "best.pt"
    save_checkpoint(model, ckpt, build_sidecar(cfg, ds.coco.class_names, model))
    files = [str(f) for f in list(ds.coco.files)[:3]]
    return ds, ckpt, files


@pytest.fixture(scope="module")
def predictor(run):
    return Predictor.from_checkpoint(run[1], "cpu")


@pytest.fixture(scope="module")
def onnx_path(predictor, tmp_path_factory):
    return predictor.export("onnx", tmp_path_factory.mktemp("onnx") / "model.onnx")


def test_prediction_decodes_with_the_predict_block(run, predictor) -> None:
    """A call decodes with the sidecar's "predict" block (`PREDICT`), stated
    in a warning; the recorded "decode" block (conf 0.3 in this fixture) is
    the evaluation's, `decode="eval"`; `conf` overrides either."""
    files = run[2]
    sc = predictor.sidecar
    assert PredictDecode.from_sidecar(sc["predict"]) == PREDICT == predict_decode(sc)
    with pytest.warns(UserWarning, match="conf 0.4, NMS IoU 0.5"):
        for conf, floor in ((None, PREDICT.conf), (0.02, 0.02)):
            for g in predictor.batch(files, conf=conf):
                assert (g.scores >= floor).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # the eval decode is not announced
        for g in predictor.batch(files, conf=0.02, decode="eval"):
            assert (g.scores >= 0.02).all()
        for g in predictor.batch(files, decode="eval"):
            assert (g.scores >= sc["decode"]["conf"]).all()
    x, metas = letterbox_batch(files, predictor.canvas)
    want = decode_sidecar(predictor._forward(x), metas, sc, None, None, "predict")
    loose = decode(predictor._forward(x), metas, len(predictor.class_names), sc["decode"]["reg_max"],
                   seg=True, kpt=3, d=dataclasses.replace(PREDICT, iou=0.7))
    for g, w, wide in zip(predictor.batch(files), want, loose, strict=True):
        assert torch.equal(g.boxes, w.boxes) and torch.equal(g.labels, w.labels)
        assert len(g) <= len(wide)                    # NMS 0.5 keeps no more than 0.7
    with pytest.raises(ValueError, match="predict"):
        predictor(files[0], decode="coco")


def test_predict_decode_of_a_checkpoint_without_a_predict_block() -> None:
    """Checkpoints written before the "predict" block (the hosted ones) get
    `PREDICT`; a block, when present, is used as recorded."""
    rec = {"strides": [8, 16, 32], "reg_max": 16, **dataclasses.asdict(DEPLOY)}
    assert predict_decode({"decode": rec}) == PREDICT
    own = {"conf": 0.6, "iou": 0.6, "max_det": 20, "topk": 500, "multi_label": True}
    assert predict_decode({"decode": rec, "predict": own}) == PredictDecode(**own)


def test_a_partial_predict_config_keeps_the_predict_defaults() -> None:
    """Setting only some `train.predict` fields keeps `PREDICT`'s others,
    not the evaluation's (`DEPLOY`)."""
    cfg = MayakuConfig.model_validate({"train": {"predict": {"conf": 0.3}}})
    assert cfg.train.predict == dataclasses.replace(PREDICT, conf=0.3)
    assert MayakuConfig().train.predict == PREDICT and MayakuConfig().train.decode == DEPLOY


def test_predict_settings_stay_out_of_the_recorded_config(predictor, run) -> None:
    """The "predict" block is the only copy, so versions that predate
    `train.predict` still read the config; reading it back restores it."""
    from mayaku.utils.checkpoint import read_deploy_checkpoint

    assert "predict" not in predictor.sidecar["config"]["train"]
    assert read_deploy_checkpoint(run[1])[1].train.predict == PREDICT


def test_decode_runs_on_the_cpu_off_cuda() -> None:
    """`host_side` (the first step of `decode`) keeps CUDA tensors where they
    are and brings every other device's to the CPU, where MPS's decode is
    several times faster."""
    t = torch.zeros(2)
    assert host_side([t])[0] is t
    if torch.backends.mps.is_available():
        assert host_side([t.to("mps")])[0].device.type == "cpu"
    if torch.cuda.is_available():
        assert host_side([t.cuda()])[0].is_cuda


def test_predictor_matches_the_evaluation_path(run, predictor) -> None:
    """Image files through the Predictor give exactly the detections the
    dataset's eval path + decode give: same preprocessing, same decode."""
    ds, _, files = run
    got = predictor.batch(files, decode="eval")
    samples = [ds[i] for i in range(len(files))]
    x = batch_to(torch.stack([s["img"] for s in samples]), "cpu")
    want = decode_sidecar(predictor.model(x), [s["meta"] for s in samples], predictor.sidecar)
    assert sum(len(d) for d in got) > 0
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(g.boxes, w.boxes) and torch.equal(g.scores, w.scores)
        assert torch.equal(g.masks, w.masks) and torch.equal(g.keypoints, w.keypoints)
        assert tuple(g.masks.shape[1:]) == tuple(ds.shapes[i])


def test_predictor_takes_paths_bgr_arrays_and_pil_images_alike(run, predictor) -> None:
    """One image three ways: a path, the BGR array `cv2.imread` gives, and a
    PIL image (RGB by its type) -- the same detections."""
    import cv2
    from PIL import Image

    files = run[2]
    a = predictor(files[0])
    for other in (cv2.imread(files[0]), Image.open(files[0])):
        b = predictor(other)
        assert torch.equal(a.boxes, b.boxes) and torch.equal(a.labels, b.labels)


def test_predictor_leaves_the_given_model_alone(run) -> None:
    from mayaku.model.quant import is_qat
    from mayaku.utils.checkpoint import read_deploy_checkpoint

    sidecar, cfg, _ = read_deploy_checkpoint(run[1])
    model = cfg.model.build(cfg.input.canvas_hw, len(sidecar["class_names"]))
    Predictor(model, sidecar, "cpu")
    assert model.training is True and is_qat(model)
    assert any(type(m).__name__ == "RepConv3x3" for m in model.modules())   # not fused


def test_onnx_export_embeds_the_sidecar_and_runs_the_same(run, predictor, onnx_path) -> None:
    ds, _, files = run
    embedded = check_sidecar(read_sidecar(onnx_path, "onnx"), str(onnx_path))
    assert embedded == {**predictor.sidecar, "export": {"target": "onnx", "precision": "fp32"}}
    a = ArtifactPredictor(onnx_path, "cpu")
    assert a.canvas == predictor.canvas and a.class_names == ds.coco.class_names
    x, _ = letterbox_batch(files, predictor.canvas)
    for gm, wm in zip(a._forward(x), predictor._forward(x), strict=True):
        assert torch.allclose(gm, wm.cpu(), atol=1e-3)
    # decoded: confident detections have their twins. Near the threshold, or
    # tied in NMS, a 1e-4 difference in the maps may legitimately flip one;
    # the untrained model scores every class of its many identical clipped
    # boxes alike, so the multi-label decode meets such ties by the dozen.
    checked = missed = 0
    for g, w in zip(a.batch(files), predictor.batch(files), strict=True):
        for k in (g.scores > 0.5).nonzero().flatten().tolist():
            # its twin: same class, same box, closest score
            twin = (box_iou(g.boxes[k:k + 1], w.boxes)[0] > 0.99) & (w.labels == g.labels[k])
            gap = torch.where(twin, (w.scores - g.scores[k]).abs(), torch.inf)
            j = int(gap.argmin())
            checked += 1
            if not gap[j] < 1e-3:
                missed += 1
                continue
            assert torch.allclose(g.keypoints[k, :, :2], w.keypoints[j, :, :2], atol=1e-1)
            # a mask may differ only on its boundary pixels
            assert (g.masks[k] != w.masks[j]).float().mean() < 0.01
    assert missed <= max(1, checked // 20), (missed, checked)


def test_from_pretrained_picks_the_backend(run, onnx_path) -> None:
    assert isinstance(from_pretrained(run[1], "cpu"), Predictor)
    assert isinstance(from_pretrained(onnx_path, "cpu"), ArtifactPredictor)


def _runtime(target: str) -> None:
    """Skip unless this host can run `target` artifacts."""
    if not TARGETS[target].module.runnable():
        pytest.skip(f"{target} does not run on this host")
    pytest.importorskip({"onnx": "onnxruntime", "coreml": "coremltools",
                         "openvino": "openvino", "tensorrt": "tensorrt"}[target])


@pytest.mark.slow
@pytest.mark.parametrize(("target", "precision"), [
    ("onnx", "int8"), ("coreml", "fp16"), ("coreml", "int8"), ("openvino", "fp32"),
    ("openvino", "fp16"), ("openvino", "int8"), ("tensorrt", "fp16"), ("tensorrt", "int8"),
    ("tensorrt", "fp32"),
])
def test_every_target_exports_and_runs(run, predictor, tmp_path, target, precision) -> None:
    """Export checks the artifact's maps against the model itself; here the
    artifact also runs end to end through `ArtifactPredictor`."""
    _runtime(target)
    path = predictor.export(target, tmp_path / ("m" + TARGETS[target].suffix), precision)
    a = ArtifactPredictor(path, "cpu" if target != "tensorrt" else "cuda")
    assert a.sidecar["export"] == {"target": target, "precision": precision}
    # an int8 artifact runs the int8 graph, so it is held to the Torch int8
    # simulation, not to the float model (on this barely trained model many
    # detections sit at the threshold, where int8 rounding moves the count)
    # the recorded decode (conf 0.3, multi-label), as before prediction defaults existed
    with qdq_export(predictor.model) if precision == "int8" else contextlib.nullcontext():
        want = predictor.batch(run[2], decode="eval")
    got = a.batch(run[2], decode="eval")
    assert all(abs(len(g) - len(w)) <= max(3, len(w) // 10) for g, w in zip(got, want, strict=True))
    if target == "tensorrt":    # the first call records a CUDA graph, later calls replay it
        s = open_session(path, a.sidecar, "cuda")
        px = torch.randint(0, 256, (2, 1, *a.canvas, 3), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(0)).numpy()
        first = [m.clone() for m in s(px[0])]
        other = [m.clone() for m in s(px[1])]
        again = s(px[0])
        assert all(torch.equal(f, g) for f, g in zip(first, again, strict=True))
        assert not all(torch.equal(f, o) for f, o in zip(first, other, strict=True))


def test_int8_onnx_stores_int8_and_runs_as_int8(predictor, onnx_path, tmp_path) -> None:
    """Weights stored as int8 (a quarter of the fp32 file) in the form ONNX
    Runtime fuses into int8 convolutions, not float convs between extra
    (de)quantize steps."""
    import onnx
    import onnxruntime as ort

    int8 = predictor.export("onnx", tmp_path / "int8.onnx", "int8")
    assert int8.stat().st_size < 0.4 * onnx_path.stat().st_size
    options = ort.SessionOptions()
    options.optimized_model_filepath = str(tmp_path / "optimized.onnx")
    ort.InferenceSession(str(int8), options, providers=["CPUExecutionProvider"])
    ops = [n.op_type for n in onnx.load(options.optimized_model_filepath).graph.node]
    convs = {op: ops.count(op) for op in ("QLinearConv", "Conv", "FusedConv")}
    # the convs that feed a residual add or an output have no int8 output range
    # and stay float; on tier n the rest are a little over half of them
    assert convs["QLinearConv"] > 0.5 * sum(convs.values()), convs


def test_export_refuses_what_a_target_or_model_cannot_do(predictor, tmp_path) -> None:
    with pytest.raises(ValueError, match="fp16"):
        predictor.export("onnx", tmp_path / "m.onnx", "fp16")
    with pytest.raises(ValueError, match="unknown export target"):
        predictor.export("tflite", tmp_path / "m.tflite")
    with pytest.raises(ValueError, match=r"written as \.onnx, not \.xml"):
        predictor.export("onnx", tmp_path / "m.xml")
    plain = {**predictor.sidecar, "quant": {**predictor.sidecar["quant"], "qat": False}}
    with pytest.raises(ValueError, match="quantization-aware"):
        export(predictor.model, plain, "onnx", tmp_path / "m.onnx", "int8")


def test_export_adds_the_target_extension(tmp_path) -> None:
    """A name without the target's extension still gives a file that loads
    back by its suffix."""
    assert _with_suffix(tmp_path / "street.v2", "onnx") == tmp_path / "street.v2.onnx"
    assert _with_suffix(tmp_path / "street", "openvino") == tmp_path / "street.xml"
    assert _with_suffix(tmp_path / "street.ONNX", "onnx") == tmp_path / "street.ONNX"


def test_onnx_on_cuda_loads_its_libraries_and_repairs_a_cpu_fallback(monkeypatch) -> None:
    """onnxruntime-gpu finds the CUDA libraries of their pip packages only
    when preloaded, and one installed without them (cuRAND, which torch's
    wheels lack) silently runs on the CPU: the session installs them once and
    retries, and says so when that does not help."""
    import types

    from mayaku.inference.export import onnx as ox

    calls: list[str] = []
    cuda_loads = [False, True]                 # the first session falls back, the retry runs

    class Sess:
        def __init__(self, path, options, providers) -> None:
            calls.append("session")
            ok = "CUDAExecutionProvider" in providers and cuda_loads.pop(0)
            self.providers = providers if ok else ["CPUExecutionProvider"]

        def get_providers(self) -> list[str]:
            return self.providers

        def get_inputs(self) -> list:
            return [types.SimpleNamespace(name="x", shape=[1, 3, 64, 64])]

    ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        preload_dlls=lambda: calls.append("preload"), SessionOptions=types.SimpleNamespace,
        InferenceSession=Sess)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr(ox, "pip_spec", lambda name: "onnxruntime-gpu[cuda,cudnn]>=1.27")
    monkeypatch.setattr(ox, "install", lambda spec, why: calls.append(spec))
    s = ox.Session(None, ["cls0"], "fp32", "cuda")
    assert s._sess.get_providers()[0] == "CUDAExecutionProvider"
    assert calls == ["preload", "session", "onnxruntime-gpu[cuda,cudnn]>=1.27", "preload", "session"]

    cuda_loads[:] = [False, False]
    with pytest.warns(UserWarning, match="runs on the CPU"):
        ox.Session(None, ["cls0"], "fp32", "cuda")
    calls.clear()
    ox.Session(None, ["cls0"], "fp32", "cpu")          # the CPU asks for nothing
    assert calls == ["session"]


def test_an_onnx_file_without_a_sidecar_is_refused(onnx_path, tmp_path) -> None:
    import onnx

    m = onnx.load(str(onnx_path))
    del m.metadata_props[:]
    onnx.save(m, str(tmp_path / "bare.onnx"))
    with pytest.raises(ValueError, match="no mayaku sidecar"):
        ArtifactPredictor(tmp_path / "bare.onnx")


def test_export_rejects_a_graph_that_drifts(predictor, tmp_path, monkeypatch) -> None:
    """The parity check is what stands between a broken exporter and a user."""
    import mayaku.inference.export as ex

    monkeypatch.setattr(ex, "parity", lambda m, path, sidecar: 1.0)
    with pytest.raises(RuntimeError, match="differ"):
        export(predictor.model, predictor.sidecar, "onnx", tmp_path / "m.onnx")


def test_tensorrt_sidecar_header_round_trip(tmp_path) -> None:
    """An engine has no metadata slot: the sidecar goes in a length-prefixed
    header, which reading recovers and stripping removes byte-exactly."""
    engine = tmp_path / "m.engine"
    engine.write_bytes(b"\x00\x01ENGINE-BYTES")
    embed_sidecar(engine, "tensorrt", {"schema_version": 2, "x": "é"})
    assert read_sidecar(engine, "tensorrt")["x"] == "é"
    assert strip_tensorrt_header(engine) == b"\x00\x01ENGINE-BYTES"
    bare = tmp_path / "bare.engine"
    bare.write_bytes(b"\x00\x01ENGINE-BYTES")
    assert read_sidecar(bare, "tensorrt") is None and strip_tensorrt_header(bare) == bare.read_bytes()
