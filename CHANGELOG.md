# Changelog

## Unreleased

- v4 detector architecture for medium and large objects (requires fresh O365 base pretraining; not
  weight-compatible with v3 checkpoints):
  - All detection tiers (`n`, `s`, `m`, `l`) add a fourth stride-64 level (`strides=[8, 16, 32, 64]`,
    canvas sides aligned to 64, default `input.size_budget` updated from `800` to `768`, giving
    `768×768` for square/mixed datasets and `576×1024` for 16:9), two extra
    backbone bottlenecks and a residual `3×3→3×3` class block at stride 32, and widened stride-32 /
    stride-64 head towers.
  - Neck bottom-up downsamples (`FPGStage.down`) now use `3×3` stride-2 convolutions instead of `1×1`
    stride-2 so every row and column participates in coarse-level features.
  - Instance mask branch widens `MASK_CH` from `8` to `16` (dynamic kernel length `233`) and adds a
    `3×3` convolution after the multi-scale feature sum before mask readout.
  - Multi-scale training supports `train.multiscale_max` above the deploy canvas (`render_canvas` +
    `multi_scale_canvases`) so batches render at the upper size and scale across budgets both below
    and above deployment size while evaluation remains at the deploy canvas.
  - Faster TensorRT single-image detection inference (`ArtifactPredictor` / `TensorRTSession`):
    caches letterbox border padding when image placement matches, folds top-K selection and DFL box
    decoding into the TensorRT CUDA graph with async pinned-memory copy of the candidates, caches DFL
    bin centres (`_dfl_bins`), and runs sorted host NMS + unletterboxing (`decode_candidates`). On
    an RTX 3060 over 2,000 COCO `val2017` images (`conf=0.25`, `decode="eval"`), `mayaku4` at
    `768×768` measures (`net` / `e2e` median): `n` `1.74 / 2.58 ms` (`fp16`), `1.25 / 2.10 ms`
    (`int8`); `s` `3.08 / 3.89 ms` (`fp16`), `1.91 / 2.73 ms` (`int8`); `m` `5.73 / 6.51 ms`
    (`fp16`), `3.22 / 4.03 ms` (`int8`); `l` `8.61 / 9.29 ms` (`fp16`), `4.49 / 5.31 ms` (`int8`);
    and on `mayaku3-n` at `800×800`, end-to-end latency drops from `2.83 / 2.30 ms` to
    `2.40 / 1.88 ms` (`fp16 / int8`).

- int8 ONNX files are a quarter of the size and run as int8 in onnxruntime. Weights were stored
  as fp32 behind a quantize step, so the file was as large as fp32 and onnxruntime ran every conv
  in float between extra quantize and dequantize steps, slower than the fp32 file. Weights are now
  stored as int8, biases as int32 and activations as uint8 on the same grid, the form onnxruntime
  fuses into int8 convolutions; the network is the same. Street nano seg (1,500 street validation
  images, onnxruntime CPU): 21.5 MB to 5.8 MB, box AP 44.21 to 44.29 (fp32 44.14); one 800x800
  image on an i9-10900X takes 131 ms on one thread instead of 201 (fp32 127), and 121 ms in
  Chrome with onnxruntime-web instead of 161 (fp32 150). Street medium seg: 131 MB to 34 MB, box
  AP 56.26 to 56.14 (fp32 56.28). The convs that feed a residual add or an output have no trained
  int8 output range and stay float, so on a many-core CPU fp32 is still faster (nano, 20 threads:
  63 ms, fp32 26). OpenVINO and TensorRT exports are unchanged.

- The docs tell the two kinds of hosted models apart: base models to fine-tune, and
  ready-to-use models to run as they are. The street models have their own page: classes,
  accuracy, speed, a comparison with COCO models and known limitations.

## v3.1.1: Street models

- int8 ONNX and OpenVINO exports lose less accuracy. Most activations are ReLU outputs, which are
  never negative, and the symmetric int8 grid gave them only its 128 non-negative levels. These
  exports now give such a tensor the whole int8 grid over the same trained range: half the step,
  and every level the model trained with is still on the grid, so existing checkpoints improve
  without retraining. On the street models (1,467 validation images, ONNX Runtime), the int8 box
  AP gap to fp32 goes from 0.37 to -0.13 for nano and from 0.44 to 0.18 for m; OpenVINO gives the
  same. TensorRT has no zero point and keeps the trained symmetric grid; Core ML keeps it until
  it is measured on the Neural Engine.
- `mayaku.model.export_onnx` takes `regrid=True` for that int8 grid; its default output is
  unchanged.
- Hosted street models: `mayaku-n-det-street`, `mayaku-n-seg-street`, `mayaku-m-det-street` and
  `mayaku-m-seg-street` detect 22 road and street classes, e-scooters, license plates, faces and
  emergency vehicles included, from street level to drone and aerial views. On the street
  validation set (17,712 images) nano scores 43.9 box AP and 35.2 mask AP, medium 56.8 and 45.5.
  They download by name in every 3.x version; the docs now list them.
- Export runtimes install in environments made by `uv venv`, which have no pip: the install falls
  back to `uv pip`. Before, every ONNX and OpenVINO export failed there with `No module named pip`.
- ONNX models run on the GPU. onnxruntime-gpu needs CUDA libraries that torch's CUDA 13 wheels
  no longer install (cuRAND), and it does not find the pip-installed ones by itself, so every
  ONNX session silently ran on the CPU. Mayaku now installs `onnxruntime-gpu[cuda,cudnn]` and
  loads those libraries before a CUDA session; an existing install without them is completed the
  first time it falls back, with a warning if CUDA still does not load. Street nano seg, one
  1080p frame end to end on an RTX 3090: 18 ms instead of 87 ms; street medium 22 ms instead
  of 232 ms.
- An export path without the target's extension gets it added: `export("openvino", "street")`
  writes `street.xml`. Before, OpenVINO failed with an internal error and ONNX wrote a file with no
  extension that could not be loaded back. A path with another target's extension is refused.

## v3.1.0: Predictions ready for applications

- Predictions now use their own post-processing: a score threshold, NMS at IoU 0.5 and one class
  per location. Before, a prediction returned the evaluation's output: every detection down to a
  score of 0.001, often several boxes of different classes on one object. `decode="eval"` still
  gives the evaluation's output.
- Training chooses the prediction score threshold on the validation split (the best F1) and saves
  it in the checkpoint. Without a validation split of at least 50 objects, it stays at 0.4.
  `mayaku tune-predict` and `mayaku.tune_predict` choose it for an existing checkpoint.
- `mayaku.evaluate(..., operating=True)` also returns the best-F1 threshold of the split.
- `best.pt` is chosen on what the model is for: mask AP for segmentation, keypoint AP for keypoints,
  box AP for detection, top-1 for classification. With quantization-aware training, every
  evaluation scores the whole validation split in both fp32 and int8, and `best.pt` is the epoch
  with the best lower of the two.
- COCO metrics use faster-coco-eval, now a dependency: the same numbers, several times faster on
  large splits. Segmentation evaluation computes masks only inside their boxes: the same masks,
  about 3x faster during training and 13x faster for exported models.
- Prediction post-processing runs on the CPU unless the model runs on CUDA. On an M1 Max, a 1080p
  frame of a segmentation model decodes in 8.5 ms instead of 107 ms on MPS.
- The documentation pages now come with the package, in the `docs` folder next to the code, with
  `llms.txt` listing every page. Coding agents can read the docs of the installed version offline.
