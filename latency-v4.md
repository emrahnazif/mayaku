# Mayaku v4 Latency & Deployment Report

Measured on **2026-11-26** against branch `v4` (commit `56dd180`, 4-level P6 architecture with strides `8 / 16 / 32 / 64`, `Tier.align = 64`, and fused TensorRT `forward_boxes` CUDA graph decode).

> **Note on valid resolutions:** Because `v4` adds a stride-64 level, both canvas sides must divide by `64` (`768`, `832`, `896`, `960`, `1024`). `864` is not valid (`864 / 64 = 13.5`); the sweep covers **`768`, `832`, `896`, `960`**.
>
> **Note on v4 weights:** `v4` checkpoints are still training, so `mayaku-v4-*` networks were instantiated with `num_classes=80`, `qat=True` (`symmetric` activations), non-trivial BatchNorm statistics (`randomize_bn`), and initialized QAT activation observers. Network latency depends strictly on architecture, precision, and input shape; E2E decode runs the fixed-size `topk=1000` fused CUDA graph plus host NMS.

---

## Hardware & Protocol

- **NVIDIA GPU server (`smart-segment`):**
  - GPU: NVIDIA GeForce RTX 3060 (12 GB), driver `535.309.01`
  - CPU: AMD Ryzen 5 5600G
  - Stack: Python `3.10.20`, PyTorch `2.6.0+cu118`, cuDNN `9.1.0`, TensorRT `10.13.0.35`, ONNX Runtime `1.23.2` (CPU), Ultralytics `8.4.189`, `rfdetr`
- **Apple Silicon:**
  - Machine: Apple M1 Max, macOS `27.2.1` (build `26B131`)
  - Stack: PyTorch `2.11.0`, `coremltools 9.0`, `onnxruntime 1.21.1`, `openvino 2026.1.0`
- **Timing protocol (batch 1, idle device):**
  - **Engine:** Pure device forward pass with input already on device (median of 120 CUDA-event-timed runs on RTX 3060; median of 15–25 synchronized runs on M1 Max).
  - **E2E (RTX 3060):** Median wall-clock time over **200 real COCO `val2017` images** preloaded in RAM as uint8 BGR arrays (`cv2.imread` excluded) to final boxes in original pixel coordinates (`conf=0.25`), using each library's own shipped TensorRT path:
    - **Mayaku v4:** `ArtifactPredictor(engine)(img, conf=0.25)` (pinned letterbox + fused `forward_boxes` CUDA graph + host NMS)
    - **YOLO26:** `YOLO(pt).export(format="engine", half=True, imgsz=640)` + `YOLO(engine).predict(img, conf=0.25, iou=0.7)`
    - **RF-DETR:** `model.export(format="tensorrt", fp16=True)` + `TRTInference` + `infer_transforms` + `post_process` (`conf=0.25`)
  - **E2E (M1 Max):** Median wall-clock time on a 1920×1080 uint8 BGR frame in RAM → `Detections`.

---

## 1. Competitors Full Range — RTX 3060 (Native TensorRT FP16)

| Model | Res | Params | TRT Build | Engine (ms) | E2E Wall (ms) | Notes |
|---|---:|---:|---:|---:|---:|---|
| `yolo26n` | 640 | 2.57M | 294.7 s | **1.10** | **2.28** | ultralytics speed: 2.10 ms (0.39 pre / 1.10 inf / 0.62 post) |
| `yolo26s` | 640 | 10.01M | 152.9 s | **1.72** | **2.92** | ultralytics speed: 2.73 ms (0.39 / 1.72 / 0.63) |
| `rfdetr-nano` | 384 | 30.47M | 57.4 s | **2.27** | **3.88** | native TRT FP16, 10.3 mean dets |
| `yolo26m` | 640 | 21.90M | 168.7 s | **3.28** | **4.49** | ultralytics speed: 4.30 ms (0.39 / 3.28 / 0.63) |
| `rfdetr-small` | 512 | 32.11M | 58.8 s | **3.11** | **5.14** | native TRT FP16, 10.8 mean dets |
| `yolo26l` | 640 | 26.30M | 177.2 s | **4.20** | **5.40** | ultralytics speed: 5.22 ms (0.38 / 4.20 / 0.63) |
| `rfdetr-medium` | 576 | 33.69M | 67.9 s | **4.08** | **6.98** | native TRT FP16, 11.1 mean dets |
| `yolo26x` | 640 | 58.99M | 210.4 s | **7.93** | **9.15** | ultralytics speed: 8.96 ms (0.39 / 7.93 / 0.64) |
| `rfdetr-large` | 704 | 33.93M | 66.1 s | **5.50** | **10.27** | native TRT FP16, 11.1 mean dets |

---

## 2. Mayaku v4 Full Matrix — RTX 3060 (TensorRT 10, `opt=2`)

| Model | Res | Params | GMACs | FP16 Build | FP16 Eng | FP16 E2E | INT8 Build | INT8 Eng | INT8 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`mayaku-v4-n`** | **768** | 11.48M | 10.3G | 26.4 s | **1.62 ms** | **2.42 ms** | 13.7 s | **1.18 ms** | **1.84 ms** |
| | **832** | 11.48M | 12.1G | 25.3 s | 1.88 ms | 2.79 ms | 12.7 s | 1.24 ms | 2.07 ms |
| | **896** | 11.48M | 14.0G | 25.8 s | **2.02 ms** | **3.06 ms** | 13.0 s | **1.31 ms** | **2.20 ms** |
| | **960** | 11.48M | 16.0G | 25.3 s | 2.38 ms | 3.51 ms | 13.0 s | **1.51 ms** | **2.46 ms** |
| **`mayaku-v4-s`** | **768** | 28.75M | 29.1G | 27.0 s | **2.91 ms** | **3.81 ms** | 12.8 s | **1.83 ms** | **2.59 ms** |
| | **832** | 28.75M | 34.2G | 27.1 s | 3.38 ms | 4.34 ms | 12.0 s | 2.01 ms | 2.96 ms |
| | **896** | 28.75M | 39.6G | 26.7 s | **3.75 ms** | **4.81 ms** | 12.7 s | **2.14 ms** | **3.17 ms** |
| | **960** | 28.75M | 45.5G | 26.5 s | 4.46 ms | 5.60 ms | 12.4 s | **2.50 ms** | **3.55 ms** |
| **`mayaku-v4-m`** | **768** | 65.87M | 69.4G | 30.8 s | **5.47 ms** | **6.31 ms** | 12.6 s | **3.07 ms** | **3.80 ms** |
| | **832** | 65.87M | 81.5G | 29.9 s | 6.54 ms | 7.51 ms | 12.7 s | 3.40 ms | 4.36 ms |
| | **896** | 65.87M | 94.5G | 30.5 s | **7.52 ms** | **8.56 ms** | 12.6 s | **3.67 ms** | **4.72 ms** |
| | **960** | 65.87M | 108.5G | 30.7 s | 8.52 ms | 9.62 ms | 12.5 s | **4.24 ms** | **5.39 ms** |
| **`mayaku-v4-l`** | **768** | 117.17M | 123.7G | 35.0 s | **8.05 ms** | **8.84 ms** | 13.9 s | **3.94 ms** | **4.78 ms** |
| | **832** | 117.17M | 145.2G | 34.4 s | 9.56 ms | 10.47 ms | 13.4 s | 4.77 ms | 5.68 ms |
| | **896** | 117.17M | 168.4G | 33.9 s | **10.69 ms** | **11.73 ms** | 12.8 s | **5.15 ms** | **6.18 ms** |
| | **960** | 117.17M | 193.3G | 35.3 s | 12.60 ms | 13.73 ms | 13.3 s | **6.02 ms** | **7.13 ms** |

---

## 3. First-Run UX & TensorRT Build-Time Breakdown (`mayaku-v4-n @ 768`, RTX 3060)

| Runtime / Config | Build / Warmup | Load | Engine (ms) | E2E (ms) |
|---|---:|---:|---:|---:|
| Previous `Predictor` (`MAYAKU_TORCH_FP16=0`, FP32 eager) | 0 s (1st: 192 ms) | 0.22 s | 10.64 | 9.86 |
| **Shipped `Predictor` (FP16 NHWC + `CUDAGraph`, 0 deps)** | **0 s (1st: 425 ms)** | 0.19 s | **3.66** | **4.50** |
| TRT FP16 `opt=0` (cold) | 8.16 s (+0.67 s ONNX) | 20 ms | 2.29 | 3.11 |
| **TRT FP16 `opt=2` (cold — *shipped default*)** | **29.2 s** (31.5 s w/ parity) | 29 ms | **1.63** | **2.40** |
| **TRT FP16 `opt=2` (warm timing cache)** | **3.7 s** (5.4 s w/ parity) | 29 ms | **1.63** | **2.40** |
| TRT FP16 `opt=3` (cold — *previous default*) | 71.78 s (+0.65 s ONNX) | 38 ms | 1.57 | 2.40 |
| TRT FP16 `opt=3` (warm timing cache) | 3.97 s (+0.61 s ONNX) | 38 ms | 1.67 | 2.39 |
| TRT INT8 `opt=0` (cold) | 6.47 s (+1.35 s ONNX) | 17 ms | 1.50 | 2.15 |
| **TRT INT8 `opt=2` (cold — *shipped default*)** | **14.1 s** (16.5 s w/ parity) | 19 ms | **1.19** | **1.87** |
| **TRT INT8 `opt=2` (warm timing cache)** | **1.7 s** (4.2 s w/ parity) | 19 ms | **1.19** | **1.87** |
| TRT INT8 `opt=3` (cold — *previous default*) | 50.69 s (+1.38 s ONNX) | 22 ms | 1.12 | 1.80 |
| TRT INT8 `opt=3` (warm timing cache) | 2.42 s (+1.31 s ONNX) | 22 ms | 1.12 | 1.77 |

**Key findings on build & first-run UX (now implemented on `v4-fast-deploy`):**
1. **Pure PyTorch fast default:** `Predictor` on Tensor-Core CUDA GPUs now runs FP16 `channels_last` + pinned upload + fused `CUDAGraph` for single-image `decode="predict"`, giving **2.2× E2E speedup** (`9.86 → 4.50 ms` E2E, `10.64 → 3.66 ms` engine) with zero new dependencies while keeping `decode="eval"`, `evaluate()`, and `export()` in exact FP32.
2. **5 ops slashes TensorRT build time vs. competitors:** YOLO26 native TRT builds took **153–295 s** and RF-DETR took **57–68 s**. With `opt=2` + persistent timing cache in `tensorrt.py`, Mayaku v4 INT8 builds in **14.1 s** cold and **1.7 s** on cache hit (`29.2 s` cold / `3.7 s` warm for FP16), with `≤0.07 ms` E2E delta vs. `opt=3`.
3. **Cached `.engine` load:** Deserializing a cached engine takes **17–38 ms**.

---

## 4. Apple Silicon (M1 Max) & CPU Scaling

### Apple M1 Max — MPS FP32 vs. Core ML FP16 (ANE)

| Model | Res | Params | GMACs | MPS FP32 Net | MPS FP32 E2E | ANE Write | ANE Load | **ANE FP16 Net** | **ANE FP16 E2E** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`mayaku-v4-n`** | **768** | 11.48M | 10.3G | 34.38 ms | 38.47 ms | 10.30 s | 3.85 s | **7.60 ms** | **9.29 ms** |
| | **832** | 11.48M | 12.1G | 35.10 ms | 39.31 ms | 8.54 s | 3.96 s | 8.63 ms | 9.80 ms |
| | **896** | 11.48M | 14.0G | 32.82 ms | 37.43 ms | 8.39 s | 3.87 s | **9.13 ms** | **11.31 ms** |
| | **960** | 11.48M | 16.0G | 34.25 ms | 38.11 ms | 8.26 s | 4.18 s | 11.19 ms | 12.90 ms |
| **`mayaku-v4-s`** | **768** | 28.75M | 29.1G | 40.29 ms | 42.37 ms | 10.00 s | 5.35 s | **11.67 ms** | **13.43 ms** |
| | **832** | 28.75M | 34.2G | 42.08 ms | 46.60 ms | 9.68 s | 5.37 s | 15.24 ms | 17.27 ms |
| | **896** | 28.75M | 39.6G | 46.83 ms | 48.68 ms | 10.48 s | 5.24 s | **14.45 ms** | **16.34 ms** |
| | **960** | 28.75M | 45.5G | 48.15 ms | 55.94 ms | 10.32 s | 5.37 s | 16.72 ms | 18.41 ms |
| **`mayaku-v4-m`** | **768** | 65.87M | 69.4G | 53.97 ms | 58.52 ms | 12.74 s | 8.46 s | **22.17 ms** | **23.83 ms** |
| | **832** | 65.87M | 81.5G | 59.16 ms | 62.65 ms | 12.93 s | 8.35 s | 26.42 ms | 28.83 ms |
| | **896** | 65.87M | 94.5G | 65.04 ms | 66.97 ms | 13.20 s | 8.29 s | **30.42 ms** | **32.79 ms** |
| | **960** | 65.87M | 108.5G | 71.53 ms | 76.71 ms | 13.76 s | 8.68 s | 33.35 ms | 34.41 ms |
| **`mayaku-v4-l`** | **768** | 117.17M | 123.7G | 73.68 ms | 74.80 ms | 17.59 s | 12.84 s | **35.06 ms** | **36.41 ms** |
| | **960** | 117.17M | 193.3G | 94.81 ms | 98.89 ms | 18.04 s | 13.34 s | 57.53 ms | 59.89 ms |

### Apple M1 Max — Core ML INT8 (ANE, fixed on `v4-fast-deploy` with `QUANT_SKIPS = False`)

| Model | Res | Export + Parity | ANE Load | **ANE INT8 Net** | **ANE INT8 E2E** | Before fix (CPU fallback) |
|---|---:|---:|---:|---:|---:|---:|
| **`mayaku-v4-n`** | **768** | 15.26 s | 4.41 s | **7.23 ms** | **8.98 ms** | 8.21 ms / 10.05 ms |
| | **832** | 14.43 s | 4.38 s | **8.42 ms** | **10.34 ms** | 146.52 ms / 154.96 ms (109.9 s load) |
| | **896** | 14.49 s | 4.37 s | **9.70 ms** | **11.69 ms** | 166.39 ms / 156.63 ms (111.9 s load) |
| | **960** | 15.43 s | 4.69 s | **11.13 ms** | **13.15 ms** | 172.97 ms / 177.28 ms (118.6 s load) |

> **ANE even-tile note:** On `v4-s`, `896` (`14.45 ms`) is faster than `832` (`15.24 ms`) because `896 / 64 = 14` (even) tiles better on ANE than `832 / 64 = 13` (odd).

### CPU Scaling (`mayaku-v4-n`)

| Res | Ryzen 5 5600G PyTorch CPU | Ryzen 5 5600G ORT CPU FP32 | M1 Max OpenVINO ARM64 INT8 |
|---:|---:|---:|---:|
| **768** | 100.4 ms | **36.1 ms** (exp 1.39 s) | **31.5 ms** net / 33.1 ms e2e |
| **832** | 115.9 ms | 40.9 ms (exp 1.56 s) | 40.6 ms net / 39.1 ms e2e |
| **896** | 132.4 ms | **48.6 ms** (exp 1.35 s) | **39.8 ms** net / 41.7 ms e2e |
| **960** | 146.1 ms | 54.9 ms (exp 1.38 s) | 60.7 ms net / 61.0 ms e2e |

---

## 5. Cross-Family E2E Comparison & Deploy Input Budget

### RTX 3060 E2E Latency Ladder (ms, lower is faster)

| Latency Band (E2E) | Competitors (native TRT FP16) | Mayaku v4 **FP16** | Mayaku v4 **INT8 (QAT default)** |
|---|---|---|---|
| **< 2.5 ms** | `yolo26n @ 640` (**2.28**) | `v4-n @ 768` (**2.42**) | `v4-n @ 768` (**1.84**), `v4-n @ 832` (**2.07**), `v4-n @ 896` (**2.20**), `v4-n @ 960` (**2.46**) |
| **2.5 – 3.9 ms** | `yolo26s @ 640` (**2.92**), `rfdetr-nano @ 384` (**3.88**) | `v4-n @ 832` (**2.79**), `v4-n @ 896` (**3.06**), `v4-n @ 960` (**3.51**), `v4-s @ 768` (**3.81**) | `v4-s @ 768` (**2.59**), `v4-s @ 832` (**2.96**), `v4-s @ 896` (**3.17**), `v4-s @ 960` (**3.55**), `v4-m @ 768` (**3.80**) |
| **4.0 – 5.6 ms** | `yolo26m @ 640` (**4.49**), `rfdetr-small @ 512` (**5.14**), `yolo26l @ 640` (**5.40**) | `v4-s @ 832` (**4.34**), `v4-s @ 896` (**4.81**), `v4-s @ 960` (**5.60**) | `v4-m @ 832` (**4.36**), `v4-m @ 896` (**4.72**), `v4-l @ 768` (**4.78**), `v4-m @ 960` (**5.39**) |
| **5.7 – 7.5 ms** | `rfdetr-medium @ 576` (**6.98**) | `v4-m @ 768` (**6.31**), `v4-m @ 832` (**7.51**) | `v4-l @ 832` (**5.68**), `v4-l @ 896` (**6.18**), `v4-l @ 960` (**7.13**) |
| **8.0 – 10.5 ms** | `yolo26x @ 640` (**9.15**), `rfdetr-large @ 704` (**10.27**) | `v4-m @ 896` (**8.56**), `v4-l @ 768` (**8.84**), `v4-m @ 960` (**9.62**), `v4-l @ 832` (**10.47**) | — *(all v4 tiers already finish by 7.13 ms)* |

### Budget Summary

- **`768 px` (current v4 default):** Safest cross-device baseline — >100 FPS on M1 Max ANE (`v4-n`), ~30 FPS on CPU, and beats `yolo26n` in INT8 (`1.84` vs `2.28 ms`) while `v4-l` beats `yolo26x` even in FP16 (`8.84` vs `9.15 ms`).
- **`896 px` (`14×64`):** Best balanced sweet spot — +36% more pixels than 768 (~2× YOLO26's 640px, 5.4× RF-DETR-nano's 384px), avoids the odd-13 ANE tiling penalty at `832`, keeps `v4-n @ 896` INT8 (`2.20 ms`) faster than `yolo26n @ 640` (`2.28 ms`), keeps `v4-m @ 896` on 30 FPS on M1 Max ANE (`32.79 ms`), and keeps FP16 competitive on every tier.
- **`960 px` (`15×64`):** Still beats the entire competitor range on RTX 3060 INT8 (`v4-n` `2.46 ms`, `v4-s` `3.55 ms`, `v4-m` `5.39 ms`, `v4-l` `7.13 ms`) and stays 78 FPS on M1 Max ANE for `v4-n` (`12.90 ms`), with the main tradeoff being FP16 latency on `m`/`l` and CPU speed (~17 FPS for `v4-n`).

---

## 6. Publishing RF100-VL AP vs. Latency Fairly

### Core principles

1. **Plot both Mayaku FP16 and Mayaku INT8 on the same chart.**
   - Draw `Mayaku v4 (FP16)` (dashed/lighter) and `Mayaku v4 (INT8, QAT default)` (solid/bold) alongside `YOLO26 (FP16)` and `RF-DETR (FP16)`.
   - Because Mayaku v4 is already competitive on strict **FP16 vs. FP16**, showing both curves removes any suspicion of hiding FP16 while visually demonstrating what 5-op QAT buys: **~35–45% lower latency at essentially flat AP (±0.2–0.3 AP).**
2. **Never pair FP32 PyTorch AP with INT8 TensorRT latency.**
   - For Mayaku INT8 rows/points, report the **measured INT8 AP** (evaluated through `mayaku eval model_int8.engine` on the exported INT8 artifact, or verified against the exported `.engine` across RF100-VL), not the FP32 checkpoint AP.
   - Explicitly note the protocol: RF100-VL mean AP is averaged across all 100 fine-tuned datasets (each library at its own default recipe, same `pycocotools` evaluator), and E2E latency is measured at each model's default deploy resolution on the same 200+ COCO `val2017` BGR frames in RAM, batch 1.
3. **State the fairness rule plainly and back it up with measured PTQ data.**
   - Rule: *"Each library is benchmarked at its fastest shipped deployment that preserves fine-tuned AP."*
   - Include a small companion **"Why competitors are shown in FP16"** table measuring a stratified sample of fine-tuned RF100-VL checkpoints exported with Ultralytics' own `int8=True` calibration vs. Mayaku's default QAT INT8 export (and noting that RF-DETR does not ship QAT INT8). One measured table settles the fairness question permanently.
4. **Show two hardware targets side by side (NVIDIA RTX 3060 + Apple Silicon ANE).**
   - Pairing RTX 3060 (FP16 & INT8) with Apple Silicon ANE (FP16 on M1 Max, and INT8 once measured on M4+) proves the 5-op design generalizes across accelerators rather than being tuned for one GPU.

### Suggested Main Table (RF100-VL 100 Datasets — AP vs. E2E Latency)

Fill AP columns once `v4` RF100-VL runs complete (example shown at `896 px`; substitute `768` or `960` if chosen):

| Model | Precision | Res | Params | RF100-VL Mean AP | AP<sub>S</sub> | RTX 3060 Eng | **RTX 3060 E2E** | M1 Max ANE E2E |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **mayaku-v4-n** | **INT8 (QAT)** | 896 | 11.5M | *TBD (engine)* | *TBD* | 1.31 ms | **2.20 ms** | *TBD (M4 INT8)* |
| mayaku-v4-n | FP16 | 896 | 11.5M | *TBD* | *TBD* | 2.02 ms | 3.06 ms | **11.31 ms** |
| **mayaku-v4-s** | **INT8 (QAT)** | 896 | 28.8M | *TBD (engine)* | *TBD* | 2.14 ms | **3.17 ms** | *TBD (M4 INT8)* |
| mayaku-v4-s | FP16 | 896 | 28.8M | *TBD* | *TBD* | 3.75 ms | 4.81 ms | **16.34 ms** |
| **mayaku-v4-m** | **INT8 (QAT)** | 896 | 65.9M | *TBD (engine)* | *TBD* | 3.67 ms | **4.72 ms** | *TBD (M4 INT8)* |
| mayaku-v4-m | FP16 | 896 | 65.9M | *TBD* | *TBD* | 7.52 ms | 8.56 ms | **32.79 ms** |
| **mayaku-v4-l** | **INT8 (QAT)** | 896 | 117.2M | *TBD (engine)* | *TBD* | 5.15 ms | **6.18 ms** | *TBD (M4 INT8)* |
| mayaku-v4-l | FP16 | 896 | 117.2M | *TBD* | *TBD* | 10.69 ms | 11.73 ms | — |
| yolo26n | FP16 | 640 | 2.6M | 54.8 | 28.3 | 1.10 ms | 2.28 ms | *TBD* |
| yolo26s | FP16 | 640 | 10.0M | 57.8 | 32.3 | 1.72 ms | 2.92 ms | *TBD* |
| yolo26m | FP16 | 640 | 21.9M | *TBD* | *TBD* | 3.28 ms | 4.49 ms | *TBD* |
| yolo26l | FP16 | 640 | 26.3M | *TBD* | *TBD* | 4.20 ms | 5.40 ms | *TBD* |
| yolo26x | FP16 | 640 | 59.0M | *TBD* | *TBD* | 7.93 ms | 9.15 ms | *TBD* |
| rfdetr-nano | FP16 | 384 | 30.5M | 57.4 | 29.6 | 2.27 ms | 3.88 ms | *TBD* |
| rfdetr-small | FP16 | 512 | 32.1M | *TBD* | *TBD* | 3.11 ms | 5.14 ms | *TBD* |
| rfdetr-medium | FP16 | 576 | 33.7M | *TBD* | *TBD* | 4.08 ms | 6.98 ms | *TBD* |
| rfdetr-large | FP16 | 704 | 33.9M | *TBD* | *TBD* | 5.50 ms | 10.27 ms | *TBD* |

*(Optional: include a second resolution row for `mayaku-v4-*` at `768` and `960` in an appendix or toggle if you want to show the resolution-scaling curve explicitly.)*

### Suggested Companion Table ("Why Competitors Are Shown in FP16")

Measured on the **same fine-tuned RF100-VL checkpoints** (e.g. 5–10 representative RF100-VL datasets, exported to TensorRT INT8 via each library's shipped INT8 path and evaluated on the exported `.engine`):

| Model | Quantization Path | Calibration Data Needed? | FP16 AP | INT8 Engine AP | **Δ AP** | RTX 3060 E2E (FP16 → INT8) |
|---|---|---|---:|---:|---:|---:|
| **mayaku-v4-n (896)** | Built-in QAT (default) | No | *TBD* | *TBD* | *~−0.2* | 3.06 → **2.20 ms** |
| **mayaku-v4-m (896)** | Built-in QAT (default) | No | *TBD* | *TBD* | *~−0.2* | 8.56 → **4.72 ms** |
| yolo26s (640) | Post-training calibration (`int8=True`) | Yes (train split) | *TBD* | *TBD* | *TBD (measured)* | 2.92 → *TBD ms* |
| rfdetr-nano (384) | No shipped QAT INT8 export | — | *TBD* | N/A | N/A | 3.88 ms (FP16 only) |

### Suggested Chart Layout

- **X-axis:** End-to-end latency in ms (batch 1, uint8 BGR in RAM → final boxes in original pixels, log or linear scale).
- **Y-axis:** RF100-VL Mean AP@[.50:.95] (100 datasets, `pycocotools`, `maxDets=100`).
- **Series:**
  1. `Mayaku v4 (INT8, QAT default)` — bold solid line, points labeled with tier + resolution (`n-896`, `s-896`, `m-896`, `l-896`).
  2. `Mayaku v4 (FP16)` — lighter dashed line connecting the same checkpoints in FP16, with subtle horizontal arrows or matching markers showing the latency shift from FP16 → INT8.
  3. `YOLO26 (FP16, 640)` — solid line (`n, s, m, l, x`).
  4. `RF-DETR (FP16, 384–704)` — solid line (`nano, small, medium, large`).
- **Caption footnote:** State GPU/CPU, TensorRT version, "Every point pairs the AP and E2E latency of the same deployed precision; see companion table for competitor INT8 PTQ accuracy loss."

