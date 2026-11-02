---
title: Export to ONNX, Core ML, OpenVINO and TensorRT
summary: Export a trained model for deployment, choose a precision, use int8, and run it with or without Mayaku.
label: Export
keywords: mayaku, export, onnx, coreml, openvino, tensorrt
---

`mayaku export` turns a trained checkpoint into a file for a deployment runtime: ONNX, Core ML, OpenVINO or TensorRT. Every task exports: detection, segmentation, keypoints, oriented boxes and classification.

```bash
mayaku export onnx     runs/exp/train/best.pt --output model.onnx
mayaku export coreml   runs/exp/train/best.pt --output model.mlpackage
mayaku export openvino runs/exp/train/best.pt --output model.xml        # writes model.bin next to it
mayaku export tensorrt runs/exp/train/best.pt --output model.engine     # on a CUDA machine
```

The exported file runs with the same tools as the checkpoint:

```bash
mayaku predict model.onnx photo.jpg
mayaku eval model.onnx --annotations data/test/_annotations.coco.json --images data/test
```

```python
from mayaku import from_pretrained

predictor = from_pretrained("model.onnx")
det = predictor("photo.jpg", conf=0.25)
```

## Targets

| Target | File | Precisions (first is the default) | Runs on | Install |
|---|---|---|---|---|
| `onnx` | `.onnx` | `fp32`, `int8` | onnxruntime: CPU, or CUDA with `onnxruntime-gpu` | `pip install "mayaku[onnx]" onnxruntime` |
| `coreml` | `.mlpackage` | `fp16`, `fp32`, `int8` | Apple devices: Neural Engine, GPU, CPU | `pip install "mayaku[coreml]"` |
| `openvino` | `.xml` + `.bin` | `fp32`, `fp16`, `int8` | Intel CPUs, integrated GPUs, NPUs | `pip install "mayaku[openvino]"` |
| `tensorrt` | `.engine` | `fp16`, `int8`, `fp32` | NVIDIA GPUs | `pip install "mayaku[tensorrt]"` |

Choose a precision with `--precision`:

```bash
mayaku export tensorrt best.pt --precision int8 --output model_int8.engine
```

Without `--output`, the file is written next to the checkpoint with the target's extension: `runs/exp/train/best.pt` becomes `runs/exp/train/best.onnx`.

## What export does

1. Loads the checkpoint on the CPU and builds its deployment graph at the model's canvas size.
2. Writes the file for the target and precision.
3. Embeds the model's description in the file's metadata: class names, canvas, preprocessing, post-processing values, task settings, precision. This is why an exported file needs no config.
4. **Checks the file.** It runs the exported file once in its runtime and compares the raw outputs with the PyTorch model on the same input. If they differ by more than the tolerance (0.1% RMS for fp32, 1% for fp16, 15% for int8), export fails with an error like `model.mlpackage: coreml fp16 outputs differ from the Torch model by 3.20% RMS (tolerance 1.0%)`. The file is not returned as a success.

The check is skipped, with the file still written, only when the target's runtime can't run on this machine: Core ML anywhere but macOS. TensorRT export always needs a CUDA GPU, both to build the engine and to check it.

## From Python

```python
from mayaku import from_pretrained

predictor = from_pretrained("runs/exp/train/best.pt", device="cpu")
path = predictor.export("onnx", "model.onnx")                 # default precision
path = predictor.export("coreml", "model_int8.mlpackage", precision="int8")
```

`export(target="onnx", output=None, precision=None)` returns the path written. Without `output`, it writes `model.<extension>` in the current directory.

## The exported graph

The exported model contains only five operations: 1×1 and 3×3 convolutions, ReLU, addition, nearest-neighbor upsampling, and 3×3 max-pooling. There are no custom operators, attention or deformable convolutions, so any runtime that supports these ops can run it. Export asserts this on every ONNX file it writes.

- **Fixed input size.** The input is one image (batch 1) at the model's canvas, `(1, 3, H, W)`. Find the canvas in `predictor.canvas` or in `canvas_hw` in the embedded metadata. `predictor.batch([...])` on an exported model runs the images one at a time.
- **Input.** ONNX, OpenVINO and TensorRT take a float32 tensor named `images`, RGB, values scaled to 0–1, already letterboxed (see [Prediction](/mayaku/deployment/prediction/#preprocessing)). Core ML takes an 8-bit RGB image and does the 1/255 scaling inside the model.
- **Outputs are raw feature maps,** not boxes: per-level class logits (`cls0`, `cls1`, `cls2`) and box distributions (`box0`, `box1`, `box2`), plus the mask, keypoint or angle maps for those tasks. The sigmoid, box decoding, score threshold, NMS, mask assembly and keypoint refinement run on the host. Mayaku's predictors do this for you.

### Running without Mayaku

`from_pretrained("model.onnx")` runs the graph in its runtime and the post-processing in Python, so the Mayaku package is still required. To run an exported model from another language or an embedded system, you must implement the post-processing yourself. The reference is `mayaku/inference/decode.py`. Everything it needs, including output names, strides, box-distribution bins (`reg_max`) and thresholds, is in the embedded metadata:

| Format | Where the metadata lives |
|---|---|
| ONNX | `metadata_props`, key `mayaku` (JSON string) |
| Core ML | `user_defined_metadata["mayaku"]` |
| OpenVINO | `rt_info`, key `mayaku` |
| TensorRT | A length-prefixed JSON header in front of the engine bytes: a 4-byte little-endian length, then the JSON, then the engine |

Read it from Python:

```python
import json, onnx

meta = json.loads(next(p.value for p in onnx.load("model.onnx").metadata_props if p.key == "mayaku"))
print(meta["class_names"], meta["canvas_hw"], meta["outputs"], meta["decode"])
```

A TensorRT `.engine` written by Mayaku is not a plain engine file. Strip the header before passing the bytes to `trt.Runtime.deserialize_cuda_engine`, with `mayaku.inference.export.metadata.strip_tensorrt_header(path)` or by skipping `4 + length` bytes.

## int8

Models are trained quantization-aware by default (`model.qat: true`, see [Training](/mayaku/data-and-training/training/#the-training-recipe)). Such a model has learned the value ranges of every layer, and int8 export writes exactly the network it was trained as. No calibration dataset is needed.

```bash
mayaku export onnx     best.pt --precision int8 --output model_int8.onnx
mayaku export openvino best.pt --precision int8 --output model_int8.xml
mayaku export tensorrt best.pt --precision int8 --output model_int8.engine
mayaku export coreml   best.pt --precision int8 --output model_int8.mlpackage
```

- A model trained with `model.qat: false` can't export int8. It fails with `int8 needs a quantization-aware model (train it with model.qat)`.
- During training, the log line `int8 vs float -0.3 AP` reports the expected accuracy cost. To measure it exactly, run `mayaku eval` on the int8 file.
- **TensorRT int8** needs symmetric activation ranges, the default (`model.qat_activations: symmetric`). Models trained with `affine`, and QAT checkpoints from before this setting existed, fail with `TensorRT runs int8 with symmetric activation ranges only`.
- **Core ML int8** needs iOS 17 / macOS 14 or newer. On Apple chips before the A17 Pro and M4 it runs, but is not faster than fp16.

## Target notes

### ONNX

- The default `fp32` graph is the portable choice, and OpenVINO and TensorRT exports are built from it.
- `from_pretrained("model.onnx")` uses onnxruntime's CUDA provider when `onnxruntime-gpu` is installed and a GPU is available, otherwise the CPU.

### Core ML

- `fp16`, the default, runs on the Apple Neural Engine. `fp32` runs on the GPU or CPU.
- `fp16` and `fp32` packages need macOS 13 / iOS 16 or newer.
- A `.mlpackage` is a directory. Copy it whole.
- `from_pretrained("model.mlpackage")` lets Core ML place the model on any compute unit. `device="cpu"` pins it to the CPU.

### OpenVINO

- `model.xml` and `model.bin` belong together. Keep them in one directory and pass the `.xml`.
- `fp16` stores weights at half precision. `fp32` and `int8` run at fp32 inference precision in Mayaku's runner.
- `from_pretrained("model.xml")` runs on the OpenVINO CPU plugin.

### TensorRT

- Build the engine **on the machine that will serve it.** An engine is tied to the GPU model and the TensorRT version it was built with.
- Building needs a CUDA GPU. Export fails with `building a TensorRT engine needs a CUDA GPU` otherwise.
- TensorRT 10 and 11 are supported. TensorRT 11 builds `fp16` and `int8` engines from an fp16 ONNX graph produced by NVIDIA ModelOpt, which `mayaku[tensorrt]` installs. If it is missing, export fails with `install NVIDIA ModelOpt: pip install 'nvidia-modelopt[onnx]'`.
- `from_pretrained("model.engine")` runs on the GPU and keeps outputs there, so post-processing also runs on the GPU.
