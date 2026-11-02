---
title: Mayaku
summary: Train detection, segmentation, keypoint, oriented-box and classification models on your own images, and export them to ONNX, Core ML, OpenVINO and TensorRT.
label: Mayaku
keywords: mayaku, object detection, instance segmentation, keypoint detection, computer vision, pytorch
template: docs
---

Mayaku is a Python library for training computer vision models on your own images and deploying them. One `pip install` trains on NVIDIA GPUs (CUDA), Apple Silicon (MPS) or the CPU, and exports to ONNX, Core ML, OpenVINO and TensorRT.

It supports five tasks:

| Task | What the model returns | Training data |
|---|---|---|
| Object detection | A box, class and score per object | COCO JSON with boxes |
| Instance segmentation | Detection plus a pixel mask per object | COCO JSON with polygons or RLE masks |
| Keypoint detection | Detection plus K keypoints per object | COCO JSON with keypoints |
| Oriented boxes | Detection plus a rotated box per object | COCO JSON with polygons |
| Image classification | Class probabilities per image | One folder per class |

Models come in four sizes, `n`, `s`, `m` and `l` (nano, small, medium, large). Pretrained base models are hosted for detection, segmentation and keypoints. Pass the name, for example `mayaku-n-det`, and Mayaku downloads the weights the first time.

## Minimal example

```python
from mayaku import train, from_pretrained

result = train(
    weights="mayaku-n-det",
    train_annotations="data/train/_annotations.coco.json",
    train_images="data/train",
    val_annotations="data/valid/_annotations.coco.json",
    val_images="data/valid",
)

predictor = from_pretrained(result["final_weights"])
detections = predictor("photo.jpg")
print(detections.boxes, detections.scores, detections.labels)
```

## Pages

First steps:

- [Installation](/mayaku/installation/): `pip install`, optional export packages, supported platforms.
- [Quickstart](/mayaku/quickstart/): train, predict, evaluate and export a detector in five commands.

Data and training:

- [Datasets](/mayaku/data-and-training/datasets/): the COCO JSON layout each task needs, and the class-folder layout for classification.
- [Models](/mayaku/data-and-training/models/): the hosted model names, the four sizes, and how weights are downloaded.
- [Training](/mayaku/data-and-training/training/): `mayaku.train`, the run directory, auto-config, resuming, multiple GPUs.

Deployment:

- [Evaluation](/mayaku/deployment/evaluation/): COCO metrics and top-1/top-5 accuracy, and what each metric key means.
- [Export](/mayaku/deployment/export/): ONNX, Core ML, OpenVINO, TensorRT, precisions and int8.
- [Prediction](/mayaku/deployment/prediction/): `from_pretrained`, the `Detections` object, batches, the JSON output of `mayaku predict`.

More tasks:

- [Image classification](/mayaku/more-tasks/classification/)
- [Keypoint detection](/mayaku/more-tasks/keypoints/)
- [Oriented boxes](/mayaku/more-tasks/oriented-boxes/)
- [Instance segmentation](/mayaku/more-tasks/segmentation/)

Reference:

- [Command line](/mayaku/reference/cli/): every `mayaku` command and option.
- [Configuration](/mayaku/reference/configuration/): every config field, its default, and how to set it.
- [Python API](/mayaku/reference/python-api/): every public function and class.
- [Troubleshooting](/mayaku/reference/troubleshooting/): error messages and what to do about them.

## Key facts

- Package: `mayaku` on PyPI. Python 3.10, 3.11 or 3.12. Linux and macOS.
- License: Apache 2.0 for the library and every published model weight.
- Public Python entry points: `mayaku.train`, `mayaku.evaluate`, `mayaku.from_pretrained`, `mayaku.health_check`.
- Command line: `mayaku train`, `mayaku eval`, `mayaku predict`, `mayaku export`, `mayaku download`.
- Every checkpoint and every exported file carries its own configuration and class names. You never need a separate config file to run a trained model.
- Source: <https://github.com/datamarkin/mayaku>
