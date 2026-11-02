---
title: Quickstart: train, predict and export a detector
summary: Fine-tune a detector on your own COCO dataset, then predict, evaluate and export it, from the command line or Python.
label: Quickstart
keywords: mayaku, quickstart, object detection, fine-tuning, coco
---

This page fine-tunes an object detector on your own images, then predicts with it, evaluates it and exports it. Every step is shown both as a command and in Python.

You need:

- Mayaku installed (`pip install mayaku`, see [Installation](/mayaku/installation/)).
- A dataset in COCO format with a training split and a validation split.

## 1. Lay out the data

Mayaku reads COCO JSON annotations directly. A typical layout, the one Roboflow's "COCO" export produces:

```text
data/
├── train/
│   ├── _annotations.coco.json
│   ├── img_0001.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   └── ...
└── test/            (optional)
    ├── _annotations.coco.json
    └── ...
```

Each `file_name` in the JSON is resolved relative to the image directory you pass. Any layout works as long as that holds. The required JSON fields are listed in [Datasets](/mayaku/data-and-training/datasets/).

## 2. Check the data (optional)

```python
import mayaku

report = mayaku.health_check("data/train/_annotations.coco.json", "data/train")
print(report)
```

The report gives image, box and class counts, the input size training would use, object-size fractions, class balance, and warnings such as images without annotations. See [Datasets](/mayaku/data-and-training/datasets/#health-check).

## 3. Train

Command line:

```bash
mayaku train --weights mayaku-n-det \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

Python:

```python
from mayaku import train

result = train(
    weights="mayaku-n-det",
    train_annotations="data/train/_annotations.coco.json",
    train_images="data/train",
    val_annotations="data/valid/_annotations.coco.json",
    val_images="data/valid",
)
print(result["final_weights"])   # runs/mayaku-n-det/train/best.pt
print(result["metrics"]["AP"])   # COCO AP@[.50:.95] on the validation split
```

What happens:

- `mayaku-n-det` is the nano detector pretrained by Datamarkin. It downloads to `./mayaku-n-det.pth` the first time, and later runs reuse that file.
- The class count and the input size (the "canvas") come from your training annotations. The class layer is re-initialized for your classes.
- Training runs 100 epochs, the fine-tune default for datasets of 10 images or more, and evaluates on the validation split after every epoch.
- The checkpoint with the best validation AP is saved as `best.pt` and scored once more at the end.

For a 300-image dataset of 4:3 photos with 3 classes, the log starts like this:

```text
[mayaku.train] auto-config input.canvas_hw: None -> [672, 896]
[mayaku.train] auto-config model.num_classes: None -> 3
[mayaku.train] auto-config train.assigner_warmup: 5 -> 0
[mayaku.train] auto-config train.aug.mosaic: 1.0 -> 0.1
[mayaku.train] auto-config train.epochs: 125 -> 100
[mayaku.train] auto-config train.final_frac: 0.16 -> 0.2
[mayaku.train] tier n, 3 classes, canvas 672x896, 100 epochs, batch 16 on cuda -> runs/mayaku-n-det/train
```

Each `auto-config` line is a setting Mayaku derived from your data because you didn't set it yourself. See [Training](/mayaku/data-and-training/training/#auto-config).

When training finishes, the command prints `final weights: runs/mayaku-n-det/train/best.pt`.

To pick a size other than nano, use `mayaku-s-det`, `mayaku-m-det` or `mayaku-l-det`. See [Models](/mayaku/data-and-training/models/). For every training option, see [Training](/mayaku/data-and-training/training/).

## 4. Predict

Command line (prints JSON):

```bash
mayaku predict runs/mayaku-n-det/train/best.pt photo.jpg
```

Python:

```python
from mayaku import from_pretrained

predictor = from_pretrained("runs/mayaku-n-det/train/best.pt")
det = predictor("photo.jpg")

for box, score, label in zip(det.boxes.tolist(), det.scores.tolist(), det.labels.tolist()):
    print(predictor.class_names[label], round(score, 3), box)   # box is [x1, y1, x2, y2] in pixels
```

The checkpoint holds the class names, input size and thresholds, so no config file is needed. See [Prediction](/mayaku/deployment/prediction/).

## 5. Evaluate on the test split

```bash
mayaku eval runs/mayaku-n-det/train/best.pt \
  --annotations data/test/_annotations.coco.json --images data/test
```

```python
from mayaku import evaluate

metrics = evaluate("runs/mayaku-n-det/train/best.pt",
                   annotations="data/test/_annotations.coco.json", images="data/test")
print(metrics["AP"], metrics["AP50"])
```

See [Evaluation](/mayaku/deployment/evaluation/) for every metric key.

## 6. Export

```bash
pip install "mayaku[onnx]" onnxruntime
mayaku export onnx runs/mayaku-n-det/train/best.pt --output model.onnx
```

The exported file runs through the same `from_pretrained` and `mayaku predict`:

```bash
mayaku predict model.onnx photo.jpg
```

Core ML, OpenVINO and TensorRT work the same way. See [Export and deployment](/mayaku/deployment/export/).
