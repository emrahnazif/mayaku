---
title: Train an image classifier
summary: Train an image classifier from one folder per class, then predict and score top-1 and top-5 accuracy.
label: Classification
keywords: mayaku, image classification, class folders, top-1 accuracy
---

A classification model assigns one class to a whole image and returns a probability for every class. In Mayaku it is the detector's backbone with a classification head (`model.task: classify`). It is trained, predicted, evaluated and exported with the same functions and commands as the detectors.

## Data

One folder per class, with no annotation files:

```text
data/
├── train/
│   ├── cat/  0001.jpg, 0002.jpg, ...
│   └── dog/  ...
└── val/
    ├── cat/
    └── dog/
```

Class names are the folder names, in sorted order. See [Datasets](/mayaku/data-and-training/datasets/#class-folders) for the file types and rules.

## Train

Set `model.task=classify` and pass only image directories:

```bash
mayaku train --weights mayaku-n-det --set model.task=classify --epochs 30 \
  --images data/train --val-images data/val
```

```python
from mayaku import train

result = train(
    weights="mayaku-n-det",
    overrides={"model": {"task": "classify"}},
    num_epochs=30,
    train_images="data/train",
    val_images="data/val",
)
```

- **Weights.** Any Mayaku checkpoint of the same size can start a classifier. From a detector such as `mayaku-n-det`, the pretrained backbone is loaded and the classification head starts fresh (the log says `backbone warm-started, head fresh`). From a classifier checkpoint with the same number of classes, the head is loaded too. Without `weights`, training starts from scratch.
- **Epochs.** Auto-config does not shorten a classifier's schedule. Without `--epochs` it trains the recipe's 125 epochs. Fine-tuning usually needs far fewer, so set `--epochs`.
- **Input size.** The default canvas is 224×224. `--size-budget 320` trains at 320×320. Classifier canvases are always square unless you set `input.canvas_hw`.
- **Validation** is optional. Without `--val-images`, there is no `best.pt` and `last.pt` is the result.
- The detection settings (`model.seg`, `model.keypoints`, `model.obb`) don't apply and are turned off.

### Augmentation

Training images get a random crop covering 25 to 100% of the image area at an aspect ratio between 3:4 and 4:3, a horizontal flip (`train.aug.hflip`, default 0.5), and color jitter (`train.aug.hsv`). `train.aug.vflip` adds vertical flips. Mosaic and mixup are not used for classification. In the final stage (the last `train.final_frac` of epochs), the crop is turned off and the whole image is used, as at prediction time.

## Validation and metrics

Classifiers are scored by accuracy:

| Key | Meaning |
|---|---|
| `top1` | Fraction of images whose most likely class is correct |
| `top5` | Fraction of images whose correct class is among the five most likely |

`best.pt` is the epoch with the highest `top1`. Each epoch logs a line like:

```text
epoch  12  cls 0.214  top1 0.9350  top5 1.0000  int8 vs float -0.2 top1
```

Evaluate a trained classifier on a class-folder split with images only, without annotations:

```bash
mayaku eval runs/mayaku-n-det/train/best.pt --images data/test
```

## Predict

```python
from mayaku import from_pretrained

predictor = from_pretrained("runs/mayaku-n-det/train/best.pt")
result = predictor("photo.jpg")      # a Classification

result.label                         # index of the most likely class
predictor.class_names[result.label]  # its name
result.scores                        # float tensor (num_classes,): probabilities, summing to 1
result.top(3)                        # [(class_index, probability), ...], best first
```

Several images at once: `predictor.batch(["a.jpg", "b.jpg"])` returns one `Classification` per image.

`mayaku predict` prints the five most likely classes:

```json
{
  "image": "photo.jpg",
  "classes": [
    {"class_id": 0, "class": "cat", "score": 0.9731},
    {"class_id": 1, "class": "dog", "score": 0.0269}
  ]
}
```

## How prediction works

The image is letterboxed onto the canvas: resized to fit while keeping its aspect ratio, then padded with gray. Prediction never crops, so objects near the edges are kept. This is the same preprocessing the detectors and every exported runtime use.

## Export

Classifiers export to every target. See [Export and deployment](/mayaku/deployment/export/). An exported classifier runs with `from_pretrained("model.onnx")` and returns the same `Classification`.
