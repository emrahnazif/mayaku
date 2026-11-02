---
title: Train an oriented bounding box model
summary: Train a model that detects rotated boxes from polygon annotations, and read its angles and corners.
label: Oriented boxes
keywords: mayaku, oriented bounding boxes, obb, rotated boxes, aerial images
---

An oriented-box model detects each object as a **rotated** rectangle: center, width, height and angle. This fits objects that are not axis-aligned, such as ships and vehicles in aerial images, text lines, or parts on a conveyor. In Mayaku it is a detector with an angle output turned on (`model.obb: true`). Everything in [Training](/mayaku/data-and-training/training/) applies.

Oriented boxes are a task of their own. They can't be combined with masks (`model.seg`) or keypoints (`model.keypoints`) in one model.

## Data

COCO JSON with a polygon in each annotation's `segmentation`:

```json
{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 80, 60], "iscrowd": 0,
 "segmentation": [[120, 100, 180, 130, 160, 160, 100, 130]]}
```

- The training target for each object is the **smallest rotated rectangle that encloses its polygon**. A four-point polygon traced around a rotated object gives exactly that rectangle. Any other polygon, including a full instance outline, gives its tightest rotated box.
- RLE segmentations work too. They are converted to polygons when loading.
- An annotation without a segmentation trains as its axis-aligned `bbox` at angle 0.

This is the same COCO file a segmentation dataset uses, so one labeled dataset can train both kinds of model.

## Train

No pretrained oriented-box models are hosted. Start from the detector weights of the size you want and turn on the angle output:

```bash
mayaku train --weights mayaku-n-det --set model.obb=true \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

```python
from mayaku import train

result = train(
    weights="mayaku-n-det",
    overrides={"model": {"obb": True}},
    train_annotations="data/train/_annotations.coco.json",
    train_images="data/train",
    val_annotations="data/valid/_annotations.coco.json",
    val_images="data/valid",
)
```

The backbone and box head start pretrained and the angle output starts fresh. A checkpoint trained this way is itself a valid `weights` for further oriented-box training.

For aerial and other top-down images, vertical flips are usually a free gain: `--set train.aug.vflip=0.5`.

## Validation and metrics

The metric keys are the detection keys (`AP`, `AP50`, `AP75`, `AP-S`, ..., see [Evaluation](/mayaku/deployment/evaluation/)), but overlap is measured between **rotated** boxes. Each ground-truth box is the rotated rectangle derived from its polygon, and it is matched to predictions by exact polygon IoU.

## Predict

```python
from mayaku import from_pretrained

predictor = from_pretrained("runs/mayaku-n-det/train/best.pt")
det = predictor("aerial.jpg")

det.rboxes   # float tensor (n, 5): cx, cy, w, h, theta
det.boxes    # float tensor (n, 4): the axis-aligned box around each rotated box
```

- `rboxes` are in the original image's pixels, with `theta` in **radians**. `theta` turns the box's width axis from the image's x axis toward its y axis. Image y points down, so a positive angle turns the box clockwise on screen. Values lie in `[-π/4, 3π/4)`.
- `rboxes` are not clipped to the image, so a box partly outside the frame keeps its full size. `boxes` are clipped to the image.
- NMS (removing duplicate detections) uses the overlap of the rotated boxes.

Get the four corners and draw them:

```python
import cv2
from mayaku.model.obb import corners

pts = corners(det.rboxes)          # (n, 4, 2) corner points, in order around each box
img = cv2.imread("aerial.jpg")
for quad in pts.round().int().numpy():
    cv2.polylines(img, [quad.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.imwrite("aerial_obb.jpg", img)
```

`mayaku predict` writes both forms in each detection:

```json
{
  "class_id": 0,
  "class": "ship",
  "score": 0.8812,
  "box_xyxy": [102.65, 100.08, 177.59, 160.02],
  "rbox_xywhr": [140.12, 130.05, 67.03, 33.51, 0.4636],
  "corners": [[117.63, 100.08], [177.59, 130.05], [162.61, 160.02], [102.65, 130.05]]
}
```

## Export

Oriented-box models export to every target like a detector. See [Export and deployment](/mayaku/deployment/export/). The rotated NMS runs in Mayaku's host-side decode, so `from_pretrained("model.onnx")` returns the same `rboxes`.
