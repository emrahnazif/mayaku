---
title: Train a keypoint detection model
summary: Train a model that locates K keypoints per object, with correct left/right flips, and read its keypoints.
label: Keypoints
keywords: mayaku, keypoint detection, pose estimation, coco keypoints, oks
---

A keypoint model detects each object, giving a box, class and score, and also locates a fixed set of `K` points on it: the joints of a person, the corners of a part, the landmarks of an animal. In Mayaku it is a detector with a keypoint branch turned on (`model.keypoints`). Everything in [Training](/mayaku/data-and-training/training/) applies.

## Data

COCO JSON whose annotations have `keypoints`, and optionally keypoint names on the categories:

```json
{
  "annotations": [
    {"id": 10, "image_id": 1, "category_id": 1, "bbox": [100, 120, 50, 80], "iscrowd": 0,
     "keypoints": [110, 130, 2, 140, 130, 2, 125, 195, 1], "num_keypoints": 3}
  ],
  "categories": [
    {"id": 1, "name": "widget", "keypoints": ["left_top", "right_top", "bottom"]}
  ]
}
```

- `keypoints` is `x, y, v` for each of the `K` points. `v`: `0` not labeled, `1` labeled but hidden, `2` labeled and visible.
- Every annotation has the same `K`.
- Annotations need an `id`, because keypoints are matched to boxes by annotation id.

### Name your keypoints `left_*` / `right_*`

Training flips images horizontally, so a point on an object's left must then be treated as its right. Mayaku pairs `left_X` with `right_X` using the names in `categories[].keypoints`. With names like `left_eye` and `right_eye`, flipping is handled correctly. Without names, a 17-point dataset is assumed to follow the COCO person skeleton, and any other count gets no pairs. If your points have a left/right meaning but different names, rename them, or turn flips off with `--set train.aug.hflip=0`.

Don't use vertical flips (`train.aug.vflip`) with keypoints. They would mislabel the pairs.

## Train

Set the number of keypoints with `model.keypoints.num`, and start from the detector weights:

```bash
mayaku train --weights mayaku-n-det --set model.keypoints.num=3 \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

```python
from mayaku import train

result = train(
    weights="mayaku-n-det",
    overrides={"model": {"keypoints": {"num": 3}}},
    train_annotations="data/train/_annotations.coco.json",
    train_images="data/train",
    val_annotations="data/valid/_annotations.coco.json",
    val_images="data/valid",
)
```

The box part starts pretrained and the keypoint branch starts fresh. Keypoint names and left/right pairs are filled in from your categories ([auto-config](/mayaku/data-and-training/training/#auto-config)) and saved in the checkpoint.

No pretrained keypoint model is hosted: keypoint detection is experimental. Start from the detector weights (`mayaku-n-det`), as shown above. A checkpoint you trained yourself continues only with the same keypoint count; a different `model.keypoints.num` fails with `state dict does not fit this model: ['aux.heat.weight', ...]`.

## Validation and metrics

The final `metrics` dict holds the box metrics (`AP`, `AP50`, ...) and the keypoint metrics with a `kpt_` prefix: `kpt_AP`, `kpt_AP50`, `kpt_AP75`, `kpt_AP-M`, `kpt_AP-L`, `kpt_AR`, `kpt_AR50`, `kpt_AR75`, `kpt_AR-M`, `kpt_AR-L`. COCO keypoint evaluation has no small-object bucket.

Keypoint AP uses COCO's OKS (object keypoint similarity). The per-keypoint tolerances are built into the model, not taken from your data.

`best.pt` is chosen by **keypoint** AP (`kpt_AP`), which each epoch's log line shows after the box metrics. With QAT, an epoch counts as the lower of its fp32 and int8 keypoint AP.

## Predict

```python
from mayaku import from_pretrained

predictor = from_pretrained("runs/mayaku-n-det/train/best.pt")
det = predictor("photo.jpg")

det.keypoints        # float tensor (n, K, 3): x, y, visibility probability
names = predictor.sidecar["keypoints"]["names"]

for obj in det.keypoints:
    for name, (x, y, p) in zip(names, obj.tolist()):
        if p > 0.5:
            print(name, round(x), round(y))
```

- `det.keypoints[i]` belongs to `det.boxes[i]`.
- `x` and `y` are in the original image's pixels.
- The third value is the model's probability, from 0 to 1, that the point is visible. It is not the COCO `v` flag. Choose a threshold that suits you, for example 0.5.

`mayaku predict` writes `"keypoints": [[x, y, p], ...]`, one triple per point, in each detection.

## Export

Keypoint models export to every target like a detector. See [Export and deployment](/mayaku/deployment/export/).
