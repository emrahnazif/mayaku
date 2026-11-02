---
title: Train an instance segmentation model
summary: Train a model that returns a pixel mask per object, and read its masks and mask metrics.
label: Segmentation
keywords: mayaku, instance segmentation, masks, coco, rle
---

An instance segmentation model detects each object, giving a box, class and score, and also returns a pixel mask for it. In Mayaku it is a detector with a mask branch turned on (`model.seg: true`). Everything in [Training](/mayaku/data-and-training/training/) applies.

## Data

COCO JSON whose annotations have a `segmentation` field, either polygons or RLE:

```json
{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 120, 50, 80], "iscrowd": 0,
 "segmentation": [[100, 120, 150, 120, 150, 200, 100, 200]]}
```

See [Datasets](/mayaku/data-and-training/datasets/#instance-segmentation-masks) for the details.

## Train

Start from the pretrained segmentation model:

```bash
mayaku train --weights mayaku-n-seg \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

```python
from mayaku import train

result = train(
    weights="mayaku-n-seg",
    train_annotations="data/train/_annotations.coco.json",
    train_images="data/train",
    val_annotations="data/valid/_annotations.coco.json",
    val_images="data/valid",
)
```

The `-seg` checkpoint already has the mask branch, so you don't need to set anything.

You can also add a mask branch to a detector's weights. The box part starts pretrained and the mask branch starts fresh:

```bash
mayaku train --weights mayaku-n-det --set model.seg=true \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

From scratch, with no weights: `--set model.seg=true`.

### Mask settings

| Field | Default | Meaning |
|---|---|---|
| `model.seg` | `false` | Turns the mask branch on. |
| `model.mask_stride` | `8` | Resolution of the mask features: `8` (1/8 of the input) or `4` (1/4, finer edges, more memory and compute). It changes the network, so it is fixed once trained. |
| `train.seg_gain` | `2.0` | Weight of the mask (Dice) loss. |
| `train.seg_cap` | `250` | Maximum number of positive samples per batch that train the mask loss, to bound memory. |

## Validation and metrics

Training scores both boxes and masks. The final `metrics` dict and `eval/metrics.json` hold the box metrics (`AP`, `AP50`, ...) and the mask metrics with a `segm_` prefix (`segm_AP`, `segm_AP50`, `segm_AP75`, `segm_AP-S`, `segm_AP-M`, `segm_AP-L`, `segm_AR1`, ...). See [Evaluation](/mayaku/deployment/evaluation/).

`best.pt` is chosen by **box** AP. Per-epoch mask AP is in `train/log.jsonl`.

## Predict

```python
from mayaku import from_pretrained

predictor = from_pretrained("runs/mayaku-n-seg/train/best.pt")
det = predictor("photo.jpg")

det.masks          # torch.bool tensor (n, H, W): one full-resolution mask per detection
det.masks[0].sum() # pixel area of the first object's mask
```

- `det.masks[i]` belongs to `det.boxes[i]`, `det.scores[i]` and `det.labels[i]`.
- Masks have the original image's height and width, not the canvas size.
- Full-resolution masks of many detections use a lot of memory on large images. Raise the score threshold to keep fewer, for example `predictor("photo.jpg", conf=0.25)`.

Save a mask as an image:

```python
import cv2

cv2.imwrite("mask0.png", det.masks[0].numpy().astype("uint8") * 255)
```

### Command-line output

`mayaku predict` writes each mask as COCO RLE under `segmentation`:

```json
{
  "class_id": 0,
  "class": "widget",
  "score": 0.9132,
  "box_xyxy": [101.3, 119.8, 151.0, 200.4],
  "segmentation": {"size": [480, 640], "counts": "Xk`35n>..."}
}
```

`size` is `[height, width]`. Decode it with pycocotools, which is installed with Mayaku:

```python
from pycocotools import mask as mask_utils

binary = mask_utils.decode(detection["segmentation"])   # uint8 array (height, width), 1 inside
```

## Export

Segmentation models export to every target like a detector. See [Export and deployment](/mayaku/deployment/export/). The exported graph outputs the mask features, and Mayaku's host-side decode assembles each object's mask. Running the artifact with `from_pretrained("model.onnx")` returns the same `masks`.
