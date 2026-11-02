---
title: Dataset formats for training
summary: The COCO JSON fields each task reads, the class-folder layout for classification, and the dataset health check.
label: Datasets
keywords: mayaku, coco format, dataset, annotations, health check
---

Mayaku reads two dataset formats:

- **COCO JSON** for detection, instance segmentation, keypoints and oriented boxes. One JSON file plus one image directory per split.
- **Class folders** for image classification. One sub-folder of images per class, and no annotation file.

Mayaku reads the files you have directly. There is no conversion step and no cache to build.

## COCO JSON

A split is a pair: an annotation JSON file and the image directory its `file_name` entries are relative to.

```python
train(
    train_annotations="data/train/_annotations.coco.json",
    train_images="data/train",
    val_annotations="data/valid/_annotations.coco.json",
    val_images="data/valid",
    weights="mayaku-n-det",
)
```

The JSON file and the images do not have to share a directory. Each image path is `os.path.join(images, file_name)`.

### Fields Mayaku reads

```json
{
  "images": [
    {"id": 1, "file_name": "img_0001.jpg", "height": 480, "width": 640}
  ],
  "annotations": [
    {
      "id": 10,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100.0, 120.0, 50.0, 80.0],
      "iscrowd": 0,
      "segmentation": [[100, 120, 150, 120, 150, 200, 100, 200]],
      "keypoints": [110, 130, 2, 140, 130, 2, 125, 195, 1],
      "num_keypoints": 3
    }
  ],
  "categories": [
    {"id": 1, "name": "widget", "keypoints": ["left_top", "right_top", "bottom"]}
  ]
}
```

| Field | Required | Used for |
|---|---|---|
| `images[].id`, `file_name`, `height`, `width` | Always | Image lookup. `height` and `width` choose the input size before any image is read, so they must match the files. |
| `annotations[].image_id`, `category_id`, `bbox` | Always | Boxes. `bbox` is `[x, y, width, height]` in pixels, from the top-left corner. |
| `annotations[].iscrowd` | Optional | `1` marks a crowd region. It is not trained on. Missing means `0`. |
| `annotations[].segmentation` | Segmentation, oriented boxes | Polygons (`[[x1, y1, x2, y2, ...], ...]`) or an RLE dict. |
| `annotations[].id`, `keypoints`, `num_keypoints` | Keypoints | `keypoints` is `[x1, y1, v1, x2, y2, v2, ...]`. |
| `categories[].id`, `name` | Always | Classes. |
| `categories[].keypoints` | Keypoints, optional | Keypoint names. They give the left/right pairs that are swapped when an image is flipped. |

Mayaku ignores every other field (`area`, `info`, `licenses`, `skeleton`, `supercategory` and so on).

### Classes

- **Every entry in `categories` becomes a class**, even one with no annotations. Some export tools add an unused parent category (Roboflow, for example, often writes a category with id `0` named after the dataset). That category then counts as a class the model never sees an example of. It does no harm, but you can remove it from the JSON if you want an exact class count.
- Classes are ordered by category `id`, ascending, and numbered `0, 1, 2, ...` in that order. Category ids don't need to be contiguous: COCO's 80 classes use ids up to 90.
- A model's class index `k` is the `k`-th category by id. Its name is `predictor.class_names[k]`.
- The validation and test files must have the same number of categories as the training file. Evaluation fails with `the model has N classes and <file> has M` when they differ.

### What is skipped

- **Crowd annotations** (`iscrowd: 1`) are never trained on.
- **Degenerate boxes**, with a width or height of 1 pixel or less, are dropped. [`health_check`](/mayaku/data-and-training/datasets/#health-check) counts them.
- **Images without annotations are kept.** They train as background: images where the model should find nothing.
- **Unreadable images.** During training, a missing or corrupt image file is replaced by another random training image. During evaluation it stops with `eval image missing: <path>`.

Images are read with OpenCV, so any format OpenCV decodes works: JPEG, PNG, BMP, WebP, TIFF and others.

### Instance segmentation masks

Set by each annotation's `segmentation`:

- **Polygons**: a list of polygons, each a flat `[x1, y1, x2, y2, ...]` list in pixels. An object in several pieces has several polygons. A polygon with fewer than 3 points is ignored.
- **RLE**: a COCO run-length-encoded dict (`{"size": [h, w], "counts": ...}`). Mayaku converts it to polygons when loading.

An annotation without a segmentation still trains its box, but has no mask to learn from. See [Instance segmentation](/mayaku/more-tasks/segmentation/).

### Keypoints

- `keypoints` holds `3 × K` numbers: `x, y, v` for each of the `K` keypoints. `v` is the COCO visibility flag: `0` not labeled, `1` labeled but hidden, `2` labeled and visible.
- Every object in a dataset has the same `K`, and you set `K` in the model config (`model.keypoints.num`).
- An annotation with no `keypoints`, or with `num_keypoints: 0`, trains as a box with all keypoints unlabeled.
- **Left/right pairs.** A horizontal flip, used as training augmentation, mirrors the image, so a left keypoint must become the right one. Mayaku derives these pairs from the names in `categories[].keypoints`: `left_X` pairs with `right_X`. With no names and `K = 17`, it uses the COCO person skeleton. Otherwise no keypoints are swapped.

See [Keypoint detection](/mayaku/more-tasks/keypoints/).

### Oriented boxes

Oriented-box models learn from each object's polygon in `segmentation`. The training target is the smallest rotated rectangle around the polygon. An object without a polygon trains as its axis-aligned `bbox` at angle 0. See [Oriented boxes](/mayaku/more-tasks/oriented-boxes/).

## Class folders

```text
data/
├── train/
│   ├── cat/
│   │   ├── 0001.jpg
│   │   └── ...
│   └── dog/
│       └── ...
└── val/
    ├── cat/
    └── dog/
```

- Each sub-folder of the split directory is a class. Classes are the sub-folder names in sorted order, and the folder name is the class name.
- Images are files ending in `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif` or `.tiff`, in any case. Other files are ignored.
- Images go directly in their class folder. Nested sub-folders are not searched.
- The validation split may leave out classes, but it may not contain a class the training split lacks. That fails with `<dir> has classes the model does not know: [...]`.
- There are no annotation files. Pass only `train_images` and `val_images` (or `--images` and `--val-images`).

This is the layout `torchvision.datasets.ImageFolder` reads. See [Image classification](/mayaku/more-tasks/classification/).

## Health check

`mayaku.health_check` scans one COCO split the way training reads it and returns a plain dict:

```python
import mayaku

report = mayaku.health_check("data/train/_annotations.coco.json", "data/train")
```

```python
{
    "images": 1200,
    "boxes": 8450,
    "classes": 3,
    "canvas": [672, 896],               # the input size training would use (H, W)
    "boxes_per_image": 7.04,
    "object_size": {"small": 0.41, "medium": 0.47, "large": 0.12},
    "aspect_ratio": {"p10": 0.48, "median": 1.02, "p90": 2.1},   # box width / height
    "class_imbalance": 6.3,             # most / least common class, by images containing it
    "class_counts": {"bolt": 1100, "nut": 950, "washer": 175},   # images containing each class
    "warnings": ["12 images with no annotations"],
}
```

- `object_size` sorts boxes into the COCO buckets (small: under 32×32 pixels; large: 96×96 and over), measured **after** resizing to `canvas`, which is the scale the model sees.
- `canvas` is the input size [auto-config](/mayaku/data-and-training/training/#auto-config) picks for this data at the default size budget of 800. To check another budget, pass a config:

  ```python
  import mayaku
  from mayaku.config import MayakuConfig

  report = mayaku.health_check("data/train/_annotations.coco.json", "data/train",
                               cfg=MayakuConfig(input={"size_budget": 640}))
  ```
- `aspect_ratio` is `None` with fewer than two boxes.
- `warnings` lists only facts: degenerate boxes that were dropped and images with no annotations. Neither is necessarily wrong.

The health check reads boxes only. It does not inspect masks or keypoints, and it does not read class-folder datasets.
