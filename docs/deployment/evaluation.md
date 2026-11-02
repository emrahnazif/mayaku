---
title: Evaluate a model: COCO metrics and accuracy
summary: Score a checkpoint or exported file on a labeled split, and what every metric key means.
label: Evaluation
keywords: mayaku, evaluation, coco metrics, mean average precision, pycocotools
---

`mayaku.evaluate` (command line: `mayaku eval`) scores a trained model on a labeled split. Detectors get COCO metrics computed by pycocotools. Classifiers get top-1 and top-5 accuracy.

```python
from mayaku import evaluate

metrics = evaluate("runs/mayaku-n-det/train/best.pt",
                   annotations="data/test/_annotations.coco.json", images="data/test")
print(metrics["AP"], metrics["AP50"])
```

```bash
mayaku eval runs/mayaku-n-det/train/best.pt \
  --annotations data/test/_annotations.coco.json --images data/test --output results/
```

## What can be evaluated

The first argument is anything [`from_pretrained`](/mayaku/deployment/prediction/) loads: a checkpoint, a hosted model name, or an **exported file** (`.onnx`, `.mlpackage`, `.xml`, `.engine`). An exported file is scored in its own runtime and precision, so you measure exactly what you deploy. Comparing a checkpoint against its int8 export, for example, shows the real accuracy cost of quantization:

```bash
mayaku eval best.pt     --annotations data/test/_annotations.coco.json --images data/test
mayaku eval model.onnx  --annotations data/test/_annotations.coco.json --images data/test
```

The model is scored with its recorded post-processing, which by default uses a score threshold of 0.001 (see [Prediction](/mayaku/deployment/prediction/#the-score-threshold)). This is the standard way to compute COCO AP.

## Arguments

| Argument | CLI | Meaning |
|---|---|---|
| `weights` | positional `WEIGHTS` | The model to score. |
| `annotations` | `--annotations` | COCO JSON of the split. Required for detectors. Leave it out for classifiers. |
| `images` | `--images` | Image directory, or the class-folder root for a classifier. |
| `output_dir` | `--output` | If given, `metrics.json` is written there. |
| `device` | `--device` | `auto`, `cuda`, `mps` or `cpu`. |
| `log` | none | Function that receives progress lines. Default `print`. |

The split must have the same number of categories as the model has classes. Otherwise evaluation stops with `the model has N classes and <file> has M`.

`evaluate` returns the metrics as a flat dict. The command line prints them as JSON.

## Detection metrics

Every detector, oriented-box model, segmentation model and keypoint model reports these box metrics. Values are fractions from 0 to 1.

| Key | Meaning |
|---|---|
| `AP` | Average precision over IoU thresholds 0.50 to 0.95 in steps of 0.05 (COCO AP@[.50:.95]). The main number. |
| `AP50` | AP at IoU 0.50 (the PASCAL VOC metric) |
| `AP75` | AP at IoU 0.75 (strict localization) |
| `AP-S` | AP on small objects: area under 32×32 pixels |
| `AP-M` | AP on medium objects: 32×32 to 96×96 pixels |
| `AP-L` | AP on large objects: over 96×96 pixels |
| `AR1`, `AR10`, `AR100` | Average recall with at most 1, 10 or 100 detections per image |
| `AR-S`, `AR-M`, `AR-L` | Average recall (100 detections) by object size |

**`-1.0` means "not applicable"**: the split has no ground-truth objects of that size. It does not mean the model scored below zero.

Object sizes come from the annotation's `area` field when it has one, otherwise from the box. They are measured in the original image's pixels, as pycocotools does.

For [oriented-box models](/mayaku/more-tasks/oriented-boxes/), the same keys are computed with the IoU between **rotated** boxes.

### Segmentation metrics

[Segmentation models](/mayaku/more-tasks/segmentation/) also report the same twelve metrics computed on masks, prefixed `segm_`: `segm_AP`, `segm_AP50`, `segm_AP75`, `segm_AP-S`, `segm_AP-M`, `segm_AP-L`, `segm_AR1`, `segm_AR10`, `segm_AR100`, `segm_AR-S`, `segm_AR-M`, `segm_AR-L`.

### Keypoint metrics

[Keypoint models](/mayaku/more-tasks/keypoints/) also report COCO keypoint metrics, which use OKS (object keypoint similarity) in place of IoU, prefixed `kpt_`:

| Key | Meaning |
|---|---|
| `kpt_AP` | AP over OKS thresholds 0.50 to 0.95 |
| `kpt_AP50`, `kpt_AP75` | AP at OKS 0.50 and 0.75 |
| `kpt_AP-M`, `kpt_AP-L` | AP on medium and large objects. COCO keypoint evaluation has no small bucket. |
| `kpt_AR`, `kpt_AR50`, `kpt_AR75`, `kpt_AR-M`, `kpt_AR-L` | The matching recall values |

In a file with several categories, keypoint AP is computed only over the categories whose `keypoints` name list has `K` entries. If no category lists names, all categories are scored.

## Classification metrics

For a [classifier](/mayaku/more-tasks/classification/), pass the class-folder root as `images` with no annotations:

```bash
mayaku eval runs/cls/train/best.pt --images data/test
```

| Key | Meaning |
|---|---|
| `top1` | Fraction of images whose most likely class is correct |
| `top5` | Fraction of images whose correct class is among the five most likely. With five classes or fewer, this is always 1.0. |

## Evaluation during training

When you pass a validation split to `train`, the model is evaluated after every epoch with the same metrics. Results go to `train/log.jsonl`, the best epoch to `train/best.json`, and the final evaluation of `best.pt` to `eval/metrics.json`. See [Training](/mayaku/data-and-training/training/#validation).
