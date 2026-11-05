---
title: Train a model with mayaku.train
summary: Warm starts, settings, input size, auto-config, the run directory, resuming and multi-GPU training with mayaku.train.
label: Training
keywords: mayaku, training, fine-tuning, auto-config, multi-gpu
---

`mayaku.train` (command line: `mayaku train`) trains every model type: detection, instance segmentation, keypoints, oriented boxes and classification. This page covers what all of them share. The task pages cover what differs: [segmentation](/mayaku/more-tasks/segmentation/), [keypoints](/mayaku/more-tasks/keypoints/), [oriented boxes](/mayaku/more-tasks/oriented-boxes/), [classification](/mayaku/more-tasks/classification/).

## The basic call

```python
from mayaku import train

result = train(
    weights="mayaku-n-det",
    train_annotations="data/train/_annotations.coco.json",
    train_images="data/train",
    val_annotations="data/valid/_annotations.coco.json",
    val_images="data/valid",
)
```

```bash
mayaku train --weights mayaku-n-det \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

## Where the model comes from

Training starts from pretrained weights, from a config, or from both:

| You pass | The architecture comes from | The starting weights |
|---|---|---|
| `weights` only | The checkpoint: its size and heads | The checkpoint (fine-tuning) |
| `config` only | The config | Random (training from scratch) |
| `config` and `weights` | The config | The checkpoint, which must fit the config |
| Neither | The defaults: size `n`, plain detector | Random (training from scratch) |

`weights` is a hosted model name such as `mayaku-n-det` (see [Models](/mayaku/data-and-training/models/)) or the path to any `.pt` or `.pth` checkpoint Mayaku wrote, including the `best.pt` of an earlier run.

**Fine-tuning from pretrained weights is the recommended default.** It converges in fewer epochs and reaches higher accuracy on small datasets than training from scratch.

### What a warm start can and cannot change

When you start from a checkpoint, the weights are loaded into the model your config describes:

- **Class count.** It may differ from the checkpoint's. The class layer is re-initialized for your classes, and the log says `classifier re-initialised for N classes`.
- **Heads.** Masks, keypoints and oriented boxes may be added to or removed from a detector's weights. For example, `mayaku-n-det` with `--set model.seg=true` trains a segmentation model whose backbone and box head start pretrained and whose mask branch starts fresh.
- **Size.** It cannot change. `mayaku-n-det` loads only into an `n` model. Setting `model.tier=s` with `n` weights fails with `state dict does not fit this model`.
- **Keypoint count.** It cannot change from the checkpoint's. To train a different keypoint count, start from the detector weights (`mayaku-n-det`). See [Keypoint detection](/mayaku/more-tasks/keypoints/).

### Training from scratch

Leave out `weights`:

```bash
mayaku train --set model.tier=s \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

A from-scratch run uses the full recipe: 125 epochs, with a longer warmup for the label assigner. Auto-config still sets the class count and input size, but it does not shorten the schedule.

## Changing settings

Every setting is a field of the config (see [Configuration](/mayaku/reference/configuration/) for all of them). There are three ways to set a field, and they combine.

**Command line**, repeatable `--set dotted.path=value` (the value is parsed as YAML, so `true`, `0.005`, `null` and `[640, 640]` get their types):

```bash
mayaku train --weights mayaku-n-det --set train.lr=0.005 --set train.batch=32 \
  --annotations data/train/_annotations.coco.json --images data/train
```

**Python**, with a nested `overrides` mapping:

```python
train(weights="mayaku-n-det", overrides={"train": {"lr": 0.005, "batch": 32}},
      train_annotations="data/train/_annotations.coco.json", train_images="data/train")
```

**A YAML file**, passed as the first argument (`config=` in Python):

```yaml
# my_run.yaml
model:
  tier: m
  seg: true
train:
  epochs: 150
  batch: 32
```

```bash
mayaku train my_run.yaml --weights mayaku-m-det \
  --annotations data/train/_annotations.coco.json --images data/train
```

Unknown keys are rejected at any level, so a typo fails immediately instead of being ignored.

Two shortcuts set common fields: `num_epochs` / `--epochs` sets `train.epochs`, and `size_budget` / `--size-budget` sets `input.size_budget`.

## Input size

Every image is resized, preserving its aspect ratio, and padded onto one fixed rectangle called the **canvas** (letterboxing). The same canvas is used for training, evaluation, export and deployment.

- `input.size_budget` (default `800`) is the compute dial. The canvas never has more than `size_budget²` pixels.
- The canvas shape follows your data. If nearly all training images share one aspect ratio, the canvas takes that ratio, so a 16:9 dataset at budget 800 trains on a 576×1024 canvas instead of 800×800 with padding bars. Mixed aspect ratios get a square canvas.
- Both sides are always multiples of 32.

| Size budget | 1:1 data | 4:3 data | 16:9 data |
|---|---|---|---|
| 640 | 640×640 | 544×736 | 480×832 |
| 800 (default) | 800×800 | 672×896 | 576×1024 |

(Height × width.)

A higher budget helps small objects and costs time and memory. A lower budget is faster. To fix the canvas exactly, set `input.canvas_hw`, for example `--set input.canvas_hw=[640,640]`.

The canvas is saved in the checkpoint, so prediction and export always use the size the model was trained at.

## Auto-config

At the start of training, Mayaku fills in the settings you left unset from your training annotations. Each change is logged:

```text
[mayaku.train] auto-config train.epochs: 125 -> 100
```

| Field | Set to | When |
|---|---|---|
| `model.num_classes` | The number of categories | Always |
| `input.canvas_hw` | The canvas for your images' aspect ratio under `input.size_budget` | Always |
| `model.keypoints.names`, `model.keypoints.flip_pairs` | Taken from the categories' keypoint names | Keypoint models |
| `train.epochs` | `100` | Fine-tuning, 10 or more images |
| `train.final_frac` | `0.2`: the last 20% of epochs train without mosaic | Fine-tuning, 10 or more images |
| `train.assigner_warmup` | `0` | Fine-tuning, 10 or more images |
| `train.aug.mosaic` | `0.1` under 500 images, `0.2` under 2,000, `0.3` under 5,000 | Fine-tuning; 5,000 images or more keep `1.0` |
| `train.optimizer`, `train.lr` | `adamw` at `0.01 / (4 + classes)` for nano; other tiers keep `sgd` at `0.01` | Fine-tuning, 10 or more images |

Rules:

- **Anything you set yourself wins.** A field counts as set when it appears in your YAML, in `overrides` or `--set`, or through `--epochs` / `--size-budget`. Auto-config never touches it.
- Auto-config never changes the model size, the heads or the loss.
- `train.optimizer` and `train.lr` go together: set either and auto-config leaves both at the recipe's values.
- `--set auto_config.enabled=false` turns it off. The class count and canvas are still filled in, because a model can't be built without them, but plainly: the data's class count and a square canvas at the size budget.
- If you set `model.num_classes` and it disagrees with the data, training stops with an error.

## The run directory

By default a run writes to `runs/<name>/`, where `<name>` is the weights file's name (`mayaku-n-det`), your YAML file's name, or `mayaku_run`. Use `output_dir` / `--output` to choose another directory.

> **Note:** two runs with the same name write to the same directory, and the second overwrites the first's files. Give each experiment its own `--output`.

```text
runs/mayaku-n-det/
├── train/
│   ├── config.yaml      the full resolved config, auto-config included
│   ├── recipe.json      the training recipe and canvas, as used
│   ├── tier.json        the network's architecture
│   ├── log.jsonl        one JSON record per epoch: losses, learning rate, time, metrics
│   ├── best.pt          the model at its best validation score (only with a validation split)
│   ├── best.json        that epoch's number and metrics
│   ├── last.pt          the model after the final epoch
│   ├── state.pt         everything needed to resume (see Resuming)
│   └── metadata.json    a run summary: sizes, timings, metrics, versions, device
└── eval/
    └── metrics.json     the final evaluation of the final weights
```

- `best.pt` and `last.pt` are **self-contained**: each holds the weights, the full config, the class names, the canvas and the prediction thresholds. Pass either to `from_pretrained`, `mayaku predict`, `mayaku eval`, `mayaku export`, or back to `train` as `weights`.
- Both hold the EMA weights (an exponential moving average of the training weights), which is the model that is evaluated.
- `best.pt` is the epoch with the highest **box AP** (top-1 accuracy for a classifier). This is also true for segmentation and keypoint models.
- `state.pt` is much larger than the others because it also holds the optimizer state. You don't need it once training is done.

## Validation

With `val_annotations` and `val_images` (`--val-annotations` and `--val-images`):

- The model is evaluated after every epoch, and the line is logged:
  ```text
  epoch  41  box 0.812 cls 0.655 dfl 0.901  AP 0.6123  AP50 0.8540  AP75 0.6702  S 0.3120  M 0.5874  L 0.7012
  ```
- `best.pt` is kept, and after training it is scored once more through the same code path a deployment uses. The result goes to `eval/metrics.json` and `result["metrics"]`.

Without a validation split, the run trains without scoring, writes no `best.pt`, and `last.pt` is the result. `result["metrics"]` is `None`.

`S`, `M` and `L` are AP on small, medium and large objects. `-1.0` means the split has no objects of that size. See [Evaluation](/mayaku/deployment/evaluation/).

## The result

`train` returns a dict:

| Key | Value |
|---|---|
| `final_weights` | Path to `best.pt`, or `last.pt` without a validation split |
| `output_dir` | The run directory |
| `metrics` | Final evaluation metrics (dict), or `None` without a validation split |
| `best` | The best epoch's number and metrics (the contents of `best.json`), or `None` |
| `train_seconds` | Training wall-clock time |
| `eval_seconds` | Final evaluation time, or `None` |
| `metadata` | The contents of `metadata.json` |

## The training recipe

The defaults are the recipe Mayaku is tuned with. The fields you are most likely to change:

| Field | Default | Meaning |
|---|---|---|
| `train.epochs` | `125` (`100` when fine-tuning) | Number of passes over the training set. |
| `train.batch` | `8` | Images per step, across all GPUs. Fits every tier on a 12 GB GPU; raise it on bigger or several GPUs. |
| `train.lr` | `0.01` | Peak learning rate. It warms up over `train.warmup_epochs` (3), then follows a cosine down to `lr × train.lr_final_frac`. |
| `train.optimizer` | `sgd` | `sgd`, `adamw` or `musgd`. |
| `train.aug.mosaic` | `1.0` | Probability of a four-image mosaic. |
| `train.aug.vflip` | `0.0` | Vertical flip probability. Useful for aerial or microscope images that have no "up". Don't use it with keypoints. |
| `train.amp` | `true` | Mixed precision on CUDA. |
| `dataloader.num_workers` | `8` | Image-loading processes. |

**Batch size and gradient accumulation.** Gradients are accumulated until about 64 images (`train.lr_ref_batch`) have been seen before each optimizer step, so `train.lr` means the same thing at any batch size. At the default `batch: 8` each step accumulates 8 batches. Lowering `train.batch` to fit in memory does not require changing the learning rate.

**The final stage.** The last `train.final_frac` of the epochs (16% from scratch, 20% when fine-tuning) train without mosaic or scale and translate jitter, on images that look like the ones the model will see in use. The log announces it: `epoch 80: mosaic off, final stage ...`.

**Quantization-aware training (QAT)** is on by default (`model.qat: true`). The model learns weights that keep their accuracy in int8, so the same checkpoint exports to float or int8 (see [Export](/mayaku/deployment/export/#int8)). After each evaluation, training also scores the first 500 validation images in simulated int8 and logs the difference:

```text
epoch  41  ...  AP 0.6123 ...  int8 vs float -0.3 AP
```

It appends `WARNING` when int8 loses more than 1 AP point. Turn QAT off with `--set model.qat=false` if you will only ever deploy in float.

All recipe fields are listed in [Configuration](/mayaku/reference/configuration/#train).

## Resuming

After every epoch, `train/state.pt` holds everything needed to continue: weights, EMA weights, optimizer, loss scaler, best score so far, and the config. To continue an interrupted run, pass the state file or the run's `train/` directory, with the same dataset paths:

```bash
mayaku train --resume runs/mayaku-n-det/train \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

```python
train(resume="runs/mayaku-n-det/train",
      train_annotations="data/train/_annotations.coco.json", train_images="data/train",
      val_annotations="data/valid/_annotations.coco.json", val_images="data/valid")
```

- Training continues at the epoch after the last completed one.
- The config comes from the state. Don't pass `config`, `weights`, `num_epochs`, `size_budget` or `overrides`: that fails with `resume restores the run's own config`.
- The training split must have the same classes as the original run, or it fails with `... is not the dataset this run was training on`.
- Output goes to the same run directory unless you pass `--output`.

## Multiple GPUs

On one machine with several NVIDIA GPUs, set `num_gpus` / `--num-gpus`:

```bash
mayaku train --weights mayaku-m-det --num-gpus 4 \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

- One process runs per GPU.
- `train.batch` is the **total** batch across GPUs, so the recipe trains the same on 1 GPU or 8: with `batch: 32` on 4 GPUs, each GPU takes 8 images per step. The batch must divide evenly by the GPU count, otherwise training stops with `train.batch 8 does not split over 3 GPUs; use a multiple of 3`.
- BatchNorm statistics are synchronized across GPUs (`train.sync_bn: true`), so the run matches its single-GPU equivalent.
- Only the first process evaluates, logs and writes files.
- MPS (Apple Silicon) does not support multiple devices.

For several machines, start Mayaku under `torchrun`, leave `--num-gpus` at 1, and torchrun sets the process count:

```bash
torchrun --nnodes 2 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint HOST:29500 \
  -m mayaku.cli train --weights mayaku-m-det \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

Every process must see the dataset at the same paths. From Python under torchrun, every process calls `mayaku.train(...)` with the same arguments, and every process gets the first process's result.

## Logging from Python

`train` prints its log lines. Pass `log=` to send them somewhere else:

```python
import logging
from mayaku import train

logger = logging.getLogger("mayaku")
train(weights="mayaku-n-det", log=logger.info,
      train_annotations="data/train/_annotations.coco.json", train_images="data/train")
```

The per-epoch records, with every loss term and metric, are in `train/log.jsonl`, one JSON object per line. Read it with `pandas.read_json("runs/mayaku-n-det/train/log.jsonl", lines=True)`.

## train() arguments

| Argument | CLI | Default | Meaning |
|---|---|---|---|
| `config` | positional `CONFIG` | `None` | YAML path or `MayakuConfig`. |
| `weights` | `--weights` | `None` | Hosted model name or checkpoint path to start from. |
| `train_annotations` | `--annotations` | `None` | Training COCO JSON. Required, except for classifiers. |
| `train_images` | `--images` | required | Training image directory, or class-folder root for a classifier. |
| `val_annotations` | `--val-annotations` | `None` | Validation COCO JSON. Pass it together with `val_images`. |
| `val_images` | `--val-images` | `None` | Validation image directory, or class-folder root. |
| `output_dir` | `--output` | `runs/<name>` | Run directory. |
| `size_budget` | `--size-budget` | `None` (800) | Sets `input.size_budget`. |
| `num_epochs` | `--epochs` | `None` | Sets `train.epochs`. |
| `overrides` | `--set key=value` | `None` | Nested mapping of config fields. |
| `device` | `--device` | `"auto"` | `auto`, `cuda`, `cuda:N`, `mps` or `cpu`. |
| `num_gpus` | `--num-gpus` | `1` | GPUs on this machine to train on. |
| `resume` | `--resume` | `None` | `state.pt` or `train/` directory to continue. |
| `log` | none | `print` | Function that receives each log line. |
