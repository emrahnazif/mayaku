---
title: Configuration reference
summary: Every configuration field with its default and meaning, and the ways to set them from the CLI, Python or YAML.
label: Configuration
keywords: mayaku, configuration, yaml, hyperparameters, training recipe
---

One configuration describes a model and how it is trained. It has five sections: `model`, `input`, `train`, `dataloader` and `auto_config`. Every run saves its full resolved configuration to `train/config.yaml`, and every checkpoint and exported file carries a copy.

## Setting fields

Use any of these. They can be combined.

```bash
# command line: --set with the dotted path; the value is parsed as YAML
mayaku train --weights mayaku-n-det --set train.batch=32 --set train.aug.vflip=0.5 \
  --set input.canvas_hw=[640,640] \
  --annotations data/train/_annotations.coco.json --images data/train
```

```python
# Python: a nested mapping
from mayaku import train

train(weights="mayaku-n-det",
      overrides={"train": {"batch": 32, "aug": {"vflip": 0.5}}, "input": {"canvas_hw": [640, 640]}},
      train_annotations="data/train/_annotations.coco.json", train_images="data/train")
```

```yaml
# a YAML file, passed as the first argument: mayaku train my_run.yaml ...
# list only the fields you change; everything else keeps its default
model:
  tier: s
train:
  batch: 32
  aug:
    vflip: 0.5
```

To combine a YAML file with extra settings in Python, pass both. `overrides` is applied on top of the file:

```python
from mayaku import train

train("my_run.yaml", weights="mayaku-m-det", overrides={"train": {"epochs": 50}},
      train_annotations="data/train/_annotations.coco.json", train_images="data/train")
```

`config` also accepts a `mayaku.config.MayakuConfig` object. Prefer `overrides` or a YAML file, though, because with an object, auto-config treats more fields as set than you might expect:

- Every field you pass to the constructor counts as set. In `train`, every recipe field counts as set as soon as you pass any of them. So `MayakuConfig(train={"batch": 32})` also fixes `train.epochs` at 125 and turns off the fine-tune schedule.
- An object returned by `merge_overrides` counts **every** field as set. That includes `input.canvas_hw: null`, so the canvas becomes a plain square at the size budget instead of following your data's aspect ratio.
- An object from `load_yaml` follows the first rule: the fields in the file count as set, plus the whole recipe if the file has a `train` section.

Passing the YAML path itself, rather than a loaded object, counts exactly the fields written in the file.

Rules:

- **Unknown fields are errors** at every level. `--set train.lerning_rate=0.01` fails instead of being ignored.
- Values are validated. A wrong type, an out-of-range value, or a size that isn't a multiple of 32 fails before training starts.
- A field you set is never changed by [auto-config](/mayaku/data-and-training/training/#auto-config).
- There is no YAML inheritance. To combine settings, merge them in Python with `merge_overrides`.
- When resuming a run, the configuration comes from the saved state and can't be changed.

To see the full default configuration:

```python
from mayaku.config import MayakuConfig, to_yaml_string

print(to_yaml_string(MayakuConfig()))
```

## model

Which network: the task, the size, and the heads.

| Field | Default | Values | Meaning |
|---|---|---|---|
| `model.task` | `detect` | `detect`, `classify` | `classify` builds an image classifier on the backbone (see [Classification](/mayaku/more-tasks/classification/)). Every detection-family task, including segmentation, keypoints and oriented boxes, is `detect`. |
| `model.tier` | `n` | `n`, `s`, `m`, `l` | Model size. See [Models](/mayaku/data-and-training/models/#sizes). Must match the warm-start weights. |
| `model.num_classes` | `null` | integer > 0 | `null` takes it from the training data. Setting it to a different count than the data is an error. |
| `model.seg` | `false` | bool | Adds the instance mask branch. See [Segmentation](/mayaku/more-tasks/segmentation/). |
| `model.keypoints` | `null` | mapping | Adds the keypoint branch. See below and [Keypoints](/mayaku/more-tasks/keypoints/). |
| `model.obb` | `false` | bool | Predicts oriented boxes. Can't be combined with `seg` or `keypoints`. See [Oriented boxes](/mayaku/more-tasks/oriented-boxes/). |
| `model.mask_stride` | `8` | `4`, `8` | Resolution of the mask features. `4` gives finer masks at more cost. |
| `model.aux_arm` | `box_tower` | `box_tower`, `tower` | Where the mask and keypoint branches read per-location features from: the detection head's box tower, or a separate small tower per level. |
| `model.qat` | `true` | bool | Quantization-aware training, so the model can export to int8. See [Export](/mayaku/deployment/export/#int8). |
| `model.qat_activations` | `symmetric` | `symmetric`, `affine` | The int8 activation scheme trained for. `symmetric` runs on every int8 runtime, TensorRT included. `affine` uses the int8 range more fully but can't run on TensorRT. |

### model.keypoints

| Field | Default | Meaning |
|---|---|---|
| `model.keypoints.num` | required | Keypoints per object (`K`). |
| `model.keypoints.names` | `[]` | `K` names. Filled from the dataset's categories by auto-config. |
| `model.keypoints.flip_pairs` | `[]` | Index pairs swapped by a horizontal flip. Derived from `left_*` / `right_*` names by auto-config. |

`--set model.keypoints.num=17` is enough to turn the branch on. Names and pairs come from the data.

## input

The canvas: the fixed input size every image is letterboxed onto, for training, evaluation, export and deployment.

| Field | Default | Meaning |
|---|---|---|
| `input.size_budget` | `800` | The canvas has at most `size_budget²` pixels. Must be a multiple of 32. The command-line shortcut is `--size-budget`. For classifiers the default is 224. |
| `input.canvas_hw` | `null` | The canvas `[height, width]`. `null` lets auto-config pick the shape that fits your data's aspect ratio under the budget. Set it to fix the canvas exactly. Both sides must be multiples of 32. |

See [Training: input size](/mayaku/data-and-training/training/#input-size).

## train

The training recipe. The defaults are the tuned recipe. Fine-tuning runs change a few of them through auto-config.

### Schedule and optimizer

| Field | Default | Meaning |
|---|---|---|
| `train.epochs` | `125` | Passes over the training set. Auto-config sets `100` when fine-tuning. Shortcut: `--epochs`. |
| `train.batch` | `8` | Images per step across all GPUs. Must divide by the GPU count. 8 fits every tier on a 12 GB GPU; raise it on bigger or several GPUs. |
| `train.optimizer` | `sgd` | `sgd` (Nesterov), `adamw`, or `musgd` (SGD mixed with an orthogonalized Muon step on hidden convolutions). Auto-config picks `adamw` for nano fine-tunes. |
| `train.lr` | `0.01` | Peak learning rate. Auto-config sets `0.01 / (4 + classes)` when it picks `adamw`. |
| `train.lr_final_frac` | `0.01` | The final learning rate as a fraction of `lr`, reached by a cosine schedule. |
| `train.lr_ref_batch` | `64` | Images per optimizer step: gradients of `lr_ref_batch / batch` micro-batches are summed into one step, so `lr` means the same at any `batch`. During warmup a step starts at one micro-batch and grows to the full count, so small datasets learn from the first epoch. Weight decay is scaled to match. |
| `train.momentum` | `0.937` | SGD and MuSGD momentum. |
| `train.beta1` | `0.9` | AdamW beta1. |
| `train.muon_mix` | `0.5` | MuSGD only: the share of each hidden-convolution step taken along the orthogonalized direction. `0` is plain SGD. |
| `train.weight_decay` | `0.0005` | Applied to weights only, not to normalization layers or biases. |
| `train.warmup_epochs` | `3.0` | Warmup length in epochs: the learning-rate ramp, and the ramp of micro-batches per step (see `lr_ref_batch`). |
| `train.warmup_momentum_start` | `0.8` | Momentum at the start of warmup. |
| `train.warmup_bias_lr_start` | `0.1` | Bias learning rate at the start of warmup. It ramps down to `lr`. SGD and MuSGD only: under `adamw` biases warm up from `0` like every other weight. |
| `train.seed` | `0` | Random seed. |
| `train.amp` | `true` | Mixed-precision training. Applies on CUDA only. |
| `train.sync_bn` | `true` | Multi-GPU only: BatchNorm statistics over the whole batch rather than each GPU's share. |

### Final stage, EMA and checkpoint quality

| Field | Default | Meaning |
|---|---|---|
| `train.final_frac` | `0.16` | Share of epochs, at the end, trained with `train.final_aug`: no mosaic or geometric jitter. Auto-config sets `0.2` when fine-tuning. |
| `train.ema_decay` | `0.9999` | Decay of the exponential moving average of the weights. The EMA weights are what is evaluated and saved. |
| `train.ema_tau` | `2000.0` | Ramp of the EMA decay in steps, so the average is useful from the first steps. |
| `train.recalibrate_images` | `640` | Before each evaluation and checkpoint, BatchNorm statistics of the EMA model are recomputed over this many unaugmented training images. `0` turns it off. |
| `train.multiscale` | `0` | Multi-scale training: the smallest size budget to train at, `0` for off. Each batch is downscaled to a random size between this budget and the canvas. Evaluation always uses the canvas. It usually costs accuracy at the canvas size. Use it only if you will deploy smaller than you train. |

### Loss and label assignment

| Field | Default | Meaning |
|---|---|---|
| `train.assigner_warmup` | `5` | Epochs of ATSS assignment before switching to task-aligned assignment (TAL). Auto-config sets `0` when fine-tuning. |
| `train.atss_soft` | `false` | Scale the ATSS class target by the best IoU. |
| `train.tal_beta` | `6.0` | The IoU exponent in TAL's alignment metric. Lower values let lower-IoU small objects rank. |
| `train.cls_loss` | `bce` | Classification loss: `bce` or `vfl` (varifocal). |
| `train.loc_weight_floor` | `0.0` | Minimum weight of the box and DFL loss per positive. `0` is off. |
| `train.seg_gain` | `2.0` | Weight of the mask (Dice) loss. |
| `train.seg_cap` | `250` | Maximum positives per batch for the mask loss. |

### train.aug

Augmentation during training. `train.final_aug` has the same fields and is used for the final stage.

| Field | Default (`aug`) | Default (`final_aug`) | Meaning |
|---|---|---|---|
| `mosaic` | `1.0` | `0.0` | Probability of combining four images into one. Auto-config lowers it for small datasets when fine-tuning. |
| `mixup` | `0.0` | `0.0` | Probability of blending in a second image. |
| `scale` | `[0.5, 1.5]` | `[1.0, 1.0]` | Random scale range. |
| `translate` | `0.1` | `0.0` | Random shift, as a fraction of the canvas. |
| `hflip` | `0.5` | `0.5` | Horizontal flip probability. |
| `vflip` | `0.0` | `0.0` | Vertical flip probability, for images with no "up" (aerial, microscopy). Don't use it with keypoints. |
| `hsv` | `[0.015, 0.7, 0.4]` | `[0.015, 0.7, 0.4]` | Hue, saturation and value jitter strengths. |

Classifiers use `hflip`, `vflip` and `hsv`, plus their own random crop, which is turned off when `scale` is `[1.0, 1.0]`. They don't use mosaic, mixup or translate.

### Decode settings

Post-processing used during evaluation and saved into the checkpoint for prediction and export.

| Field | Default | Meaning |
|---|---|---|
| `train.decode.conf` | `0.001` | Minimum score. Low, as COCO AP needs. You can override it on each prediction call with `conf=`. |
| `train.decode.iou` | `0.7` | NMS overlap threshold. |
| `train.decode.max_det` | `300` | Maximum detections per image. |
| `train.decode.topk` | `1000` | Candidates kept before NMS. |
| `train.decode.multi_label` | `true` | Allow one box to be reported under several classes. |

See [Prediction: the score threshold](/mayaku/deployment/prediction/#the-score-threshold).

## dataloader

| Field | Default | Meaning |
|---|---|---|
| `dataloader.num_workers` | `8` | Image-loading worker processes. Lower it if you run short of CPU or RAM. `0` loads images in the main process. |

## auto_config

| Field | Default | Meaning |
|---|---|---|
| `auto_config.enabled` | `true` | Fill unset fields from the training data. With `false`, only the class count and a square canvas at the size budget are filled. |

See [Training: auto-config](/mayaku/data-and-training/training/#auto-config).
