---
title: Command-line reference
summary: Every mayaku command and option: train, eval, predict, export and download.
label: Command line
keywords: mayaku, cli, command line, train, export
---

The `mayaku` command has five subcommands. `python -m mayaku.cli` is the same program.

```text
mayaku train    [CONFIG] --images DIR [options]     train a model
mayaku eval     WEIGHTS --images DIR [options]      score a model on a labeled split
mayaku predict  WEIGHTS IMAGE [options]             run a model on one image, print JSON
mayaku export   TARGET WEIGHTS [options]            write a deployment file
mayaku download [NAME] [options]                    fetch hosted pretrained models
```

`WEIGHTS` is a checkpoint path, a hosted model name such as `mayaku-n-det`, or, for `eval` and `predict`, an exported file (`.onnx`, `.mlpackage`, `.xml`, `.engine`). Every subcommand takes `--help`.

## mayaku train

Train a detector, segmentation, keypoint or oriented-box model on COCO data, or a classifier on class folders. The command-line form of [`mayaku.train`](/mayaku/data-and-training/training/).

```bash
mayaku train [CONFIG] [--weights W] --annotations A --images I
             [--val-annotations VA --val-images VI] [--output DIR]
             [--epochs N] [--size-budget S] [--set key=value ...]
             [--device D] [--num-gpus N] [--resume PATH]
```

| Argument / option | Meaning |
|---|---|
| `CONFIG` | Optional YAML config. Without it, `--weights` defines the architecture, or the defaults apply. |
| `--weights W` | Checkpoint path or hosted model name to start from. |
| `--annotations A` | Training COCO JSON. Leave it out for a classifier. |
| `--images I` | **Required.** Training image directory, or the class-folder root for a classifier. |
| `--val-annotations VA` | Validation COCO JSON. Turns on per-epoch evaluation and `best.pt`. |
| `--val-images VI` | Validation image directory, or class-folder root. |
| `--output DIR` | Run directory. Default `./runs/<name>`. |
| `--epochs N` | Sets `train.epochs`. |
| `--size-budget S` | Sets `input.size_budget`, a multiple of 32. |
| `--set key=value` | Sets any [config field](/mayaku/reference/configuration/) by its dotted path. Repeatable. The value is parsed as YAML. |
| `--device D` | `auto` (default), `cuda`, `cuda:N`, `mps` or `cpu`. |
| `--num-gpus N` | Train on N GPUs of this machine. Default 1. |
| `--resume PATH` | Continue a run from its `state.pt` or its `train/` directory. Give the same dataset options. Don't combine with `CONFIG`, `--weights`, `--epochs`, `--size-budget` or `--set`. |

When it finishes, it prints `final weights: <path>`.

Examples:

```bash
# fine-tune the small detector
mayaku train --weights mayaku-s-det \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid

# segmentation from detector weights, 640 budget, batch 8
mayaku train --weights mayaku-n-det --set model.seg=true --size-budget 640 --set train.batch=8 \
  --annotations data/train/_annotations.coco.json --images data/train

# classifier
mayaku train --weights mayaku-n-det --set model.task=classify --epochs 30 \
  --images data/train --val-images data/val
```

## mayaku eval

Score a model on a labeled split and print the metrics as JSON. The command-line form of [`mayaku.evaluate`](/mayaku/deployment/evaluation/).

```bash
mayaku eval WEIGHTS [--annotations A] --images I [--output DIR] [--device D]
```

| Argument / option | Meaning |
|---|---|
| `WEIGHTS` | Checkpoint, hosted model name or exported file. |
| `--annotations A` | COCO JSON. Required for detectors. Leave it out for a classifier. |
| `--images I` | **Required.** Image directory, or the class-folder root for a classifier. |
| `--output DIR` | Also write `DIR/metrics.json`. |
| `--device D` | `auto` (default), `cuda`, `mps` or `cpu`. |

## mayaku predict

Run a model on one image and print the result as JSON. See [Prediction](/mayaku/deployment/prediction/#command-line) for the format.

```bash
mayaku predict WEIGHTS IMAGE [--conf C] [--output FILE] [--device D]
```

| Argument / option | Meaning |
|---|---|
| `WEIGHTS` | Checkpoint, hosted model name or exported file. |
| `IMAGE` | Image file. |
| `--conf C` | Score threshold. Default: the model's recorded one, usually `0.001`, which returns many low-score detections. Pass a value such as `0.25` for practical output. |
| `--output FILE` | Write the JSON to a file instead of printing it. |
| `--device D` | `auto` (default), `cuda`, `mps` or `cpu`. |

## mayaku export

Write a deployment file. The file's outputs are checked against the PyTorch model before the command succeeds. See [Export and deployment](/mayaku/deployment/export/).

```bash
mayaku export TARGET WEIGHTS [--output FILE] [--precision P]
```

| Argument / option | Meaning |
|---|---|
| `TARGET` | `onnx`, `coreml`, `openvino` or `tensorrt`. |
| `WEIGHTS` | Checkpoint or hosted model name. An exported file can't be exported again. |
| `--output FILE` | Output path. Default: next to the checkpoint, with the target's extension (`.onnx`, `.mlpackage`, `.xml`, `.engine`). |
| `--precision P` | `fp32`, `fp16` or `int8`, as the target supports. Default: `fp32` for ONNX and OpenVINO, `fp16` for Core ML and TensorRT. |

Prints the path written.

## mayaku download

Fetch hosted pretrained checkpoints. See [Models](/mayaku/data-and-training/models/#downloads).

```bash
mayaku download [NAME] [--list] [--all] [--cache-dir DIR] [--manifest-url URL] [--no-verify]
```

| Argument / option | Meaning |
|---|---|
| `NAME` | Model name, for example `mayaku-m-seg`. |
| `--list` | List hosted models, grouped by task. |
| `--all` | Download every hosted model. |
| `--cache-dir DIR` | Download directory. Default: `MAYAKU_CACHE_DIR`, else the current directory. |
| `--manifest-url URL` | Manifest of a mirror. |
| `--no-verify` | Skip the SHA256 check of new downloads. |

Prints `<name>: <path>` for each model. A model already in the download directory is not downloaded again.

## Environment variables

| Variable | Effect |
|---|---|
| `MAYAKU_CACHE_DIR` | Directory hosted models download to and are looked up in. Default: the current directory. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Set to `1` by Mayaku on import, unless you set it yourself, so operations without an Apple GPU kernel fall back to the CPU. |
