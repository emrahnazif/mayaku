---
title: Python API reference
summary: Every public Mayaku function and class, with its signature, parameters and return value.
label: Python API
keywords: mayaku, python api, train, from_pretrained, detections
---

The public API is the four functions in the top-level `mayaku` package, plus the classes and helpers listed below. Anything else under `mayaku.*` is internal and may change.

```python
import mayaku

mayaku.train            # train a model
mayaku.evaluate         # score a model
mayaku.from_pretrained  # load a model for prediction
mayaku.health_check     # inspect a COCO split
mayaku.__version__      # e.g. "3.0.0"
```

## mayaku.train

```python
mayaku.train(
    config=None, *, weights=None,
    train_annotations=None, train_images=None,
    val_annotations=None, val_images=None,
    output_dir=None, size_budget=None, num_epochs=None, overrides=None,
    device="auto", num_gpus=1, resume=None, log=print,
) -> dict
```

Trains a detector (detection, segmentation, keypoints, oriented boxes) from COCO JSON, or a classifier (`model.task: classify`) from class folders.

| Parameter | Type | Meaning |
|---|---|---|
| `config` | `str`, `Path`, `MayakuConfig` or `None` | YAML path or config object. |
| `weights` | `str`, `Path` or `None` | Hosted model name or checkpoint to start from. Without `config`, it also defines the architecture. |
| `train_annotations` | path or `None` | Training COCO JSON. Required for detectors, not allowed for classifiers. |
| `train_images` | path | **Required.** Training images, or the class-folder root. |
| `val_annotations`, `val_images` | path or `None` | Validation split. Pass both for detectors, `val_images` only for classifiers. |
| `output_dir` | path or `None` | Run directory. Default `runs/<name>`. |
| `size_budget` | `int` or `None` | Sets `input.size_budget`. |
| `num_epochs` | `int` or `None` | Sets `train.epochs`. |
| `overrides` | nested mapping or `None` | Config fields, e.g. `{"train": {"lr": 0.005}}`. |
| `device` | `str` | `"auto"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"cpu"`. |
| `num_gpus` | `int` | GPUs on this machine. Leave at 1 under `torchrun`. |
| `resume` | path or `None` | `state.pt` or `train/` directory of a run to continue. |
| `log` | callable | Receives each log line as a string. |

Returns a dict with `final_weights` (`Path`), `output_dir` (`Path`), `metrics` (dict or `None`), `best` (dict or `None`), `train_seconds`, `eval_seconds` and `metadata`. See [Training](/mayaku/data-and-training/training/).

## mayaku.evaluate

```python
mayaku.evaluate(weights, *, annotations=None, images, output_dir=None,
                device="auto", log=print) -> dict
```

Scores a checkpoint, hosted model or exported file. Returns a flat dict of metrics: COCO metrics for detectors (`AP`, `AP50`, ..., plus `segm_*` / `kpt_*`), or `top1` and `top5` for classifiers, whose `annotations` must be `None`. With `output_dir`, it also writes `metrics.json` there. See [Evaluation](/mayaku/deployment/evaluation/).

## mayaku.from_pretrained

```python
mayaku.from_pretrained(source, device="auto") -> Predictor | ArtifactPredictor
```

Loads a model for prediction:

- A checkpoint path or hosted model name gives a `Predictor`, which runs in PyTorch.
- A path ending in `.onnx`, `.mlpackage`, `.xml` or `.engine` gives an `ArtifactPredictor`, which runs in that runtime.

Both have the same interface. See [Prediction](/mayaku/deployment/prediction/).

### Predictor and ArtifactPredictor

`from mayaku.inference import Predictor, ArtifactPredictor`

| Member | Meaning |
|---|---|
| `__call__(image, conf=None)` | One image (path, or `(H, W, 3)` RGB `uint8` array) → `Detections`, or `Classification` for a classifier. `conf` overrides the recorded score threshold. |
| `batch(images, conf=None)` | A list of images → a list of results. |
| `class_names` | `list[str]` |
| `canvas` | `(height, width)` input size |
| `sidecar` | `dict`: the model's embedded description (config, classes, canvas, outputs, decode values, preprocessing, task settings, quantization, export info). |

`Predictor` only:

| Member | Meaning |
|---|---|
| `Predictor.from_checkpoint(path, device="auto")` | Load from a checkpoint file. A name is not downloaded here; use `from_pretrained` for that. |
| `export(target="onnx", output=None, precision=None) -> Path` | Write an exported file. See [Export](/mayaku/deployment/export/#from-python). |
| `model` | The underlying `torch.nn.Module`, in its deployment form. |
| `device` | The `torch.device` it runs on. |

### Detections

`from mayaku.inference import Detections`

A dataclass, with one per image, on the CPU, in original-image pixels, sorted by score:

| Field | Type |
|---|---|
| `boxes` | `Tensor (n, 4)` float32, `[x1, y1, x2, y2]` |
| `scores` | `Tensor (n,)` float32 |
| `labels` | `Tensor (n,)` int64 |
| `masks` | `Tensor (n, H, W)` bool, or `None` |
| `keypoints` | `Tensor (n, K, 3)` float32 `(x, y, visibility probability)`, or `None` |
| `rboxes` | `Tensor (n, 5)` float32 `(cx, cy, w, h, theta)`, theta in radians, or `None` |

`len(detections)` is `n`.

### Classification

`from mayaku.inference.decode import Classification`

| Member | Type |
|---|---|
| `scores` | `Tensor (num_classes,)` probabilities |
| `label` | `int`, the most likely class index |
| `top(k=5)` | `list[tuple[int, float]]`, best first |

## mayaku.health_check

```python
mayaku.health_check(annotations, images, cfg=None) -> dict
```

Scans one COCO split: counts, the canvas training would use (from `cfg`, default `MayakuConfig()`), object-size fractions, box aspect percentiles, class balance and warnings. See [Datasets](/mayaku/data-and-training/datasets/#health-check).

## mayaku.config

```python
from mayaku.config import (
    MayakuConfig, ModelConfig, InputConfig, KeypointConfig, DataLoaderConfig, AutoConfig,
    load_yaml, read_yaml, dump_yaml, to_yaml_string, merge_overrides, parse_assignments,
)
```

| Name | Meaning |
|---|---|
| `MayakuConfig(model=..., input=..., train=..., dataloader=..., auto_config=...)` | The whole configuration. Sections accept dicts. Immutable (pydantic, frozen, unknown fields rejected). See [Configuration](/mayaku/reference/configuration/). |
| `load_yaml(path) -> MayakuConfig` | Read and validate a YAML file. |
| `read_yaml(path) -> (dict, MayakuConfig)` | The same, also returning the raw mapping as written. |
| `dump_yaml(cfg, path)` / `to_yaml_string(cfg)` | Write a config as YAML. |
| `merge_overrides(cfg, overrides) -> MayakuConfig` | Deep-merge a nested mapping into a new config. |
| `parse_assignments(["train.lr=0.005", ...]) -> dict` | Turn `--set`-style strings into a nested mapping. |

## mayaku.utils.download

```python
from mayaku.utils.download import download_model, list_models, list_revisions, resolve_weights, DownloadError
```

| Name | Meaning |
|---|---|
| `download_model(name, *, revision=None, cache_dir=None, manifest_url=..., verify_sha256=True) -> Path` | Return `<dir>/<name>.pth`, downloading it if missing. `cache_dir` is a `pathlib.Path`. |
| `list_models(*, manifest_url=...) -> dict[str, list[str]]` | Hosted model names by task. |
| `list_revisions(name, *, manifest_url=...) -> list[str]` | Dated revisions of a model, oldest first. |
| `resolve_weights(weights) -> Path` | What `train`, `from_pretrained` and the CLI do with a weights argument: an existing file, else a download. |
| `DownloadError` | Raised when a download or integrity check fails. |

See [Models](/mayaku/data-and-training/models/#downloads).

## Helpers for task outputs

| Name | Meaning |
|---|---|
| `mayaku.model.obb.corners(rboxes)` | `(..., 5)` oriented boxes → `(..., 4, 2)` corner points. See [Oriented boxes](/mayaku/more-tasks/oriented-boxes/#predict). |
| `mayaku.inference.export.metadata.read_sidecar(path, target)` | Read the embedded description of an exported file (`target`: `"onnx"`, `"coreml"`, `"openvino"`, `"tensorrt"`). |
| `mayaku.inference.export.metadata.strip_tensorrt_header(path) -> bytes` | The plain engine bytes of a Mayaku `.engine` file. |
