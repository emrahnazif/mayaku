# Pretrained models, sizes and downloads

<!--
summary: The hosted model names, the model sizes with their parameters and compute, and how weights download.
label: Models
keywords: mayaku, pretrained models, model zoo, model sizes, download
-->

## Hosted pretrained models

Pass a model name wherever Mayaku takes weights (`train(weights=...)`, `from_pretrained(...)`, `mayaku predict`, `mayaku eval`, `mayaku export`). The checkpoint downloads the first time it is used.

| Name | Size | Task | Status |
|---|---|---|---|
| `mayaku-n-det` | nano | Object detection | released |
| `mayaku-n-seg` | nano | Instance segmentation | released |
| `mayaku-m-det` | medium | Object detection | released |
| `mayaku-m-seg` | medium | Instance segmentation | released |
| `mayaku-n-det-street` | nano | Object detection, 22 street classes | released |
| `mayaku-n-seg-street` | nano | Instance segmentation, 22 street classes | released |
| `mayaku-m-det-street` | medium | Object detection, 22 street classes | released |
| `mayaku-m-seg-street` | medium | Instance segmentation, 22 street classes | released |

The first four are pretrained on Objects365 with its 365 classes, and are the starting point for training on your own data. The pattern is `mayaku-<size>-<task>`. The `-street` models are ready to use on road and street scenes without training (see [Street models](#street-models)). To see what is currently hosted:

```bash
mayaku download --list
```

No keypoint, oriented-box or classification models are hosted; those tasks are experimental. Train them from the detector weights: [keypoints](/docs/more-tasks/keypoints.md), [oriented boxes](/docs/more-tasks/oriented-boxes.md#train), [classification](/docs/more-tasks/classification.md#train).

## Street models

The `-street` models detect what moves on or around a road, from every viewpoint a camera on a road has: dashcam, handheld, pole and CCTV, rooftop, drone and straight-down aerial. Use them as they are, or fine-tune them on your own street data like any other model.

```python
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-n-seg-street")
detections = predictor("crossing.jpg")
```

The 22 classes:

| Group | Classes |
|---|---|
| People | person, face, helmet |
| Micromobility and assistive | bicycle, e-scooter, motorcycle, wheelchair, stroller |
| Vehicles | car, truck, bus, emergency vehicle, train, three-wheeler, construction vehicle, trailer |
| Road objects | license plate, traffic light, safety sign, stop sign, traffic cone |
| Other | animal |

Some classes are broader or narrower than in COCO:

- **car** is every light vehicle, vans, SUVs and pickups included. **truck** is heavy goods vehicles only.
- **emergency vehicle** is police cars, ambulances and fire engines.
- **e-scooter** is stand-up scooters. Seated scooters and mopeds are **motorcycle**.
- **train** includes trams. **three-wheeler** is tricycles, rickshaws and tuk-tuks. **construction vehicle** is excavators and forklifts.
- **safety sign** is regulatory and warning signs. Street-name, direction and shop signs are not detected. **stop sign** is its own class.
- **helmet** is any helmet, worn or not. **animal** is one class for dogs, cats, horses, cattle, sheep, birds and deer.

Accuracy on the street validation set (17,712 images from Objects365, COCO, Wikimedia Commons, drone and aerial sources, 22 classes, COCO AP @[.5:.95]):

| Model | Box AP | Box AP50 | Mask AP |
|---|---|---|---|
| street nano | 43.9 | 60.0 | 35.2 |
| street medium | 56.8 | 72.1 | 45.5 |

The `-det` and `-seg` models of one size give the same boxes; the `-seg` model also gives masks. The validation labels come from the same automatic labelling as the training labels, so they share its mistakes: for example, faces on posters and statues are labelled as faces, and the model detects them too.

The training images include Wikimedia Commons images under their individual free licences, the Songdo Vision dataset (Fonod, Cho, Yeo and Geroliminis, 2025), roundabout aerial images by De-Las-Heras, Sánchez-Soriano and Puertas, and aerial photos from Beeldmateriaal Nederland / PDOK. The labels were made with SAM 3.

## Sizes

All sizes share one architecture and differ in width and depth. Larger is more accurate and slower. Pretrained weights are hosted for nano and medium; small and large can be trained from scratch with `model.tier`.

| Size | `model.tier` | Parameters | Compute at 640×640 | Pretrained |
|---|---|---|---|---|
| nano | `n` | 5.2 M | 6.7 GMACs | yes |
| small | `s` | 13.9 M | 19.7 GMACs | no |
| medium | `m` | 32.4 M | 46.3 GMACs | yes |
| large | `l` | 57.9 M | 81.9 GMACs | no |

Figures are for a detection model with 80 classes in its exported form. A mask branch adds about 0.1 to 0.3 M parameters and 0.4 to 1 GMACs. The class count changes the parameter count slightly. Compute grows with the canvas: a 576×1024 canvas costs about 1.44 times what 640×640 costs.

How to choose:

- Start with **nano** to check your data and pipeline. It trains fastest and runs in real time on CPUs and edge devices.
- Move up a size when accuracy matters more than speed, especially for small objects and many classes.
- Changing the canvas (`--size-budget`) is the other speed and accuracy dial, and it does not need a different pretrained model.

The architecture:

- A convolutional backbone, a three-level feature pyramid at strides 8, 16 and 32, and an anchor-free detection head that predicts each box side as a distribution (DFL).
- Masks, keypoints and oriented boxes are extra branches on the same detector.
- Overlapping detections are removed by NMS in post-processing, not inside the network.
- The deployed network uses only 1×1 and 3×3 convolutions, ReLU, addition, nearest upsampling and max-pooling (see [Export](/docs/deployment/export.md#the-exported-graph)).

## Downloads

### Where files go

A hosted model downloads to **`<name>.pth` in the current working directory**, for example `./mayaku-n-det.pth`. Set the environment variable `MAYAKU_CACHE_DIR` to download to a fixed directory instead, which is useful for shared machines, CI and containers:

```bash
export MAYAKU_CACHE_DIR=~/.cache/mayaku
```

- **A file already present is used as is.** No network access, no update check, no re-verification. This makes later runs fast and offline.
- **Models never update automatically.** Delete the `.pth` file to get the latest revision on the next use.
- Every download is checked against the size and SHA256 listed in the hosted manifest. A partial or corrupt download is deleted and raises `DownloadError`.

### How a weights argument is resolved

For a `weights` value:

1. If a file exists at that path, it is used. This includes `mayaku-n-det.pth` already in the download directory.
2. Otherwise, if the value contains a `/` or `\`, it was meant as a path, and Mayaku raises `FileNotFoundError: weights file not found: ...`.
3. Otherwise it is treated as a model name and downloaded. A trailing `.pth` is ignored, so `mayaku-n-det` and `mayaku-n-det.pth` are the same name.

### The download command

```bash
mayaku download mayaku-m-det                 # one model
mayaku download --list                       # list hosted models by task
mayaku download --all                        # every hosted model
mayaku download mayaku-m-det --cache-dir weights/
```

| Option | Meaning |
|---|---|
| `--cache-dir DIR` | Download directory. Default: `MAYAKU_CACHE_DIR`, else the current directory. |
| `--manifest-url URL` | Use a mirror's manifest instead of `https://dtmfiles.com/mayaku/v1/models/manifest.json`. |
| `--no-verify` | Skip the SHA256 check. |

### From Python

```python
from pathlib import Path
from mayaku.utils.download import download_model, list_models, list_revisions

list_models()                          # {"detection": ["mayaku-m-det", "mayaku-n-det"], "segmentation": [...]}
list_revisions("mayaku-n-det")         # ["2026-07-02", ...], oldest first
path = download_model("mayaku-n-det")  # Path to the .pth
path = download_model("mayaku-n-det", revision="2026-07-02", cache_dir=Path("weights"))
```

`cache_dir` must be a `pathlib.Path`.

`revision` pins a dated snapshot. It only takes effect when the file isn't already present, because the local file name has no revision in it. Use a separate `cache_dir` per revision if you need several.

### Offline machines and mirrors

Download on a connected machine, copy the `.pth` files, and pass their paths. A local file never touches the network. To serve your own mirror, host the files with a `manifest.json` in the same layout and point `mayaku download --manifest-url` at it.

## Checkpoint compatibility

Checkpoints and exported files record the format version they were written with. Mayaku 3 reads only Mayaku 3 files. A model trained with Mayaku 2.x fails with:

```text
<file> was trained with mayaku 2.x (sidecar schema 1); run it with `pip install 'mayaku<3'`
```

A file that wasn't written by Mayaku, or whose metadata was stripped, fails with `<file> carries no mayaku sidecar`.
