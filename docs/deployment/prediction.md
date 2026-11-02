---
title: Run predictions with a trained model
summary: Load a checkpoint or exported file with from_pretrained, set the score threshold, and read Detections.
label: Prediction
keywords: mayaku, inference, prediction, from_pretrained, detections
---

`mayaku.from_pretrained` loads any trained Mayaku model, whether a checkpoint, a hosted model name or an exported file, and returns a predictor. Call the predictor with an image to get the results in that image's pixel coordinates.

```python
from mayaku import from_pretrained

predictor = from_pretrained("runs/mayaku-n-det/train/best.pt")
det = predictor("photo.jpg", conf=0.25)

for box, score, label in zip(det.boxes.tolist(), det.scores.tolist(), det.labels.tolist()):
    print(predictor.class_names[label], f"{score:.2f}", [round(v) for v in box])
```

## Loading a model

```python
from_pretrained(source, device="auto")
```

`source` may be:

| Source | Example | Runs on |
|---|---|---|
| A checkpoint written by Mayaku | `"runs/exp/train/best.pt"` | PyTorch |
| A hosted model name | `"mayaku-n-det"` | PyTorch, after downloading to `./mayaku-n-det.pth` |
| An ONNX file | `"model.onnx"` | onnxruntime (CUDA if available, else CPU) |
| A Core ML package | `"model.mlpackage"` | Core ML (macOS), Neural Engine where possible |
| An OpenVINO model | `"model.xml"` (with `model.bin` next to it) | OpenVINO CPU |
| A TensorRT engine | `"model.engine"` | TensorRT on a CUDA GPU |

The file type is chosen by the extension. Every one of these is called the same way and returns the same result types. A checkpoint or exported file carries its own class names, canvas size, preprocessing and thresholds, so no config file is needed.

`device` (`"auto"`, `"cuda"`, `"cuda:1"`, `"mps"`, `"cpu"`) chooses where a PyTorch checkpoint runs. `auto` picks CUDA, then MPS, then CPU. For exported files, each runtime decides placement as described in the table.

## Calling the predictor

```python
det = predictor(image, conf=None)            # one image -> Detections (or Classification)
dets = predictor.batch([img1, img2], conf=None)   # several images in one pass -> list
```

`image` is either of these:

- **A file path** (`str` or `pathlib.Path`), read with OpenCV.
- **A NumPy array of shape (H, W, 3) in RGB order.** `uint8` is expected. Other types are clipped to 0–255 and converted.

> **Note:** OpenCV frames are BGR. Convert them before passing: `predictor(frame[:, :, ::-1])` or `predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))`. A file path needs no conversion.

From PIL: `predictor(np.asarray(pil_image.convert("RGB")))`.

`batch` takes a list of paths and arrays, which may be mixed and of different sizes, and runs them as one batch on the device. That is faster per image than calling the predictor in a loop.

### The score threshold

**The default threshold is very low: `0.001`.** It is the threshold the model was evaluated with, because COCO AP is measured over all detections. Called without `conf`, a predictor returns up to 300 detections per image, most with low scores. For an application, pass a threshold:

```python
det = predictor("photo.jpg", conf=0.25)
```

`mayaku predict` takes the same value as `--conf 0.25`.

The other post-processing values are recorded in the model when it is trained (field `train.decode`, see [Configuration](/mayaku/reference/configuration/#decode-settings)):

| Value | Default | Meaning |
|---|---|---|
| `conf` | `0.001` | Minimum score. You can override it on each call. |
| `iou` | `0.7` | Overlap above which NMS removes the lower-scoring of two same-class boxes. |
| `max_det` | `300` | Maximum detections per image. |
| `topk` | `1000` | Candidates kept before NMS. |
| `multi_label` | `true` | An object may be reported once per class that scores above `conf`, so the same box can appear with two different classes. With `false`, each location gives at most its best class. |

They are stored in `predictor.sidecar["decode"]`. Changing a value in that dict changes the predictor's later calls, for example `predictor.sidecar["decode"]["max_det"] = 50`.

## Detections

Detection, segmentation, keypoint and oriented-box models return a `Detections` object for each image. Its tensors are on the CPU, use the original image's pixel coordinates, and are sorted by score, highest first.

| Field | Shape and type | Present for |
|---|---|---|
| `boxes` | `(n, 4)` float32, `[x1, y1, x2, y2]`, clipped to the image | Every detector |
| `scores` | `(n,)` float32, 0–1 | Every detector |
| `labels` | `(n,)` int64 class indices into `predictor.class_names` | Every detector |
| `masks` | `(n, H, W)` bool, at the image's full size | Segmentation models, otherwise `None` |
| `keypoints` | `(n, K, 3)` float32: `x`, `y`, visibility probability | Keypoint models, otherwise `None` |
| `rboxes` | `(n, 5)` float32: `cx`, `cy`, `w`, `h`, `theta` (radians) | Oriented-box models, otherwise `None` |

`len(det)` is the number of detections. Row `i` of every field describes the same object.

Details on each optional field are on the task pages: [masks](/mayaku/more-tasks/segmentation/#predict), [keypoints](/mayaku/more-tasks/keypoints/#predict), [oriented boxes](/mayaku/more-tasks/oriented-boxes/#predict).

## Classification

A classifier returns a `Classification` for each image:

| Member | Meaning |
|---|---|
| `scores` | `(num_classes,)` float tensor of probabilities, summing to 1 |
| `label` | Index of the most likely class |
| `top(k=5)` | List of `(class_index, probability)` pairs, best first |

`conf` has no effect on a classifier. See [Image classification](/mayaku/more-tasks/classification/#predict).

## Predictor attributes

| Attribute | Meaning |
|---|---|
| `class_names` | List of class names. `class_names[label]` names a detection. |
| `canvas` | `(height, width)` input size the model runs at |
| `sidecar` | The model's full description, a dict: config, classes, canvas, outputs, post-processing values, preprocessing, mask/keypoint/oriented-box settings, quantization, export target and precision |

## Example: a video or webcam

```python
import cv2
from mayaku import from_pretrained

predictor = from_pretrained("runs/mayaku-n-det/train/best.pt")
cap = cv2.VideoCapture(0)              # or a file path

while True:
    ok, frame = cap.read()
    if not ok:
        break
    det = predictor(frame[:, :, ::-1], conf=0.3)       # BGR -> RGB
    for (x1, y1, x2, y2), label in zip(det.boxes.int().tolist(), det.labels.tolist()):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, predictor.class_names[label], (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("mayaku", frame)
    if cv2.waitKey(1) == 27:           # Esc
        break
```

## Command line

```bash
mayaku predict WEIGHTS IMAGE [--conf C] [--output FILE] [--device D]
```

It prints JSON, or writes it to `--output`:

```json
{
  "image": "photo.jpg",
  "detections": [
    {"class_id": 2, "class": "bolt", "score": 0.9312, "box_xyxy": [101.42, 87.5, 188.03, 160.27]}
  ]
}
```

- `box_xyxy` is in the image's pixels.
- Segmentation models add `segmentation`, a COCO RLE dict.
- Keypoint models add `keypoints`, a list of `[x, y, visibility_probability]`.
- Oriented-box models add `rbox_xywhr` (`[cx, cy, w, h, theta_radians]`) and `corners`, the four corner points.
- Classifiers print `"classes"` (the top five) instead of `"detections"`.

`mayaku predict` reads one image per call. For many images, use the Python API with `predictor.batch`.

## Preprocessing

Every predictor, every exported runtime, and training itself prepare images the same way:

1. Resize to fit the canvas, keeping the aspect ratio (bilinear interpolation).
2. Pad the rest of the canvas with gray (value 114), keeping the image centered.
3. RGB channel order, scale pixel values by 1/255.

Results are mapped back through the same resize and padding, so coordinates are always in the original image. This recipe is recorded in `predictor.sidecar["preprocess"]`. Implement exactly this if you write your own runtime around an exported model.
