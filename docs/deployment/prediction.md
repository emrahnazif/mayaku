# Run predictions with a trained model

<!--
summary: Load a checkpoint or exported file with from_pretrained, set the score threshold, and read Detections.
label: Prediction
keywords: mayaku, inference, prediction, from_pretrained, detections
-->

`mayaku.from_pretrained` loads any trained Mayaku model, whether a checkpoint, a hosted model name or an exported file, and returns a predictor. Call the predictor with an image to get the results in that image's pixel coordinates.

```python
from mayaku import from_pretrained

predictor = from_pretrained("runs/mayaku-n-det/train/best.pt")
det = predictor("photo.jpg")

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
det = predictor(image, conf=None, decode="predict")            # one image -> Detections (or Classification)
dets = predictor.batch([img1, img2], conf=None, decode="predict")   # several images in one pass -> list
```

`image` is any of these:

- **A file path** (`str` or `pathlib.Path`), read with OpenCV.
- **A NumPy array of shape (H, W, 3) in BGR order,** OpenCV's order, as `cv2.imread` and `cv2.VideoCapture` give it. Pass OpenCV frames as they are. `uint8` is expected. Other types are clipped to 0–255 and converted.
- **A PIL image.** Its colors are known from its type, so pass it as it is: `predictor(pil_image)`.

> **Note:** a NumPy array that holds RGB (from PIL via `np.asarray`, imageio, or matplotlib) must be reversed first: `predictor(rgb[:, :, ::-1])`. Passing the PIL image itself avoids this.

`batch` takes a list of paths, arrays and PIL images, which may be mixed and of different sizes, and runs them as one batch on the device. That is faster per image than calling the predictor in a loop.

### The score threshold

A prediction keeps detections scoring at least the model's **prediction threshold**, removes a box that overlaps a higher-scoring box of the same class by more than **0.5** IoU (NMS), and gives each location at most its best class. The first prediction states these settings once as a Python warning. Pass `conf` to change the threshold for one call:

```python
det = predictor("photo.jpg", conf=0.5)
```

`mayaku predict` takes the same value as `--conf 0.5`.

**The threshold is chosen when the model is trained.** At the end of training, the final weights are scored on the validation split, and the same pass measures precision and recall at every threshold from 0.1 to 0.7 in steps of 0.05. The checkpoint records the threshold with the best F1, the balance of precision and recall. Training logs the choice, for example `prediction threshold 0.25 (precision 0.806, recall 0.684, F1 0.740 on the validation split)`, and keeps it in the checkpoint as `predictor.sidecar["predict"]["conf"]`, with how it was chosen in `predictor.sidecar["predict_tuned"]`. Models whose scores are spread over many classes get a lower threshold (the Objects365 nano gets 0.2); confident fine-tuned models usually get 0.4 to 0.7. On the 100 RF100-VL datasets, the threshold picked on each validation split gave up 0.006 F1 on its test split against the best possible threshold, a fixed 0.4 gave up 0.008, and validation splits of only 100 to 500 objects still picked better than 0.4. Without a validation split, or with fewer than 50 validation objects, the threshold stays at **0.4**. If you prefer fewer false detections to more found objects, set a higher threshold yourself. A threshold you set in the config (`train.predict.conf`) is kept. `mayaku tune-predict` runs the same choice on an existing checkpoint (see [Command line](/docs/reference/cli.md#mayaku-tune-predict)).

The chosen value reflects the validation split. Missing labels in it count as false detections and lower the threshold; a split unlike your footage may suit a different one.

A higher threshold gives fewer false detections and misses more real objects. On the street models' validation set, 0.25 finds 68% of the objects with 81% of the detections correct; 0.4 finds 61% with 90% correct; 0.5 finds 55% with 94% correct. Objects that score near the threshold also appear and disappear more between video frames. A tracker that uses low-scoring detections to keep its tracks alive, such as ByteTrack, needs a lower threshold, for example `conf=0.1`.

NMS at 0.5 removes almost every second box on the same object (94% of them on the street models). Its cost is in crowds: two people side by side or a row of parked cars overlap enough that the lower-scoring one is sometimes removed, about 6% of such overlapping objects.

### Prediction and evaluation post-processing

A model records two sets of post-processing values. Predictions use one; evaluation (`mayaku eval`, and the evaluation during training) uses the other, because COCO AP is measured over every detection: it keeps everything down to a score of 0.001 and lets one box be reported under several classes. In an application those values return up to 300 detections per image, mostly low-scoring, often several boxes with different classes on one object. Pass `decode="eval"` to get exactly the evaluation's output:

```python
det = predictor("photo.jpg", decode="eval")
```

| Value | Prediction (`train.predict`) | Evaluation (`train.decode`) | Meaning |
|---|---|---|---|
| `conf` | chosen at the end of training, else `0.4` | `0.001` | Minimum score. You can override it on each call with `conf=`. |
| `iou` | `0.5` | `0.7` | Overlap above which NMS removes the lower-scoring of two same-class boxes. |
| `max_det` | `300` | `300` | Maximum detections per image. |
| `topk` | `1000` | `1000` | Candidates kept before NMS. |
| `multi_label` | `false` | `true` | An object may be reported once per class that scores above `conf`, so the same box can appear with two different classes. With `false`, each location gives at most its best class. |

Set a model's own prediction values when you train it (see [Configuration](/docs/reference/configuration.md#decode-settings)). They are in `predictor.sidecar["predict"]`, and the evaluation's in `predictor.sidecar["decode"]`. Checkpoints saved before prediction values existed use the prediction defaults above.

AP computed from predictions is lower than the model's real AP, because predictions leave out the low-scoring detections that AP counts: the street nano measures 43.5 AP with `decode="eval"` and 34.6 from its predictions at 0.4. Use `mayaku eval`, or `decode="eval"` when you compute metrics yourself.

Post-processing runs on the GPU when the model runs on CUDA, and on the CPU otherwise, including on Apple GPUs (MPS), where the CPU is several times faster for this work.

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

Details on each optional field are on the task pages: [masks](/docs/more-tasks/segmentation.md#predict), [keypoints](/docs/more-tasks/keypoints.md#predict), [oriented boxes](/docs/more-tasks/oriented-boxes.md#predict).

## Classification

A classifier returns a `Classification` for each image:

| Member | Meaning |
|---|---|
| `scores` | `(num_classes,)` float tensor of probabilities, summing to 1 |
| `label` | Index of the most likely class |
| `top(k=5)` | List of `(class_index, probability)` pairs, best first |

`conf` has no effect on a classifier. See [Image classification](/docs/more-tasks/classification.md#predict).

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
    det = predictor(frame)                             # OpenCV frames are BGR, as expected
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
3. Feed the network planar RGB channels scaled by 1/255. (Mayaku's own predictors take BGR images and reverse the channels in the same step.)

Results are mapped back through the same resize and padding, so coordinates are always in the original image. This recipe is recorded in `predictor.sidecar["preprocess"]`. Implement exactly this if you write your own runtime around an exported model.
