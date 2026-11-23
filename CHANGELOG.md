# Changelog

## Unreleased

- int8 ONNX and OpenVINO exports lose less accuracy. Most activations are ReLU outputs, which are
  never negative, and the symmetric int8 grid gave them only its 128 non-negative levels. These
  exports now give such a tensor the whole int8 grid over the same trained range: half the step,
  and every level the model trained with is still on the grid, so existing checkpoints improve
  without retraining. On the street models (1,467 validation images, ONNX Runtime), the int8 box
  AP gap to fp32 goes from 0.37 to -0.13 for nano and from 0.44 to 0.18 for m; OpenVINO gives the
  same. TensorRT has no zero point and keeps the trained symmetric grid; Core ML keeps it until
  it is measured on the Neural Engine.
- `mayaku.model.export_onnx` takes `regrid=True` for that int8 grid; its default output is
  unchanged.

## v3.1.0: Predictions ready for applications

- Predictions now use their own post-processing: a score threshold, NMS at IoU 0.5 and one class
  per location. Before, a prediction returned the evaluation's output: every detection down to a
  score of 0.001, often several boxes of different classes on one object. `decode="eval"` still
  gives the evaluation's output.
- Training chooses the prediction score threshold on the validation split (the best F1) and saves
  it in the checkpoint. Without a validation split of at least 50 objects, it stays at 0.4.
  `mayaku tune-predict` and `mayaku.tune_predict` choose it for an existing checkpoint.
- `mayaku.evaluate(..., operating=True)` also returns the best-F1 threshold of the split.
- `best.pt` is chosen on what the model is for: mask AP for segmentation, keypoint AP for keypoints,
  box AP for detection, top-1 for classification. With quantization-aware training, every
  evaluation scores the whole validation split in both fp32 and int8, and `best.pt` is the epoch
  with the best lower of the two.
- COCO metrics use faster-coco-eval, now a dependency: the same numbers, several times faster on
  large splits. Segmentation evaluation computes masks only inside their boxes: the same masks,
  about 3x faster during training and 13x faster for exported models.
- Prediction post-processing runs on the CPU unless the model runs on CUDA. On an M1 Max, a 1080p
  frame of a segmentation model decodes in 8.5 ms instead of 107 ms on MPS.
- The documentation pages now come with the package, in the `docs` folder next to the code, with
  `llms.txt` listing every page. Coding agents can read the docs of the installed version offline.
