# Street models: road scenes without training

<!--
summary: Detect people, vehicles, e-scooters, faces and license plates in road scenes, from dashcam to aerial, with a model you use as it is.
label: Street
keywords: mayaku, street, traffic, road scenes, pretrained detector, drone, license plate, face
-->

The street models detect what moves on or around a road, from every viewpoint a camera there has: dashcam, handheld, pole and CCTV, rooftop, drone and straight-down aerial. They are finished models: download one by name and run it, no data and no training needed.

```python
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-m-det-street")   # downloads on first use
det = predictor("crossing.jpg")                     # a path, a PIL image, or an OpenCV (BGR) frame
for box, score, label in zip(det.boxes, det.scores, det.labels):
    print(predictor.class_names[label], round(float(score), 2), box.tolist())
```

The same from the command line:

```bash
mayaku predict mayaku-m-det-street crossing.jpg
```

## The four models

| Name | Size | Gives | Box AP | Mask AP | RTX 3060, fp16 | RTX 3060, int8 |
|---|---|---|---:|---:|---:|---:|
| `mayaku-n-det-street` | nano, 5.2 M | boxes | 43.9 | | 2.61 ms | 2.11 ms |
| `mayaku-n-seg-street` | nano, 5.3 M | boxes and masks | 43.9 | 35.2 | 5.54 ms | 5.64 ms |
| `mayaku-m-det-street` | medium, 32.4 M | boxes | 56.8 | | 6.42 ms | 3.94 ms |
| `mayaku-m-seg-street` | medium, 32.7 M | boxes and masks | 56.8 | 45.5 | 10.4 ms | 7.86 ms |

AP is COCO AP@[.50:.95] on the street validation set (below). Times are end to end on one RTX 3060: a 1920×1080 frame in memory to final boxes (and masks pasted at frame size) in host memory, TensorRT through Mayaku's own export, batch 1, median of 1,000 frames. The `-det` and `-seg` models of one size give the same boxes. For a segmentation model, most of the time goes into pasting the masks into the full frame on the CPU, which grows with the number of objects and which int8 does not speed up.

Which one: **medium** when accuracy matters, especially for small and distant objects and aerial views; **nano** for CPUs, edge devices and many streams per GPU. Take a `-seg` model only when you need the masks.

## The 22 classes

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

**face** and **license plate** are there so you can blur them before you store or share footage.

## Accuracy

On the street validation set: 17,712 images from Objects365, COCO, Wikimedia Commons, drone and aerial sources, 22 classes, COCO AP@[.50:.95].

| | Box AP | AP50 | AP small | AP large | Mask AP |
|---|---:|---:|---:|---:|---:|
| street nano | 43.9 | 60.0 | 23.7 | 62.8 | 35.2 |
| street medium | 56.8 | 72.1 | 34.3 | 73.2 | 45.5 |

Strongest classes (medium, box AP): bus 78.6, train 75.1, car 72.3, person 66.1, face 65.7, e-scooter 65.2, emergency vehicle 64.2, license plate 64.1. Weakest: trailer 24.5, stroller 26.9, wheelchair 32.5 and construction vehicle 39.7, the classes with the fewest training examples.

## Compared with COCO models

On the 10 classes both have (person, bicycle, motorcycle, car, truck, bus, train, traffic light, stop sign, animal), with car and truck merged into one vehicle class because COCO calls pickups trucks and the street models call them cars. Each model runs at its library's default input size, box AP@[.50:.95]:

| Model | Params | Street validation set | COCO val2017 | Network time, RTX 3060 fp16 |
|---|---:|---:|---:|---:|
| **street medium** | 32.4 M | **65.3** | 57.1 | 5.36 ms |
| RF-DETR large | 33.9 M | 56.3 | **61.8** | 5.42 ms |
| YOLO26l | 26.3 M | 53.2 | 60.4 | 4.19 ms |
| **street nano** | 5.2 M | 52.2 | 51.2 | 1.49 ms |
| YOLO26n | 2.6 M | 34.1 | 48.6 | |

Each model wins on its own ground. On street scenes, street medium leads RF-DETR large by 9 AP and wins every shared class, and street nano, at a fifth of its parameters, is within 1 AP of YOLO26l. On COCO's own images and labels, the COCO models lead: the street models never saw a COCO label, and part of the gap is how COCO's annotators drew their boxes. The clearest difference is from above: on drone and aerial images, YOLO26l scores 3 to 9 AP and RF-DETR large 6 to 17, where street medium scores 23 to 25.

Network time is the engine alone, the only figure measured the same way for all three libraries; end to end, each library's own pre- and post-processing adds to it (street medium: 6.42 ms, above).

> **Protocol.** COCO models with their published COCO weights, at their default input size (YOLO26 640, RF-DETR large 704); street models at 800. On the street set, objects of classes COCO has no name for (e-scooter, three-wheeler, trailer...) are ignored for COCO models, so missing them costs nothing. Each side is favoured by its own labelling style, and the COCO models had seen the COCO and Objects365 images of the street set during their training. Parameters are counted from each weights file.

## Prediction settings

The models predict at score 0.4 with NMS at IoU 0.5, set in the checkpoint. [Prediction](/docs/deployment/prediction.md#the-score-threshold) shows, on these models, what a lower or higher threshold trades and what to use with a tracker. They export like every Mayaku model, int8 included: see [Export](/docs/deployment/export.md).

## Fine-tune on your own street data

When the street models are close but not quite right (your own classes, an unusual camera position, more accuracy on your scenes), fine-tune one instead of a base model: it already knows road scenes.

```python
from mayaku import train

result = train(
    weights="mayaku-m-det-street",
    train_annotations="data/train/_annotations.coco.json", train_images="data/train",
    val_annotations="data/valid/_annotations.coco.json", val_images="data/valid",
)
```

With the same 22 classes in the same order, the model keeps everything it learned. With a different number of classes, the class layer starts fresh and the rest of the network is kept. See [What a warm start can and cannot change](/docs/data-and-training/training.md#what-a-warm-start-can-and-cannot-change).

## Known limitations

- **Faces without a person.** Faces on posters, statues and dolls are detected as faces.
- **Truck and car** are the most confused pair: the line between a large van or pickup (car) and a truck is not sharp.
- **Tiny people on drone images**, under about 8 pixels, cause most person false positives from above.
- **Crowds.** NMS removes some overlapping objects of the same class, such as people side by side (see [Prediction](/docs/deployment/prediction.md#the-score-threshold)).
- **Emergency vehicles** are under-counted in the validation labels, so their AP and precision are lower than what you see in use.

## Training data and credits

The training images are from Objects365, COCO, Wikimedia Commons (under their individual free licences), the Songdo Vision dataset (Fonod, Cho, Yeo and Geroliminis, 2025), roundabout aerial images by De-Las-Heras, Sánchez-Soriano and Puertas, and aerial photos from Beeldmateriaal Nederland / PDOK. The labels were made with SAM 3. The weights are Apache 2.0, like the library.
