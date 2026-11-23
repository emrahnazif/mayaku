---
title: Mayaku Street
emoji: 🚦
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Road objects, faces and plates, dashcam to drone
models:
- enazif/mayaku-m-seg-street
- enazif/mayaku-m-det-street
- enazif/mayaku-n-seg-street
- enazif/mayaku-n-det-street
tags:
- mayaku
- object-detection
- instance-segmentation
- traffic
- privacy
---

Detect people, vehicles, e-scooters, faces, license plates and 15 more road classes in a photo
with the [Mayaku street models](https://emrah.fr/mayaku/ready-to-use/street/), and blur faces and
license plates with [PixelFlow](https://pypi.org/project/pixelflow/).

Run the same on your machine:

```bash
pip install mayaku
mayaku predict mayaku-m-seg-street photo.jpg
```
