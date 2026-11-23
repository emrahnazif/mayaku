"""Mayaku street models: a demo on Hugging Face Spaces.

Detect road and street objects in a photo with one of the four street models,
and blur faces and license plates with PixelFlow.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import replace
from functools import cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pixelflow as pf
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from mayaku import from_pretrained

OWNER = "enazif"
MODELS = {
    "mayaku-m-seg-street (medium, boxes and masks)": "mayaku-m-seg-street",
    "mayaku-m-det-street (medium, boxes)": "mayaku-m-det-street",
    "mayaku-n-seg-street (nano, boxes and masks)": "mayaku-n-seg-street",
    "mayaku-n-det-street (nano, boxes)": "mayaku-n-det-street",
}
PRIVATE = ["face", "license plate"]
# Faces and plates are blurred down to this score, whatever the display threshold:
# a missed blur costs more than an extra one.
PRIVATE_CONF = 0.2
EXAMPLES = sorted(str(p) for p in (Path(__file__).parent / "assets").glob("*.jpg"))
# Phone photos are 12+ MP: the model sees 800 px anyway, and a smaller result loads faster.
MAX_SIDE = 1920
# Past this many objects the labels cover the image: boxes and masks only.
MAX_LABELS = 40


def blur_private(frame: np.ndarray, detections) -> np.ndarray:
    """Blur each face and plate with a kernel half its size.

    PixelFlow's blur takes one kernel for all boxes and skips every box smaller
    than it, so one size either leaves small faces sharp or large plates readable.
    """
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        kernel = int(min(x2 - x1, y2 - y1)) // 2 | 1  # odd, below the box size
        one = pf.Detections()
        one.add_detection(det)
        frame = pf.annotators.blur(frame, one, kernel_size=max(kernel, 3))
    return frame


@cache
def load(name: str):
    folder = snapshot_download(f"{OWNER}/{name}")
    return from_pretrained(f"{folder}/{name}.pth", device="cpu")


def subset(result, keep):
    """The detections where `keep` is true (a mayaku `Detections`)."""
    fields = ("boxes", "scores", "labels", "masks")
    return replace(
        result, **{f: getattr(result, f)[keep] for f in fields if getattr(result, f) is not None}
    )


def infer(image: Image.Image | None, model: str, conf: float) -> dict | None:
    """Run the model once, down to the lowest score any drawing may need.

    The threshold slider and the blur checkbox then only redraw (`draw`), unless
    the slider goes below the score this ran at.
    """
    if image is None:
        return None
    predictor = load(MODELS[model])
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    floor = min(conf, PRIVATE_CONF)
    start = time.perf_counter()
    result = predictor(image, conf=floor)
    return {
        "frame": cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
        "result": result,
        "names": predictor.class_names,
        "ms": (time.perf_counter() - start) * 1000,
        "floor": floor,
    }


def draw(state: dict | None, image, model: str, conf: float, blur: bool):
    if state is not None and conf < state["floor"]:
        state = infer(image, model, conf)
    if state is None:
        return None, "", None
    result, names, frame = state["result"], state["names"], state["frame"].copy()
    # pixelflow 0.11's filter_by_class_id matches ids only, not class names
    is_private = torch.isin(result.labels, torch.tensor([names.index(n) for n in PRIVATE]))
    shown = pf.from_mayaku(subset(result, result.scores >= conf), labels=names)
    note = f"{len(shown)} objects in {state['ms']:.0f} ms on CPU"

    if blur:
        hidden = pf.from_mayaku(subset(result, is_private), labels=names)
        frame = blur_private(frame, hidden)
        note += f", {len(hidden)} faces and plates blurred"
    if len(shown):
        if result.masks is not None:
            frame = pf.annotators.mask(frame, shown)
        frame = pf.annotators.box(frame, shown)
        if len(shown) <= MAX_LABELS:
            frame = pf.annotators.label(frame, shown)

    counts = Counter(det.class_name for det in shown).most_common()
    summary = ", ".join(f"{n} {name}" for name, n in counts)
    text = f"{note}. {summary}" if summary else note
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), text, state


load(next(iter(MODELS.values())))  # the default model downloads at startup, not on first use

with gr.Blocks(title="Mayaku street models") as demo:
    gr.Markdown(
        "# Mayaku street models\n"
        "People, vehicles, e-scooters, faces, license plates and 15 more road classes, from "
        "dashcam to drone, with a model you use as it is. "
        "[Library](https://github.com/emrahnazif/mayaku) · "
        "[Docs](https://emrah.fr/mayaku/ready-to-use/street/) · "
        "Blurring with [PixelFlow](https://pypi.org/project/pixelflow/)"
    )
    state = gr.State()
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Image")
            model = gr.Dropdown(list(MODELS), value=next(iter(MODELS)), label="Model")
            conf = gr.Slider(0.05, 0.95, value=0.4, step=0.05, label="Score threshold")
            blur = gr.Checkbox(value=True, label="Blur faces and license plates")
        with gr.Column():
            out = gr.Image(label="Result")
            info = gr.Markdown()
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=inp, label="Examples: click one")
    # A new image or model runs the model; the threshold and the blur only redraw.
    redraw = (draw, [state, inp, model, conf, blur], [out, info, state])
    gr.on([inp.change, model.change], infer, [inp, model, conf], state).then(*redraw)
    gr.on([conf.release, blur.change], *redraw)

if __name__ == "__main__":
    demo.launch()
