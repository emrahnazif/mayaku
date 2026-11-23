"""Build, and optionally upload, the Hugging Face repos of the street models.

Each model gets its own repo, ``enazif/<name>``, holding:

- ``README.md``: the model card, rendered from ``hf/models/street/card.md``
- ``config.json``: task, input size, prediction settings and class names, read
  from the checkpoint (Hugging Face also counts downloads by this file)
- ``<name>.pth``: the checkpoint, fetched from the hosted manifest with its
  sha256 checked, so Hugging Face serves the same bytes as ``from_pretrained``

Run from the repo root:

    python tools/hf_upload.py --out /tmp/hf-street            # build only, to review
    python tools/hf_upload.py --out /tmp/hf-street --upload   # build, then upload
    python tools/hf_upload.py --space mayaku-street           # upload hf/spaces/mayaku-street/

Uploading needs ``hf auth login`` with a write token. The card text lives in
``hf/models/street/card.md`` and each Space in ``hf/spaces/<name>/``; edit them there, never on the Hugging Face website.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mayaku.inference.decode import predict_decode
from mayaku.utils.checkpoint import load_checkpoint
from mayaku.utils.download import download_model

OWNER = "enazif"
HF_DIR = Path(__file__).resolve().parent.parent / "hf"
CARD = HF_DIR / "models" / "street" / "card.md"
# Each folder here is one Space, uploaded to <OWNER>/<folder name>.
SPACES = HF_DIR / "spaces"

# name -> (params, box AP, mask AP, RTX 3060 fp16 ms, int8 ms). These numbers, and
# the tables and prose in hf/models/street/card.md, come from docs/ready-to-use/street.md:
# when the models are re-measured, update all three.
MODELS = {
    "mayaku-n-det-street": ("5.2 M", "43.9", "", "2.61", "2.11"),
    "mayaku-n-seg-street": ("5.3 M", "43.9", "35.2", "5.54", "5.64"),
    "mayaku-m-det-street": ("32.4 M", "56.8", "", "6.42", "3.94"),
    "mayaku-m-seg-street": ("32.7 M", "56.8", "45.5", "10.4", "7.86"),
}

# tier -> (word, who it is for)
TIERS = {
    "n": ("nano", "for CPUs, edge devices and many streams per GPU."),
    "m": (
        "medium",
        "for when accuracy matters, especially for small and distant objects and aerial views.",
    ),
}


def tier(name: str) -> str:
    return name.split("-")[1]


def is_seg(name: str) -> bool:
    return "-seg-" in name


def gives(name: str) -> str:
    return "boxes and masks" if is_seg(name) else "boxes"


def family_table(current: str) -> str:
    rows = [
        "| Model | Size | Gives | Box AP | Mask AP | RTX 3060, fp16 | RTX 3060, int8 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, (params, box, mask, fp16, int8) in MODELS.items():
        label = (
            f"**{name}** (this model)"
            if name == current
            else f"[{name}](https://huggingface.co/{OWNER}/{name})"
        )
        size = f"{TIERS[tier(name)][0]}, {params}"
        rows.append(
            f"| {label} | {size} | {gives(name)} | {box} | {mask} | {fp16} ms | {int8} ms |"
        )
    return "\n".join(rows)


def render_card(name: str, cfg: dict) -> str:
    word, pitch = TIERS[tier(name)]
    height, width = cfg["input_size"]
    values = {
        "name": name,
        "pipeline_tag": "image-segmentation" if is_seg(name) else "object-detection",
        "task_tag": cfg["task"],
        "tier_word": word,
        "gives": gives(name),
        "tier_pitch": pitch,
        "family_table": family_table(name),
        "width": str(width),
        "height": str(height),
        "conf": f"{cfg['predict']['score']:g}",
        "iou": f"{cfg['predict']['nms_iou']:g}",
    }
    text = CARD.read_text()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        raise ValueError(f"unfilled placeholder in {CARD}")
    return text


def config(name: str, ckpt: Path) -> dict:
    sidecar, _ = load_checkpoint(ckpt)
    if sidecar is None:
        raise ValueError(f"{ckpt} has no mayaku sidecar")
    predict = predict_decode(sidecar)  # the checkpoint's settings, else the library's
    return {
        "library_name": "mayaku",
        "model": name,
        "task": "instance-segmentation"
        if sidecar["config"]["model"]["seg"]
        else "object-detection",
        "tier": sidecar["config"]["model"]["tier"],
        "input_size": sidecar["canvas_hw"],
        "predict": {"score": predict.conf, "nms_iou": predict.iou},
        "num_classes": len(sidecar["class_names"]),
        "class_names": sidecar["class_names"],
        "weights": f"{name}.pth",
        "requires": "mayaku>=3.0",
    }


def build(out: Path) -> list[Path]:
    folders = []
    for name in MODELS:
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        ckpt = download_model(name, cache_dir=folder)  # sha256-checked on download
        cfg = config(name, ckpt)
        (folder / "README.md").write_text(render_card(name, cfg))
        (folder / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"built {folder}")
        folders.append(folder)
    return folders


def push(repo: str, folder: Path, repo_type: str = "model", **create) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type=repo_type, exist_ok=True, **create)
    api.upload_folder(
        repo_id=repo,
        repo_type=repo_type,
        folder_path=folder,
        commit_message="Update",
        ignore_patterns=["__pycache__/*"],
    )
    prefix = "spaces/" if repo_type == "space" else ""
    print(f"uploaded https://huggingface.co/{prefix}{repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--space",
        choices=sorted(p.name for p in SPACES.iterdir() if p.is_dir()),
        help="upload this Space from hf/spaces/ instead of the models",
    )
    parser.add_argument("--out", type=Path, help="where to build the repo folders")
    parser.add_argument(
        "--upload", action="store_true", help="upload to Hugging Face after building"
    )
    args = parser.parse_args()
    if args.space:
        push(f"{OWNER}/{args.space}", SPACES / args.space, repo_type="space", space_sdk="gradio")
        return
    if args.out is None:
        parser.error("--out is required unless --space is given")
    folders = build(args.out)
    if args.upload:
        for folder in folders:
            push(f"{OWNER}/{folder.name}", folder)


if __name__ == "__main__":
    main()
