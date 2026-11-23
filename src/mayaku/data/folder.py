"""Image classification data: one folder per class.

    root/<class name>/<image>

the layout torchvision's ImageFolder reads and most tools export. Classes are
the sub-folders in sorted order. Training samples are a random-resized crop
(Szegedy et al. 2015, arXiv 1409.4842; the area and aspect ranges are
torchvision's defaults bar the area floor), flips and colour jitter;
evaluation samples are the whole image letterboxed onto the canvas, the same
letterbox the detector and every runtime use, so a classifier deploys through
the one preprocessing contract.
"""

import dataclasses
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mayaku.data.augment import apply_hsv, sample_hsv
from mayaku.data.batch import to_tensor
from mayaku.data.geometry import letterbox
from mayaku.model.blocks import as_canvas

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
# random-resized-crop: the crop's share of the image area and its aspect.
# torchvision's area floor is 0.08, tuned for 90+ ImageNet epochs from
# scratch; fine-tunes on a few epochs keep more of the object.
CROP_AREA = (0.25, 1.0)
CROP_ASPECT = (3 / 4, 4 / 3)


@dataclasses.dataclass
class FolderLabels:
    """A parsed class-folder split: the class names, and each image's file
    and class index."""

    root: str
    class_names: list
    files: list
    labels: np.ndarray

    @property
    def cat_ids(self):
        return list(range(len(self.class_names)))


def load_folder(root, class_names=None):
    """Parse `root/<class>/<image>`. `class_names` (a training split's) fixes
    the class order for a validation split, which must not hold a class the
    training split lacks."""
    found = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    names = list(class_names) if class_names is not None else found
    extra = sorted(set(found) - set(names))
    if extra:
        raise ValueError(f"{root} has classes the model does not know: {extra[:5]}")
    files, labels = [], []
    for c, name in enumerate(names):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMAGE_EXT):
                files.append(os.path.join(d, f))
                labels.append(c)
    if not files:
        raise ValueError(f"no images under {root}/<class>/")
    return FolderLabels(str(root), names, files, np.asarray(labels, np.int64))


class FolderDataset(Dataset):
    """Images and their class from a class-folder split, on an (H, W) canvas.

    Items are the detection dataset's: {"img", "labels", "meta"}, the labels
    a (1, 1) [class] table, so the trainer's collate, recalibration and
    loaders take either. `aug` None is evaluation; an `Augment` trains, using
    its flips and colour jitter, and a random-resized crop unless its scale
    range is (1, 1) (the clean stage), when the whole image is letterboxed.
    """

    def __init__(self, root, canvas=224, aug=None, seed=0, labels=None):
        self.canvas = as_canvas(canvas)
        self.aug = aug
        self.rng = random.Random(seed)
        self.data = labels or load_folder(root)

    class_names = property(lambda self: self.data.class_names)
    cat_ids = property(lambda self: self.data.cat_ids)

    @property
    def nc(self):
        return len(self.data.class_names)

    @property
    def ids(self):
        return list(range(len(self)))

    def __len__(self):
        return len(self.data.files)

    def read(self, i):
        return cv2.imread(self.data.files[i])

    def load(self, i):
        img = self.read(i)
        if img is None:
            return None, None, None
        return letterbox(img, self.canvas)

    def _crop(self, img):
        """A random crop of `CROP_AREA` of the image at an aspect in
        `CROP_ASPECT` (log-uniform), the whole image after ten misses."""
        h, w = img.shape[:2]
        for _ in range(10):
            area = h * w * self.rng.uniform(*CROP_AREA)
            ar = math.exp(self.rng.uniform(*np.log(CROP_ASPECT)))
            cw, ch = round(math.sqrt(area * ar)), round(math.sqrt(area / ar))
            if 0 < cw <= w and 0 < ch <= h:
                x, y = self.rng.randint(0, w - cw), self.rng.randint(0, h - ch)
                return img[y:y + ch, x:x + cw]
        return img

    def __getitem__(self, i):
        img = self.read(i)
        while img is None:                      # a missing or corrupt file
            assert self.aug is not None, "eval image missing: %s" % self.data.files[i]
            i = self.rng.randrange(len(self))
            img = self.read(i)
        label = torch.tensor([[float(self.data.labels[i])]])
        if self.aug is None:
            shape = img.shape[:2]
            img, r, pad = letterbox(img, self.canvas)
            return {"img": to_tensor(img), "labels": label,
                    "meta": {"id": i, "ratio": r, "pad": pad, "shape": tuple(shape)}}
        aug = self.aug
        if tuple(aug.scale) != (1.0, 1.0):
            img = self._crop(img)
        if aug.hflip and self.rng.random() < aug.hflip:
            img = img[:, ::-1]
        if aug.vflip and self.rng.random() < aug.vflip:
            img = img[::-1]
        img = apply_hsv(np.ascontiguousarray(img), sample_hsv(self.rng, aug.hsv))
        img = letterbox(img, self.canvas)[0]
        return {"img": to_tensor(img), "labels": label, "meta": None}
