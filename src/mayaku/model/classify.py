"""The classifier: the detector's backbone and a class-map head, five ops.

The head is a 1x1 convolution to `HIDDEN` channels with BatchNorm and ReLU
on the backbone's stride-32 map, then a 1x1 convolution to the classes: a
per-location MLP whose output is a class map. The graph stops there; the host
takes the map's spatial mean and a softmax. The mean of a 1x1 convolution is
the 1x1 convolution of the mean, so this is exactly global average pooling
followed by a linear classifier (the MobileNetV3 / YOLO-cls head, conv to a
wide feature then pool then linear), without the global pooling the five-op
contract has no op for.
"""

import copy

import torch.nn as nn
import torch.nn.functional as F

from mayaku.model.backbone import Backbone
from mayaku.model.blocks import ConvBNReLU, fuse_tree

HIDDEN = 1280       # the head's feature width (MobileNetV3's and YOLO-cls's)
DROPOUT = 0.2       # on that feature, in training only


class Classifier(nn.Module):
    """`forward` returns a one-element list, the (B, nc, H/32, W/32) class
    logit map, so the trainer, the exporter and the runners see the same
    flat-output interface as the detector's."""

    task = "classify"

    def __init__(self, cfg, nc=1000, canvas=224):
        super().__init__()
        self.cfg, self.nc = cfg, nc
        self.backbone = Backbone(cfg)
        self.hidden = ConvBNReLU(cfg.width[-1], HIDDEN)
        self.logits = nn.Conv2d(HIDDEN, nc, 1)
        nn.init.normal_(self.logits.weight, std=0.01)
        nn.init.zeros_(self.logits.bias)

    @property
    def out_names(self):
        return ["logits"]

    def split(self, preds):
        return {"logits": preds[0]}

    def readouts(self):
        return [self.logits]

    def forward(self, x):
        _, bb = self.backbone(x)
        h = self.hidden(bb[-1])
        # functional, so the eval graph (and the export) holds no dropout node
        return [self.logits(F.dropout(h, DROPOUT, self.training))]

    def fuse(self):
        return fuse_tree(self.eval())

    def for_deploy(self):
        from mayaku.model.quant import strip_fake_quant

        return strip_fake_quant(copy.deepcopy(self).fuse())

    def deploy_spec(self):
        from mayaku.model.quant import activation_scheme, is_qat

        return {"task": "classify", "outputs": self.out_names, "strides": [32], "reg_max": 0,
                "qat": is_qat(self), "activations": activation_scheme(self),
                "mask": None, "keypoints": None, "obb": None}


def logits(preds):
    """The class map -> (B, nc) logits: its spatial mean (the host's global
    average pooling)."""
    return preds[0].float().mean((2, 3))


def load_backbone(model, state):
    """Warm-start a classifier: the backbone from any mayaku checkpoint of the
    same tier (a detector's or a classifier's), all of it or fail; the head
    too when the state is a classifier's with the same class count, else the
    head starts fresh. Returns {"head": loaded?}."""
    own = model.state_dict()
    back = {k: v for k, v in state.items() if k.startswith("backbone.")}
    missing = [k for k in own if k.startswith("backbone.") and k not in back]
    wrong = [k for k, v in back.items() if k not in own or own[k].shape != v.shape]
    assert not missing and not wrong, \
        "backbone does not fit this tier: %s" % (missing[:4] + wrong[:4])
    head = {k: v for k, v in state.items() if k.startswith(("hidden.", "logits."))}
    fits = bool(head) and all(k in own and own[k].shape == v.shape for k, v in head.items())
    model.load_state_dict({**back, **(head if fits else {})}, strict=False)
    return {"head": fits}
