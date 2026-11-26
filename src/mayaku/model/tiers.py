"""The architecture family: one frozen `Tier` per size, n / s / m / l."""

import dataclasses

from mayaku.model.blocks import LEVEL_STRIDES, per_level


@dataclasses.dataclass(frozen=True)
class Tier:
    """One member of the family. Frozen and hashable; vary one with
    `dataclasses.replace(TIERS["n"], seg=True)`."""

    stem: int
    width: tuple                    # backbone stage widths, fine to coarse; 3 or 4 levels
    depth: tuple                    # bottlenecks per stage, fine to coarse
    neck: tuple                     # neck width per level, fine to coarse
    paths: int                      # neck pathways
    head_width: int | tuple = 0     # head tower width, one or per level; 0 = the neck's width
    head_conv: int = 2              # 3x3 convolutions per head tower
    reg_max: int = 16               # DFL bins per box side
    # Auxiliary dense heads, off by default. Turning them on adds `aux.*`
    # weights only; see `load_weights`.
    seg: bool = False               # instance masks
    kpt: int = 0                    # keypoints per instance (0 = none)
    aux_arm: str = "box_tower"      # per-anchor aux outputs read from: "box_tower" | "tower"
    aux_width: int = 64             # trunk width of the aux branch (and of its own towers)
    mask_stride: int = 8            # stride of the mask feature: 8 (the trunk's) or 4
    # Backbone layout: `stem_mid` > 0 gives the two-conv stem 3 -> stem_mid
    # -> stem, `s4_depth` bottlenecks at stride 4, `expansion` the bottleneck
    # ratio of every block, `global_ctx` the global-context term in the
    # spatial pyramid pooling.
    stem_mid: int = 0
    s4_depth: int = 0
    expansion: int = 4
    global_ctx: bool = False
    # Oriented boxes: an angle readout per level, sides along the box's axes
    # (`mayaku.model.obb`). A detection task of its own, so not combined with
    # the mask / keypoint branches.
    obb: bool = False
    # Stride 32 deepened: `s32_blocks` bottlenecks on the stride-32 output
    # after the stride-64 stage has branched off it (the stride-64 stage reads
    # the plain stage output), and `s32_cls_block` a residual pair of 3x3s on
    # the stride-32 class tower, started as the identity.
    s32_blocks: int = 0
    s32_cls_block: bool = False

    def __post_init__(self):
        n = len(self.width)
        assert n in (3, 4) and len(self.depth) == len(self.neck) == n, \
            "one width, depth and neck width per level, 3 or 4 levels"
        assert min(per_level(self.head_width, n)) >= 0
        assert self.s32_blocks >= 0
        assert min(self.width) > 0 and min(self.depth) > 0 and min(self.neck) > 0
        assert self.paths >= 1
        assert self.head_conv >= 1
        assert self.aux_arm in ("box_tower", "tower"), self.aux_arm
        assert self.kpt >= 0 and self.aux_width >= 8
        assert self.mask_stride in (4, 8), self.mask_stride
        assert self.stem_mid >= 0 and self.s4_depth >= 0 and self.expansion in (1, 2, 4)
        assert not (self.obb and (self.seg or self.kpt)), \
            "oriented boxes are a task of their own: no masks or keypoints"

    @property
    def strides(self):
        """The detection strides, fine to coarse: 8 / 16 / 32, and 64 with a
        fourth level."""
        return LEVEL_STRIDES[:len(self.width)]

    @property
    def align(self):
        """What every canvas side must divide by: the coarsest stride."""
        return self.strides[-1]


# Widths are multiples of 32. Every tier adds a stride-64 level at stride 32's
# widths, deepens stride 32 by two backbone bottlenecks plus a residual BasicBlock
# in the stride-32 class tower, and widens the stride-32 / 64 heads. Canvases
# divide by 64 (`Tier.align`).
TIERS = {
    "n": Tier(stem=32, stem_mid=16, s4_depth=2, width=(96, 192, 320, 320), depth=(3, 3, 2, 2),
              expansion=2, global_ctx=True, neck=(64, 128, 256, 256), paths=3,
              head_width=(64, 80, 128, 128), s32_blocks=2, s32_cls_block=True),
    "s": Tier(stem=32, stem_mid=16, s4_depth=2, width=(192, 320, 512, 512), depth=(3, 4, 2, 2),
              expansion=2, global_ctx=True, neck=(128, 224, 384, 384), paths=3,
              head_width=(96, 128, 192, 192), s32_blocks=2, s32_cls_block=True),
    "m": Tier(stem=64, stem_mid=32, s4_depth=2, width=(288, 448, 768, 768), depth=(4, 4, 3, 2),
              expansion=2, global_ctx=True, neck=(192, 320, 576, 576), paths=3,
              head_width=(160, 192, 288, 288), s32_blocks=2, s32_cls_block=True),
    "l": Tier(stem=64, stem_mid=32, s4_depth=2, width=(384, 576, 1024, 1024), depth=(4, 6, 3, 2),
              expansion=2, global_ctx=True, neck=(256, 384, 768, 768), paths=3,
              head_width=(224, 256, 384, 384), s32_blocks=2, s32_cls_block=True),
}

# Not a product tier: a few tens of thousands of parameters, so a test can
# train a real network end to end in seconds.
TINY = Tier(stem=8, width=(16, 32, 64), depth=(1, 1, 1), neck=(16, 16, 16),
            paths=1, head_conv=1, reg_max=8)
