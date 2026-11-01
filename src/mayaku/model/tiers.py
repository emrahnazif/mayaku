"""The architecture family: one frozen `Tier` per size, n / s / m / l."""

import dataclasses

from mayaku.model.blocks import STRIDES


@dataclasses.dataclass(frozen=True)
class Tier:
    """One member of the family. Frozen and hashable; vary one with
    `dataclasses.replace(TIERS["n"], seg=True)`."""

    stem: int
    width: tuple                    # backbone stage widths, fine to coarse
    depth: tuple                    # bottlenecks per stage, fine to coarse
    neck: tuple                     # neck width per level, fine to coarse
    paths: int                      # neck pathways
    head_width: int = 0             # head tower width; 0 = the neck's width per level
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

    def __post_init__(self):
        n = len(STRIDES)
        assert len(self.width) == len(self.depth) == len(self.neck) == n, \
            "one width, depth and neck width per stride in %s" % (STRIDES,)
        assert min(self.width) > 0 and min(self.depth) > 0 and min(self.neck) > 0
        assert self.paths >= 1
        assert self.head_width >= 0 and self.head_conv >= 1
        assert self.aux_arm in ("box_tower", "tower"), self.aux_arm
        assert self.kpt >= 0 and self.aux_width >= 8
        assert self.mask_stride in (4, 8), self.mask_stride
        assert self.stem_mid >= 0 and self.s4_depth >= 0 and self.expansion in (1, 2, 4)


# Widths are multiples of 32. n and m spend about half their compute in the
# backbone, at strides 4-16: a two-conv stem into a stride-4 stage, the 3x3s
# at half width, and global context in the pyramid pooling. s and l push
# depth and width toward the coarse stages, where parameters are cheap to
# run, with the bottleneck block making backbone depth cheap enough to spend.
TIERS = {
    # 6.7 GMACs at 640, the backbone 41% of them
    "n": Tier(stem=32, stem_mid=16, s4_depth=2, width=(96, 192, 320), depth=(3, 3, 2),
              expansion=2, global_ctx=True, neck=(64, 128, 256), paths=3, head_width=80),
    "s": Tier(stem=32, width=(144, 288, 576), depth=(3, 5, 7),
              neck=(96, 192, 384), paths=3, head_width=96),
    # stride 8 / 16 wider and depth moved off stride 32; a three-path neck
    # with stride 32 at 576 (narrower lost large objects). 46.3 GMACs at 640,
    # the backbone 46% of them
    "m": Tier(stem=64, stem_mid=32, s4_depth=2, width=(288, 448, 768), depth=(4, 4, 3),
              expansion=2, global_ctx=True, neck=(192, 320, 576), paths=3, head_width=192),
    "l": Tier(stem=64, width=(256, 512, 1024), depth=(4, 8, 10),
              neck=(256, 512, 1024), paths=5, head_width=256),
}

# Not a product tier: a few tens of thousands of parameters, so a test can
# train a real network end to end in seconds.
TINY = Tier(stem=8, width=(16, 32, 64), depth=(1, 1, 1), neck=(16, 16, 16),
            paths=1, head_conv=1, reg_max=8)
