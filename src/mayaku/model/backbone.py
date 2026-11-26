"""Backbone: a stride-4 stem (and optionally a stride-4 stage), then three
bottleneck stages at strides 8 / 16 / 32, or four with a stride-64 stage."""

import torch.nn as nn

from mayaku.model.blocks import S32, Bottleneck, RepConv3x3, SPPFAdd, Stem


class Backbone(nn.Module):
    """Returns the stride-4 feature (the stride-4 mask feature reads it) and
    the stage outputs, fine to coarse; the coarsest passes through the
    concat-free spatial pyramid pooling. `Tier.s32_blocks` deepen the
    stride-32 output after the stride-64 stage has read it.

    `Tier.s4_depth` blocks after the stem make the stride-4 feature a stage
    of its own rather than the stem's raw output, and `Tier.expansion` sets
    every block's bottleneck: 4 runs each 3x3 at a quarter of the stage
    width, 2 at half.
    """

    def __init__(self, cfg):
        super().__init__()
        self.stem = Stem(cfg.stem, cfg.stem_mid)
        self.s4 = nn.Sequential(*[Bottleneck(cfg.stem, cfg.expansion)
                                  for _ in range(cfg.s4_depth)])
        chs = (cfg.stem, *tuple(cfg.width))
        self.down, self.stages = nn.ModuleList(), nn.ModuleList()
        for i, n in enumerate(cfg.depth):
            self.down.append(RepConv3x3(chs[i], chs[i + 1], 2))
            self.stages.append(nn.Sequential(
                *[Bottleneck(chs[i + 1], cfg.expansion) for _ in range(n)]))
        self.s32 = nn.Sequential(*[Bottleneck(cfg.width[S32], cfg.expansion)
                                   for _ in range(cfg.s32_blocks)])
        self.sppf = SPPFAdd(cfg.width[-1], cfg.global_ctx)

    def forward(self, x):
        x = stem = self.s4(self.stem(x))
        outs = []
        for down, stage in zip(self.down, self.stages, strict=True):
            x = stage(down(x))
            outs.append(x)
        outs[S32] = self.s32(outs[S32])
        outs[-1] = self.sppf(outs[-1])
        return stem, outs
