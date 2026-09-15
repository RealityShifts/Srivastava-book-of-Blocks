"""Runnable PyTorch examples: ``python -m Blocks.visualized.examples_torch [outdir]``.

The Flax examples live in ``examples.py``; this is the torch twin, and it needs
no JAX installed. Each example writes **two** files:

    <name>.html              module-level graph (this package)
    <name>.torchvista.html   operation-level graph (torchvista, if installed)

Read in order, these demonstrate the three things that actually trip people up:

  1. ``collapse_modules_after_depth`` -- how much of the hierarchy to show.
  2. **Multi-input models take VARARGS, not a tuple.** This is the one that
     silently produces a useless picture rather than an error (example 2).
  3. A model that raises mid-forward still renders everything traced up to the
     failure, which is what makes this useful for debugging (example 4).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from .visualize_torch import summary, visualize


class ConvBlock(nn.Module):
    """conv -> norm -> activation, the unit the graph should show as one box."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Stage(nn.Module):
    """Two blocks plus a residual add.

    Grouping a stage into its own module is the point of the example: the add
    below is a *bare op*, and a bare op is drawn inside whichever module ran it.
    Written as a flat loop in the parent instead, every add from every stage
    would pile up at the parent's level with nothing to say which stage it
    belonged to -- which is exactly how a readable graph turns into a hairball.
    """

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.block1 = ConvBlock(cin, cout, stride=2)
        self.block2 = ConvBlock(cout, cout)
        self.skip = nn.Conv2d(cin, cout, 1, stride=2, bias=False)

    def forward(self, x):
        return self.block2(self.block1(x)) + self.skip(x)


class TinyEncoder(nn.Module):
    """Three stages: enough depth that collapsing actually changes the picture."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBlock(3, 32)
        self.stage1 = Stage(32, 64)
        self.stage2 = Stage(64, 128)
        self.head = nn.Conv2d(128, 64, 1)

    def forward(self, x):
        return self.head(self.stage2(self.stage1(self.stem(x))))


class TwoInputModel(nn.Module):
    """Takes two tensors -- the shape that makes the varargs rule matter."""

    def __init__(self) -> None:
        super().__init__()
        self.enc_a = ConvBlock(3, 32)
        self.enc_b = ConvBlock(3, 32)
        self.fuse = ConvBlock(64, 32)

    def forward(self, a, b):
        return self.fuse(torch.cat([self.enc_a(a), self.enc_b(b)], dim=1))


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    outdir = argv[0] if argv else "visualized_examples_torch"
    os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)  # noqa: E731

    # eval(): tracing runs a REAL forward pass, so a model left in train mode
    # would update BatchNorm running stats and sample dropout as a side effect
    # of drawing a picture.
    model = TinyEncoder().eval()
    img = torch.randn(1, 3, 64, 64)

    # 1. text summary first -- often all you need, and it costs one line.
    print(summary(model, img), "\n")

    # 2. the collapse control. `collapse_modules_after_depth` is the knob:
    #    0  = maximum collapse, one box per top-level child (stem/stage1/...)
    #    2  = the default; stages open up, leaf convs stay closed
    #    99 = everything, down to individual ops
    #    Start at 0 on an unfamiliar model and open up only where you care --
    #    a large model drawn fully expanded is unreadable and slow to render.
    for depth, label in ((0, "collapsed"), (2, "default"), (99, "expanded")):
        print(visualize(model, img, path=out(f"tiny_encoder_{label}.html"),
                        title=f"TinyEncoder ({label}, depth={depth})",
                        collapse_modules_after_depth=depth))

    # 3. MULTI-INPUT: pass the tensors as separate arguments.
    #
    #        visualize(model, a, b)      correct
    #        visualize(model, (a, b))    WRONG -- one tuple, not two tensors
    #
    #    The wrong form does not raise. The tracer catches the TypeError from
    #    forward(), records it on the graph, and writes a file containing a
    #    single error node -- a picture that looks merely empty rather than
    #    broken. If a graph comes out with one node, check this first and read
    #    `trace_model(...).error`.
    two = TwoInputModel().eval()
    a, b = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    print(visualize(two, a, b, path=out("two_input.html"),
                    title="TwoInputModel (varargs)",
                    collapse_modules_after_depth=0))

    # 4. a model that raises mid-forward: THIS package's tracer records the
    #    error on the graph and still draws everything traced up to it, which
    #    is usually faster than a traceback for finding a shape mismatch.
    #
    #    torchvista does not do that -- it lets the exception propagate -- so
    #    `torchvista=False` here. On a model you expect to fail, ask for the
    #    module-level graph only.
    class Broken(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = ConvBlock(3, 16)
            self.b = ConvBlock(99, 32)          # wrong in_channels

        def forward(self, x):
            return self.b(self.a(x))

    print(visualize(Broken().eval(), torch.randn(1, 3, 32, 32),
                    path=out("broken_model.html"),
                    title="Broken (partial graph)",
                    torchvista=False))

    print(f"\nopen the files in {os.path.abspath(outdir)}")
    print("each <name>.html has a <name>.torchvista.html beside it "
          "(pip install torchvista if those are missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
