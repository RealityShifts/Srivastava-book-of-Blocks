"""Interactive visualizer for Flax NNX and PyTorch models (TorchVista-style).

Traces one real forward pass, recovers the module graph with concrete shapes
and parameter counts, and renders it to a single self-contained HTML file.

Both frameworks share ``_core`` (the graph model, the source scan that finds
bare array ops, and the post-trace surgery) and the whole of ``renderer.py``.
Only the part that actually touches a framework is written twice, so an
improvement to the layout or the interaction lands on both at once.

Flax NNX::

    import jax
    from flax import nnx
    from Blocks.visualized import visualize

    model = MyModel(rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 128, 128, 3))
    visualize(model, x, path="model.html", open_browser=True)

PyTorch - any ``nn.Module``, not just the blocks in this repo. This writes two
files: the module-level graph at the given path, and torchvista's
operation-level graph beside it as ``rn18.torchvista.html``::

    import torch
    from Blocks.visualized import visualize_torch

    visualize_torch(resnet18(), torch.randn(1, 3, 224, 224), path="rn18.html")

In a notebook::

    from Blocks.visualized import visualize_notebook
    visualize_notebook(model, x)

From the command line::

    python -m Blocks.visualized.cli model.id_encoder:FeatResBlock \\
        --args 32 --input 5,128,128,32
"""

from .renderer import render_html, save_html
from .tracer import Edge, Graph, Node, Tensor, trace_model
from .visualize import summary, visualize, visualize_notebook


# The torch entry points are resolved on first use rather than imported here:
# this package is usable in a JAX-only environment (and the torch modules in a
# torch-only one), so importing either framework eagerly would break the other.
_TORCH_EXPORTS = {
    "visualize_torch": "visualize",
    "summary_torch": "summary",
    "trace_model_torch": "trace_model",
    "visualize_notebook_torch": "visualize_notebook",
    "visualize_torchvista": "visualize_torchvista",
    "torchvista_path_for": "torchvista_path_for",
}


def __getattr__(name: str):
    if name in _TORCH_EXPORTS:
        # ``import_module`` by full path, not ``from . import visualize_torch``:
        # the submodule shares its name with one of the exports above, so the
        # relative form re-enters this very function and recurses forever.
        from importlib import import_module
        mod = import_module(f"{__name__}.visualize_torch")
        return getattr(mod, _TORCH_EXPORTS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "visualize",
    "visualize_notebook",
    "summary",
    "trace_model",
    "visualize_torch",
    "summary_torch",
    "trace_model_torch",
    "visualize_notebook_torch",
    "visualize_torchvista",
    "torchvista_path_for",
    "render_html",
    "save_html",
    "Graph",
    "Node",
    "Edge",
    "Tensor",
]
