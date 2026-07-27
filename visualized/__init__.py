"""Interactive visualizer for Flax NNX models (a TorchVista-style view).

Traces one real forward pass, recovers the module graph with concrete shapes
and parameter counts, and renders it to a single self-contained HTML file.

Quick start::

    import jax
    from flax import nnx
    from Blocks.visualized import visualize

    model = MyModel(rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 128, 128, 3))
    visualize(model, x, path="model.html", open_browser=True)

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

__all__ = [
    "visualize",
    "visualize_notebook",
    "summary",
    "trace_model",
    "render_html",
    "save_html",
    "Graph",
    "Node",
    "Edge",
    "Tensor",
]
