"""User-facing entry points for visualizing a PyTorch model.

Mirrors ``visualize.py`` (the Flax NNX side) call for call, so the two feel the
same. Kept in its own module so importing either one never drags in the other
framework: a torch-only environment can use this without JAX installed, and
vice versa.
"""

from __future__ import annotations

import webbrowser
from typing import Any, Optional

from torch import nn

from .renderer import render_html, save_html
from .tracer_torch import Graph, trace_model


def visualize(
    model: nn.Module,
    *args: Any,
    path: str = "model_graph.html",
    title: Optional[str] = None,
    open_browser: bool = False,
    **kwargs: Any,
) -> str:
    """Trace ``model(*args, **kwargs)`` and write an interactive HTML graph.

    Args:
        model: any ``torch.nn.Module``. Nothing here is specific to a
            particular library - the trace hooks ``nn.Module`` itself, so a
            torchvision ResNet, a HuggingFace transformer and a hand-written
            block are all handled the same way.
        *args: example inputs forwarded to ``model`` (real tensors - the pass
            runs eagerly so the recorded shapes are concrete).
        path: destination ``.html`` file.
        title: browser-tab title; defaults to the model class name.
        open_browser: open the file in the default browser when done.
        **kwargs: extra keyword arguments forwarded to ``model``.

    Returns:
        Absolute path to the written file.

    Example::

        import torch
        from torchvision.models import resnet18
        from Blocks.visualized.visualize_torch import visualize

        visualize(resnet18(), torch.randn(1, 3, 224, 224), path="resnet.html")
    """
    graph = trace_model(model, *args, model_name=title, **kwargs)
    out = save_html(graph, path, title=title)
    if open_browser:
        webbrowser.open("file://" + out)
    return out


def visualize_notebook(
    model: nn.Module,
    *args: Any,
    height: int = 720,
    title: Optional[str] = None,
    **kwargs: Any,
):
    """Render inline in Jupyter/Colab and return an ``IPython`` display object.

    The document is embedded in a sandboxed iframe via ``srcdoc``, so it needs
    no server and no temp file.
    """
    from IPython.display import HTML  # local import: optional dependency

    graph = trace_model(model, *args, model_name=title, **kwargs)
    doc = render_html(graph, title=title)
    escaped = doc.replace("&", "&amp;").replace('"', "&quot;")
    return HTML(
        f'<iframe srcdoc="{escaped}" style="width:100%;height:{height}px;'
        'border:1px solid #ccc;border-radius:8px" '
        'sandbox="allow-scripts"></iframe>'
    )


def summary(model: nn.Module, *args: Any, **kwargs: Any) -> str:
    """Return a plain-text tree of the traced model - handy in a terminal."""
    graph = trace_model(model, *args, **kwargs)
    lines = [
        f"{graph.model_name}  ({graph.total_params:,} params, "
        f"{len(graph.nodes)} modules)",
        "-" * 78,
    ]
    for n in graph.nodes:
        shape = f"{tuple(n.outputs[0].shape)}" if n.outputs else "-"
        lines.append(
            f"{'  ' * n.depth}{n.name:<24} {n.cls:<20} "
            f"{n.params:>10,}  {shape}"
        )
    if graph.error:
        lines += ["-" * 78, f"FAILED: {graph.error}"]
    return "\n".join(lines)


__all__ = ["visualize", "visualize_notebook", "summary", "trace_model", "Graph"]
