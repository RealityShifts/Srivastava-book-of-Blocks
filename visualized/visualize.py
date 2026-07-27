"""User-facing entry points for visualizing a Flax NNX model."""

from __future__ import annotations

import os
import webbrowser
from typing import Any, Optional

from flax import nnx

from .renderer import render_html, save_html
from .tracer import Graph, trace_model


def visualize(
    model: nnx.Module,
    *args: Any,
    path: str = "model_graph.html",
    title: Optional[str] = None,
    open_browser: bool = False,
    **kwargs: Any,
) -> str:
    """Trace ``model(*args, **kwargs)`` and write an interactive HTML graph.

    Args:
        model: the NNX module to visualize.
        *args: example inputs forwarded to ``model`` (real arrays - the pass
            runs eagerly so the recorded shapes are concrete).
        path: destination ``.html`` file.
        title: browser-tab title; defaults to the model class name.
        open_browser: open the file in the default browser when done.
        **kwargs: extra keyword arguments forwarded to ``model``.

    Returns:
        Absolute path to the written file.

    Example::

        from Blocks.visualized import visualize
        visualize(my_model, x, path="out/model.html")
    """
    graph = trace_model(model, *args, model_name=title, **kwargs)
    out = save_html(graph, path, title=title)
    if open_browser:
        webbrowser.open("file://" + out)
    return out


def visualize_notebook(
    model: nnx.Module,
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


def summary(model: nnx.Module, *args: Any, **kwargs: Any) -> str:
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
