"""User-facing entry points for visualizing a PyTorch model.

Mirrors ``visualize.py`` (the Flax NNX side) call for call, so the two feel the
same. Kept in its own module so importing either one never drags in the other
framework: a torch-only environment can use this without JAX installed, and
vice versa.

Two graphs, not one
-------------------
``visualize()`` writes **two** files, because the torch ecosystem has two
tracers that answer different questions and disagreeing with yourself is
informative:

``<name>.html`` - this package
    Same renderer the Flax side uses, so a torch graph and its JAX twin open in
    the same viewer and compare block for block. Traced at *module*
    granularity: one pill per ``nn.Module`` call, parameter counts rolled up.

``<name>.torchvista.html`` - torchvista
    An independent trace at *operation* granularity: it follows ``torch``
    functions and tensor methods, not just module boundaries, so it shows the
    bare ops (the ``+ x`` in a residual, a ``cat``, a ``grid_sample``) that a
    module-level trace can only infer. Strongest exactly where the module view
    is thinnest.

torchvista is an optional dependency; without it the first graph is still
written and a note is printed. Pass ``torchvista=False`` to skip it.
"""

from __future__ import annotations

import contextlib
import io
import webbrowser
from pathlib import Path
from typing import Any, Optional

from torch import nn

from .prune_torch import prune_passthrough
from .renderer import render_html, save_html
from .tracer_torch import Graph, trace_model

TORCHVISTA_SUFFIX = ".torchvista.html"


def torchvista_path_for(path: str) -> str:
    """Where :func:`visualize` puts the torchvista graph for ``path``."""
    text = str(path)
    stem = text[: -len(".html")] if text.endswith(".html") else text
    return stem + TORCHVISTA_SUFFIX


def visualize_torchvista(
    model: nn.Module,
    *args: Any,
    path: str,
    title: Optional[str] = None,
    collapse_modules_after_depth: int = 2,
    **kwargs: Any,
) -> Optional[str]:
    """Write torchvista's operation-level graph. ``None`` if it isn't installed.

    The file is post-processed into a standalone document - see
    :func:`_wrap_as_document` for why that is required, not cosmetic.

    ``collapse_modules_after_depth`` defaults to 2 rather than torchvista's own
    1: at depth 1 a composed model collapses to a handful of opaque boxes,
    which is less than the module graph already shows.
    """
    try:
        from torchvista import trace_model as tv_trace
    except ImportError:
        return None

    payload = args[0] if len(args) == 1 else tuple(args)
    # trace_model targets notebooks: it returns an IPython display object and
    # prints it. Swallow that so a terminal run stays readable.
    with contextlib.redirect_stdout(io.StringIO()):
        tv_trace(model, payload, export_format="html", export_path=str(path),
                 collapse_modules_after_depth=collapse_modules_after_depth,
                 **kwargs)
    if not Path(path).is_file():
        return None
    _wrap_as_document(Path(path), title=title)
    return str(path)


_DOCUMENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; min-height: 100%; background: #fff; }}
</style>
</head>
<body>
{fragment}
</body>
</html>
"""


def _wrap_as_document(path: Path, title: str = "torchvista") -> None:
    """Turn torchvista's notebook fragment into a standalone HTML document.

    ``torchvista``'s file export writes the *same* string it hands to Jupyter's
    ``display(HTML(...))``: a bare ``<div>`` with no doctype, no ``<html>`` and
    no ``<head>``. That is right for a notebook cell, which already supplies the
    document, and wrong for a file you open directly.

    The part that actually breaks is the missing charset. The export inlines
    d3, Viz.js and jsoneditor - about 2.7 MB of JavaScript containing ~11k
    non-ASCII bytes - and a local file with no ``<meta charset>`` is decoded as
    windows-1252, not UTF-8. Those multi-byte sequences turn to mojibake inside
    the bundled scripts, the parse fails, and the page renders blank.

    Idempotent: a file already carrying a doctype is left alone, so re-running
    over an existing graph does not nest documents.
    """
    text = path.read_text(encoding="utf-8")
    if text.lstrip()[:15].lower().startswith("<!doctype"):
        return
    path.write_text(_DOCUMENT.format(title=title, fragment=text),
                    encoding="utf-8")


def visualize(
    model: nn.Module,
    *args: Any,
    path: str = "model_graph.html",
    title: Optional[str] = None,
    open_browser: bool = False,
    prune: bool = True,
    torchvista: bool = True,
    collapse_modules_after_depth: int = 2,
    **kwargs: Any,
) -> str:
    """Trace ``model(*args, **kwargs)`` and write interactive HTML graphs.

    Args:
        model: any ``torch.nn.Module``. Nothing here is specific to a
            particular library - the trace hooks ``nn.Module`` itself, so a
            torchvision ResNet, a HuggingFace transformer and a hand-written
            block are all handled the same way.
        *args: example inputs forwarded to ``model`` (real tensors - the pass
            runs eagerly so the recorded shapes are concrete).
        path: destination ``.html`` file for the module-level graph. The
            torchvista graph goes beside it, at
            ``<name>``:data:`TORCHVISTA_SUFFIX`.
        title: browser-tab title; defaults to the model class name.
        open_browser: open the module-level graph when done.
        prune: drop parameter-free, shape-preserving pills (activations,
            ``Identity``) that the Flax tracer never sees - see
            ``prune_torch``. The dropped activation is recorded on its parent,
            so nothing is lost.
        torchvista: also write the operation-level graph.
        **kwargs: extra keyword arguments forwarded to ``model``.

    Returns:
        Absolute path to the module-level graph. Use
        :func:`torchvista_path_for` for the other one.

    Example::

        import torch
        from torchvision.models import resnet18
        from Blocks.visualized.visualize_torch import visualize

        visualize(resnet18(), torch.randn(1, 3, 224, 224), path="resnet.html")
    """
    was_training = model.training
    model.eval()
    try:
        graph = trace_model(model, *args, model_name=title, **kwargs)
        if prune:
            prune_passthrough(graph)
        out = save_html(graph, path, title=title)

        if torchvista:
            tv = visualize_torchvista(
                model, *args, path=torchvista_path_for(path),
                title=title or type(model).__name__,
                collapse_modules_after_depth=collapse_modules_after_depth)
            if tv is None:
                print("  (torchvista not installed - skipped the op-level "
                      "graph; pip install torchvista)")
    finally:
        model.train(was_training)

    if open_browser:
        webbrowser.open("file://" + out)
    return out


def visualize_notebook(
    model: nn.Module,
    *args: Any,
    height: int = 720,
    title: Optional[str] = None,
    prune: bool = True,
    **kwargs: Any,
):
    """Render inline in Jupyter/Colab and return an ``IPython`` display object.

    The document is embedded in a sandboxed iframe via ``srcdoc``, so it needs
    no server and no temp file. For torchvista's own inline view, call
    ``torchvista.trace_model(model, inputs)`` directly - it already renders in
    a notebook without an export step.
    """
    from IPython.display import HTML  # local import: optional dependency

    graph = trace_model(model, *args, model_name=title, **kwargs)
    if prune:
        prune_passthrough(graph)
    doc = render_html(graph, title=title)
    escaped = doc.replace("&", "&amp;").replace('"', "&quot;")
    return HTML(
        f'<iframe srcdoc="{escaped}" style="width:100%;height:{height}px;'
        'border:1px solid #ccc;border-radius:8px" '
        'sandbox="allow-scripts"></iframe>'
    )


def summary(model: nn.Module, *args: Any, prune: bool = True,
            **kwargs: Any) -> str:
    """Return a plain-text tree of the traced model - handy in a terminal."""
    graph = trace_model(model, *args, **kwargs)
    if prune:
        prune_passthrough(graph)
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


__all__ = [
    "visualize",
    "visualize_notebook",
    "visualize_torchvista",
    "torchvista_path_for",
    "summary",
    "trace_model",
    "Graph",
    "TORCHVISTA_SUFFIX",
]
