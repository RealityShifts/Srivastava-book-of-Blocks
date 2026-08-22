"""Trace a PyTorch model's forward pass into the same graph the JAX tracer builds.

The graph model, the source scan that recovers bare array ops, and all the
post-trace surgery live in ``_core`` and are shared verbatim with the Flax NNX
tracer - only the parts that actually touch the framework are written twice.
That is what lets one renderer serve both, and it is why fixing a layout or
edge-stitching bug fixes it for PyTorch and JAX at once.

Strategy: ``torch.fx`` and ``jit.trace`` both flatten a model to primitives and
lose the module boundaries that make a picture readable, so instead we attach
forward hooks to every submodule and run one real forward pass, recording:

    - the module hierarchy (from ``named_modules``, so names are real attribute
      paths like ``encoder.blocks.0.conv``)
    - call order and nesting depth
    - concrete input/output shapes and dtypes
    - dataflow edges, recovered by matching ``id()`` of the tensors a module
      consumed against tensors previous modules produced

Hooks rather than monkeypatching: PyTorch calls ``forward`` through
``Module.__call__``, which already fires ``forward_pre_hook`` and
``forward_hook`` around it. They see the same arguments and return value a
wrapper would, they nest correctly, they fire once per *call* so a tied module
invoked twice is recorded twice, and they are removed by handle rather than by
restoring patched classes - no chance of leaving a model mutated if the pass
raises.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import torch
from torch import nn
from torch.utils import _pytree as pytree

try:                                   # pragma: no cover - older torch
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:                    # counter is optional; -1 means unmeasured
    FlopCounterMode = None

from ._core import (
    Edge,
    Graph,
    INPUT_NODE,
    IN_BASE,
    Node,
    Tensor,
    _scan_ops,
    finalize_graph,
)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def _leaves(obj: Any) -> list:
    """Every tensor inside a nested tuple/list/dict, in a stable order.

    ``torch.utils._pytree`` is the same flattening the framework itself uses
    for structured inputs and outputs, so a module returning ``(out, aux)`` or
    a dict of feature maps is handled the way PyTorch would handle it.
    """
    try:
        leaves = pytree.tree_leaves(obj)
    except Exception:                      # pragma: no cover - exotic types
        leaves = [obj]
    return [l for l in leaves if isinstance(l, torch.Tensor)]


def _as_tensors(obj: Any) -> list:
    """Flatten a structure into the concrete tensors it holds."""
    return [Tensor(tuple(t.shape), _dtype(t)) for t in _leaves(obj)]


def _dtype(t: torch.Tensor) -> str:
    """``torch.float32`` reads as noise in a card; show ``float32``."""
    return str(t.dtype).replace("torch.", "")


def _array_ids(obj: Any) -> list:
    """``id()`` of every tensor leaf - used to stitch dataflow edges."""
    return [id(t) for t in _leaves(obj)]


def _count_params(module: nn.Module) -> int:
    """Total parameter elements owned by ``module`` and its children.

    ``parameters()`` deduplicates shared tensors, which is what we want: a
    weight-tied block should not have its cost counted twice inside one module.
    """
    return sum(p.numel() for p in module.parameters())


# Attributes worth surfacing in the details panel. PyTorch keeps layer
# hyper-parameters as plain attributes, so a scalar allowlist reads well
# without dumping tensors into the HTML. Superset of the NNX list: the two
# frameworks use different names for the same ideas (``strides`` vs ``stride``,
# ``epsilon`` vs ``eps``, ``rate`` vs ``p``) and carrying both costs nothing.
_TORCH_INTERESTING = (
    "in_features", "out_features", "in_channels", "out_channels",
    "kernel_size", "stride", "padding", "dilation", "groups", "output_padding",
    "num_features", "num_groups", "num_embeddings", "embedding_dim",
    "eps", "momentum", "affine", "track_running_stats",
    "bias", "p", "inplace", "negative_slope",
    "num_heads", "embed_dim", "hidden_size", "num_layers", "batch_first",
    "bidirectional", "dropout", "normalized_shape", "scale_factor", "mode",
    "align_corners", "start_dim", "end_dim", "dim", "output_size",
)


def _config_of(module: nn.Module) -> dict:
    cfg: dict = {}
    for key in _TORCH_INTERESTING:
        if not hasattr(module, key):
            continue
        val = getattr(module, key)
        if isinstance(val, bool) or isinstance(val, (int, float, str)):
            cfg[key] = val
        elif isinstance(val, (tuple, list)) and all(
                isinstance(v, (int, float)) for v in val):
            cfg[key] = list(val)
        elif val is None:
            cfg[key] = "None"
        elif isinstance(val, torch.nn.Parameter):
            # ``bias`` is a tensor when present; report only whether it exists.
            cfg[key] = True
    return cfg


def _input_names(model: nn.Module, args: tuple, kwargs: dict) -> list:
    """One display name per input *tensor*, aligned with ``_leaves`` order.

    Names come from ``forward``'s signature, so a two-argument generator shows
    "reference" and "driver" instead of "input 0" and "input 1". An argument
    holding several tensors (a list of feature maps) contributes one name per
    tensor, suffixed by position - the pills are per-tensor, not per-parameter.
    """
    try:
        params = list(inspect.signature(model.forward).parameters.values())
    except (TypeError, ValueError):
        params = []
    if params and params[0].name in ("self", "cls"):
        params = params[1:]

    names: list = []
    for i, val in enumerate(args):
        base = params[i].name if i < len(params) else f"input {i}"
        n = len(_leaves(val))
        names.extend([base] if n == 1 else [f"{base}[{j}]" for j in range(n)])
    for key, val in kwargs.items():
        n = len(_leaves(val))
        names.extend([key] if n == 1 else [f"{key}[{j}]" for j in range(n)])
    return names


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

def trace_model(
    model: nn.Module,
    *args: Any,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Graph:
    """Run one forward pass of ``model`` and record its module graph.

    ``*args`` / ``**kwargs`` are passed straight to ``model(...)``. The pass
    runs under ``torch.no_grad()`` in eval mode: nothing here needs gradients,
    and eval keeps dropout and batch-norm from making the recorded shapes
    depend on training state. If the forward pass raises, the partial graph is
    still returned with ``error`` set - a failed model is exactly when the
    picture is most useful.
    """
    # Real attribute paths for every reachable module, keyed by identity. The
    # root's own path is "", which becomes "<root>" on its card.
    paths: dict = {}
    for path, sub in model.named_modules():
        paths.setdefault(id(sub), path)

    nodes: list = []
    edges: list = []
    producer: dict = {}      # id(tensor) -> node id that produced it
    # CPython reuses the address of a freed object, so an intermediate tensor
    # that goes out of scope can hand its ``id()`` to a later, unrelated tensor
    # and fabricate an edge. Holding a reference to everything we key on keeps
    # those ids unique for the duration of the trace.
    keepalive: list = []
    stack: list = []         # currently-executing node ids
    order: list = []         # completion order, for skip-edge detection
    child_of: dict = {}      # node id -> ids of modules it called directly
    produced_by: dict = {}   # node id -> ids of the tensors it returned
    merge_candidates: list = []   # calls that combined values with bare ops
    # Inputs recorded by the pre hook, read back by the matching post hook -
    # that is where they are known, and where merge placement needs them.
    node_meta: dict = {}
    # Source scan per class, computed once: reading ``forward``'s AST is what
    # makes residual adds and ``torch.cat`` visible, and it does not change
    # between calls of the same class.
    ops_cache: dict = {}
    # FLOPs are read off the framework's own op-level counter rather than
    # estimated per layer type, so attention, einsum and custom ops are counted
    # the same way conv is. ``flop_at_entry`` holds the running total when each
    # node opened; the delta at exit is that call's cost, children included.
    counter = FlopCounterMode(display=False) if FlopCounterMode else None
    flop_at_entry: dict = {}

    def _flops_now() -> int:
        if counter is None:
            return -1
        try:
            return counter.get_total_flops()
        except Exception:              # pragma: no cover - counter API drift
            return -1

    def _register(obj: Any, node_id: int, *, claim: bool = True) -> None:
        """Record ``node_id`` as the producer of every tensor leaf in ``obj``.

        ``claim=False`` keeps the first (innermost) producer: a container
        returns its child's tensor unchanged, and the child is the real origin
        of that value.
        """
        for leaf in _leaves(obj):
            if claim or id(leaf) not in producer:
                producer[id(leaf)] = node_id
            keepalive.append(leaf)

    def pre_hook(mod: nn.Module, inp: tuple, kw: dict = None):
        """Opens a node when a module is entered."""
        kw = kw or {}
        path = paths.get(id(mod))
        if path is None:                   # built during the call
            return
        node = Node(
            id=len(nodes),
            path=path or "<root>",
            name=path.split(".")[-1] if path else type(mod).__name__,
            cls=type(mod).__name__,
            depth=len(stack),
            parent=stack[-1] if stack else None,
            params=_count_params(mod),
            own_params=0,
            inputs=_as_tensors((inp, kw)),
            outputs=[],
            config=_config_of(mod),
        )
        nodes.append(node)
        if node.parent is not None:
            child_of.setdefault(node.parent, []).append(node.id)
        stack.append(node.id)
        flop_at_entry[node.id] = _flops_now()

        in_leaves = _leaves((inp, kw))
        keepalive.extend(in_leaves)
        # Pair each consumed tensor with its own descriptor, so an edge reports
        # the tensor that actually travelled it. Labelling every incoming edge
        # with ``inputs[0]`` misreports each argument after the first - a style
        # vector arriving at ``Modulated(out, w)`` would claim the feature
        # map's shape.
        for leaf in in_leaves:
            src = producer.get(id(leaf))
            if src is None or src == node.id or src in stack:
                continue
            if src == INPUT_NODE and node.depth == 0:
                continue  # the root receiving the input is not an edge
            edges.append(Edge(src, node.id,
                              Tensor(tuple(leaf.shape), _dtype(leaf))))
        # Stash for the matching post hook: the pre hook is where the inputs
        # are known, and the post hook needs them to place merge nodes.
        node_meta[node.id] = {
            "in_ids": [id(l) for l in in_leaves],
            "in_arrays": [Tensor(tuple(l.shape), _dtype(l)) for l in in_leaves],
        }

    def post_hook(mod: nn.Module, inp: tuple, out: Any):
        """Closes the node opened by ``pre_hook`` and records what it produced."""
        if id(mod) not in paths or not stack:
            return
        nid = stack.pop()
        order.append(nid)
        node = nodes[nid]
        node.outputs = _as_tensors(out)
        entry = flop_at_entry.pop(nid, -1)
        if entry >= 0:
            now = _flops_now()
            if now >= 0:
                node.flops = max(0, now - entry)

        cls = type(mod)
        if cls not in ops_cache:
            ops_cache[cls] = _scan_ops(cls.forward)
        source_ops = ops_cache[cls]

        out_ids = set(_array_ids(out))
        child_ids: set = set()
        for cid in child_of.get(nid, []):
            child_ids.update(produced_by.get(cid, ()))
        if source_ops:
            meta = node_meta.get(nid, {"in_ids": [], "in_arrays": []})
            merge_candidates.append({
                "node": nid,
                # A list, not a set: ``in_arrays`` is zipped against it, so the
                # order of the two must agree.
                "in_ids": meta["in_ids"],
                "in_arrays": meta["in_arrays"],
                "child_ids": child_ids,
                "out": next(iter(node.outputs), None),
                "in_tensors": list(node.inputs),
                "ops": source_ops,
                # True when the value the host returned is *not* one a child
                # produced - the mark of a trailing combine such as ``out + x``
                # that closes a residual branch.
                "tail_combine": bool(out_ids) and not (out_ids & child_ids),
            })

        produced_by[nid] = out_ids
        # A wrapper module often returns its last child's tensor verbatim. Keep
        # the child as the producer so edges and output attribution point at
        # the module that actually computed the value.
        _register(out, nid, claim=False)

    # Install hooks on every submodule. ``with_kwargs`` keeps keyword arguments
    # visible; it is a newer addition, so fall back when it is unavailable.
    handles: list = []
    for sub in dict.fromkeys(m for _, m in model.named_modules()):
        try:
            handles.append(sub.register_forward_pre_hook(
                pre_hook, with_kwargs=True))
        except TypeError:                  # pragma: no cover - older torch
            handles.append(sub.register_forward_pre_hook(
                lambda m, i: pre_hook(m, i, {})))
        handles.append(sub.register_forward_hook(post_hook))

    error = None
    outputs: list = []
    result = None
    # Node id -1 stands for "the model's own input", so a module consuming the
    # raw input (a residual shortcut, typically) still gets a visible edge. A
    # model called with several arguments gets one pill per argument: leaf i is
    # owned by IN_BASE - i, matching the ids the renderer lays out. Leaf 0 keeps
    # INPUT_NODE so single-input graphs are unchanged.
    for i, leaf in enumerate(_leaves((args, kwargs))):
        _register(leaf, INPUT_NODE if i == 0 else IN_BASE - i)

    was_training = model.training
    total_flops = -1
    try:
        model.eval()
        with torch.no_grad():
            if counter is not None:
                # A dispatch mode, so it has to wrap the call the hooks run
                # inside. Any failure here propagates like a normal trace error
                # rather than being retried - a half-built node list cannot be
                # reused, and silently re-running the model could have side
                # effects.
                with counter:
                    result = model(*args, **kwargs)
                total_flops = _flops_now()
            else:
                result = model(*args, **kwargs)
        outputs = _as_tensors(result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    # Which module produced each returned tensor? Recorded here (while the
    # producer map is still populated) so the renderer can draw one output node
    # per result and wire it to the module it came from.
    out_sources: list = []
    if result is not None:
        for leaf in _leaves(result):
            out_sources.append({
                "tensor": Tensor(tuple(leaf.shape), _dtype(leaf)),
                # ``None`` means the value was built by bare torch ops after the
                # last module ran, so no module owns it directly.
                "src": producer.get(id(leaf)),
            })

    # channel_axis=1: everything here is NCHW, so a concat of two equal-shaped
    # feature maps widens dim 1. The default (-1) is the Flax tracer's NHWC.
    unique_edges = finalize_graph(nodes, edges, producer, child_of,
                                  out_sources, merge_candidates,
                                  channel_axis=1)

    return Graph(
        nodes=nodes,
        edges=unique_edges,
        model_name=model_name or type(model).__name__,
        total_params=_count_params(model),
        input_tensors=_as_tensors((args, kwargs)),
        output_tensors=outputs,
        error=error,
        output_sources=out_sources,
        input_names=_input_names(model, args, kwargs),
        total_flops=total_flops,
    )


__all__ = ["trace_model", "Graph", "Node", "Edge", "Tensor"]
