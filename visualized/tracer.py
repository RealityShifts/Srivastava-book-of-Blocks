"""Trace a Flax NNX model's forward pass into a serializable graph.

Strategy (and why): ``jax.make_jaxpr`` flattens everything to primitives and
NNX emits no ``name_stack``, so a jaxpr cannot tell you *which module* an op
belongs to. Instead we temporarily wrap ``__call__`` on every ``nnx.Module``
class reachable from the model, run one real forward pass, and record:

    - the module hierarchy (from ``nnx.iter_graph``, so names are real
      attribute paths like ``conv1.norm``)
    - call order and nesting depth
    - concrete input/output shapes and dtypes
    - dataflow edges, recovered by matching ``id()`` of the arrays a module
      consumed against arrays previous modules produced

The wrapping is installed on the *class* (NNX modules are dataclass-like and
reject stray instance attributes) and always removed in a ``finally``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx


# Synthetic node id meaning "the model's own input" (see ``trace_model``).
INPUT_NODE = -1


# ---------------------------------------------------------------------------
# Graph data model
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Tensor:
    """One array flowing along an edge."""

    shape: tuple
    dtype: str

    def to_dict(self) -> dict:
        return {"shape": list(self.shape), "dtype": self.dtype}

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclasses.dataclass
class Node:
    """A single module invocation."""

    id: int
    path: str            # attribute path, e.g. "conv1.norm"
    name: str            # leaf name, e.g. "norm"
    cls: str             # class name, e.g. "LayerNorm"
    depth: int           # nesting depth during the traced call
    parent: Optional[int]
    params: int          # trainable parameters owned (incl. children)
    own_params: int      # parameters owned directly (excl. child modules)
    inputs: list         # list[Tensor]
    outputs: list        # list[Tensor]
    config: dict         # scalar attributes worth showing (kernel size, ...)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "cls": self.cls,
            "depth": self.depth,
            "parent": self.parent,
            "params": self.params,
            "own_params": self.own_params,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "config": self.config,
            "error": self.error,
        }


@dataclasses.dataclass
class Edge:
    """Dataflow: ``src`` produced an array that ``dst`` consumed."""

    src: int
    dst: int
    tensor: Tensor
    skip: bool = False       # True when it bypasses the immediate predecessor
    inferred: bool = False   # from execution order, not array identity

    def to_dict(self) -> dict:
        return {
            "src": self.src,
            "dst": self.dst,
            "tensor": self.tensor.to_dict(),
            "skip": self.skip,
            "inferred": self.inferred,
        }


@dataclasses.dataclass
class Graph:
    """The traced model: nodes, edges, and roll-up stats."""

    nodes: list
    edges: list
    model_name: str
    total_params: int
    input_tensors: list
    output_tensors: list
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "model_name": self.model_name,
            "total_params": self.total_params,
            "input_tensors": [t.to_dict() for t in self.input_tensors],
            "output_tensors": [t.to_dict() for t in self.output_tensors],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def _as_tensors(obj: Any) -> list:
    """Flatten a pytree into the concrete arrays it holds."""
    leaves = jax.tree_util.tree_leaves(obj)
    out = []
    for leaf in leaves:
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            out.append(Tensor(tuple(leaf.shape), str(leaf.dtype)))
    return out


def _array_ids(obj: Any) -> list:
    """``id()`` of every array leaf - used to stitch dataflow edges."""
    return [id(l) for l in jax.tree_util.tree_leaves(obj)
            if hasattr(l, "shape") and hasattr(l, "dtype")]


def _count_params(module: nnx.Module) -> int:
    """Total parameter elements owned by ``module`` and its children."""
    total = 0
    for _, leaf in nnx.iter_graph(module):
        if isinstance(leaf, nnx.Param) and hasattr(leaf.value, "size"):
            total += int(leaf.value.size)
    return total


# Attributes worth surfacing in the details panel. NNX modules keep their
# hyper-parameters as plain attributes, so a scalar allowlist reads well
# without dumping arrays into the HTML.
_INTERESTING = (
    "in_features", "out_features", "kernel_size", "strides", "padding",
    "num_features", "num_groups", "epsilon", "use_bias", "use_scale",
    "feature_group_count", "kernel_dilation", "rate", "deterministic",
    "num_heads", "in_ch", "out_ch", "dim", "hidden", "mode", "axis",
    "momentum", "dtype", "param_dtype",
)


def _config_of(module: nnx.Module) -> dict:
    cfg = {}
    for key in _INTERESTING:
        if not hasattr(module, key):
            continue
        val = getattr(module, key)
        if isinstance(val, (bool, int, float, str)):
            cfg[key] = val
        elif isinstance(val, tuple) and all(
                isinstance(v, (int, float)) for v in val):
            cfg[key] = list(val)
        elif val is None or isinstance(val, jnp.dtype):
            cfg[key] = str(val)
    # Activations are stored as bare callables; show something readable.
    act = getattr(module, "act", None)
    if callable(act):
        cfg["activation"] = getattr(act, "__name__", type(act).__name__)
    return cfg


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

def trace_model(
    model: nnx.Module,
    *args: Any,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Graph:
    """Run one forward pass of ``model`` and record its module graph.

    ``*args`` / ``**kwargs`` are passed straight to ``model(...)``. The pass
    runs eagerly (no ``jit``) so shapes are concrete. If the forward pass
    raises, the partial graph is still returned with ``error`` set - a
    failed model is exactly when the picture is most useful.
    """
    # Real attribute paths for every reachable module, keyed by identity.
    paths: dict = {}
    for path, sub in nnx.iter_graph(model):
        if isinstance(sub, nnx.Module):
            paths[id(sub)] = ".".join(str(p) for p in path)

    nodes: list = []
    edges: list = []
    producer: dict = {}      # id(array) -> node id that produced it
    # CPython reuses the address of a freed object, so an intermediate array
    # that goes out of scope can hand its ``id()`` to a later, unrelated array
    # and fabricate an edge. Holding a reference to everything we key on keeps
    # those ids unique for the duration of the trace.
    keepalive: list = []
    stack: list = []         # currently-executing node ids
    order: list = []         # completion order, for skip-edge detection

    def _register(obj: Any, node_id: int) -> None:
        """Record ``node_id`` as the producer of every array leaf in ``obj``."""
        for leaf in jax.tree_util.tree_leaves(obj):
            if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                producer[id(leaf)] = node_id
                keepalive.append(leaf)

    def _wrap(fn: Callable) -> Callable:
        def wrapper(self, *a, **kw):
            key = id(self)
            if key not in paths:      # a module built during the call
                return fn(self, *a, **kw)

            path = paths[key]
            node = Node(
                id=len(nodes),
                path=path or "<root>",
                name=path.split(".")[-1] if path else type(self).__name__,
                cls=type(self).__name__,
                depth=len(stack),
                parent=stack[-1] if stack else None,
                params=_count_params(self),
                own_params=0,
                inputs=_as_tensors((a, kw)),
                outputs=[],
                config=_config_of(self),
            )
            nodes.append(node)
            stack.append(node.id)

            # Edges: which earlier module produced the arrays we consumed?
            # An ancestor already on the stack passing its own input down is
            # containment, not dataflow, so skip those.
            keepalive.extend(
                l for l in jax.tree_util.tree_leaves((a, kw))
                if hasattr(l, "shape") and hasattr(l, "dtype"))
            for aid in _array_ids((a, kw)):
                src = producer.get(aid)
                if src is None or src == node.id or src in stack:
                    continue
                if src == INPUT_NODE and node.depth == 0:
                    continue  # the root receiving the input is not an edge
                edges.append(Edge(
                    src, node.id,
                    next(iter(node.inputs), Tensor((), ""))))

            try:
                out = fn(self, *a, **kw)
            except Exception as exc:  # keep the partial graph
                node.error = f"{type(exc).__name__}: {exc}"
                stack.pop()
                order.append(node.id)
                raise
            finally:
                if stack and stack[-1] == node.id:
                    stack.pop()
                    order.append(node.id)

            node.outputs = _as_tensors(out)
            _register(out, node.id)
            return out

        wrapper._visualized_tracer = True
        return wrapper

    # Install wrappers per class (not per instance - NNX rejects stray attrs).
    classes = {type(sub) for _, sub in nnx.iter_graph(model)
               if isinstance(sub, nnx.Module)}
    patched: dict = {}
    for cls in classes:
        fn = cls.__dict__.get("__call__")
        # ``__wrapped__`` is not a valid guard here: ``@typecheck`` already
        # sets it, which would make us skip every checked module.
        if fn is None or getattr(fn, "_visualized_tracer", False):
            continue
        patched[cls] = fn
        setattr(cls, "__call__", _wrap(fn))

    error = None
    outputs: list = []
    # Node id -1 stands for "the model's own input", so a module consuming the
    # raw input (a residual shortcut, typically) still gets a visible edge.
    _register((args, kwargs), INPUT_NODE)
    try:
        result = model(*args, **kwargs)
        outputs = _as_tensors(result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for cls, fn in patched.items():
            setattr(cls, "__call__", fn)

    # own_params = params not attributable to any child module
    children: dict = {}
    for n in nodes:
        if n.parent is not None:
            children.setdefault(n.parent, []).append(n.id)
    for n in nodes:
        child_total = sum(nodes[c].params for c in children.get(n.id, []))
        n.own_params = max(0, n.params - child_total)

    # Array identity only survives while a value passes straight from one
    # module to the next. A bare ``jnp``/``jax.nn`` call in between (an
    # activation, a mean, a reshape) produces a *new* array, so the chain
    # breaks and the consumer looks like it has no producer at all. Fall back
    # to execution order for those: link a leaf with no recorded producer to
    # the previously completed leaf, which is what actually fed it.
    leaf_ids = [n.id for n in nodes if not children.get(n.id)]
    has_producer = {e.dst for e in edges}
    prev_leaf = None
    for nid in leaf_ids:                      # leaf_ids is in call order
        node = nodes[nid]
        if (prev_leaf is not None
                and nid not in has_producer
                and node.inputs                       # consumes something
                and nodes[prev_leaf].outputs):        # predecessor produced
            edges.append(Edge(prev_leaf, nid,
                              next(iter(node.inputs), Tensor((), "")),
                              inferred=True))
        prev_leaf = nid

    # Deduplicate: one module pair can exchange several arrays.
    seen = set()
    unique_edges = []
    for e in edges:
        sig = (e.src, e.dst)
        if sig in seen:
            continue
        seen.add(sig)
        unique_edges.append(e)

    # Mark skip connections by fan-out: when one value feeds several
    # consumers, the immediate next one is the "main" path and any consumer
    # further along is a shortcut that bypasses it. This is the residual
    # shape (``x`` going to both ``conv1`` and ``shortcut``); a purely
    # sequential chain has fan-out 1 everywhere and stays unmarked.
    outgoing: dict = {}
    for e in unique_edges:
        if not e.inferred:      # order-based links are never branches
            outgoing.setdefault(e.src, []).append(e)
    for group in outgoing.values():
        if len(group) < 2:
            continue
        main = min(e.dst for e in group)
        for e in group:
            if e.dst != main:
                e.skip = True

    return Graph(
        nodes=nodes,
        edges=unique_edges,
        model_name=model_name or type(model).__name__,
        total_params=_count_params(model),
        input_tensors=_as_tensors((args, kwargs)),
        output_tensors=outputs,
        error=error,
    )


__all__ = ["trace_model", "Graph", "Node", "Edge", "Tensor"]
