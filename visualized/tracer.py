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

import ast
import dataclasses
import inspect
import textwrap
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx


# Synthetic node id meaning "the model's own input" (see ``trace_model``).
INPUT_NODE = -1
# Arguments after the first get their own input pills, numbered IN_BASE - i.
# The renderer mirrors both constants - keep them in step.
IN_BASE = -500


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
    # "module" for a real nnx.Module, "merge" for a synthesized node standing
    # in for a bare array combine (``out + x``) that owns no parameters.
    kind: str = "module"
    op: Optional[str] = None      # for merges: "add", "concat", ...
    # Position in the forward pass. Equals ``id`` for real modules; merges are
    # appended after tracing, so they carry the order of the call they belong
    # to and still sort next to their siblings.
    order: int = -1

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
            "kind": self.kind,
            "op": self.op,
            "order": self.order if self.order >= 0 else self.id,
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
    # One entry per returned array: ``{"tensor": Tensor, "src": node id|None}``
    output_sources: list = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "model_name": self.model_name,
            "total_params": self.total_params,
            "input_tensors": [t.to_dict() for t in self.input_tensors],
            "output_tensors": [t.to_dict() for t in self.output_tensors],
            "error": self.error,
            "output_sources": [
                {"tensor": o["tensor"].to_dict(), "src": o["src"]}
                for o in self.output_sources
            ],
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


def _is_descendant(node_id: int, ancestor_id: int, nodes: list) -> bool:
    """Is ``node_id`` inside ``ancestor_id``'s subtree?"""
    cur = nodes[node_id].parent
    while cur is not None:
        if cur == ancestor_id:
            return True
        cur = nodes[cur].parent
    return False


def _combine_op(fn: Optional[Callable]) -> Optional[str]:
    """Does ``fn`` (an unwrapped ``__call__``) combine tensors with array ops?

    Returns ``"add"`` / ``"concat"`` / ``None``. Residual adds are written as
    ``out = out + x`` rather than as a module, so nothing in the runtime trace
    marks them. Reading the source is exact where guessing from array identity
    is not: ``ConvBlock`` also returns a fresh array (its activation) but
    contains no add, and only the source distinguishes the two.
    """
    if fn is None:
        return None
    fn = getattr(fn, "__wrapped__", fn)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None                      # C-implemented or source unavailable
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return "add"
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) in
                ("concatenate", "concat")):
            return "concat"
    return None


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
    child_of: dict = {}      # node id -> ids of modules it called directly
    produced_by: dict = {}   # node id -> ids of the arrays it returned
    merge_candidates: list = []   # calls that combined values with bare ops

    def _register(obj: Any, node_id: int, *, claim: bool = True) -> None:
        """Record ``node_id`` as the producer of every array leaf in ``obj``.

        ``claim=False`` keeps the first (innermost) producer: a container
        returns its child's array unchanged, and the child is the real
        origin of that value.
        """
        for leaf in jax.tree_util.tree_leaves(obj):
            if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                if claim or id(leaf) not in producer:
                    producer[id(leaf)] = node_id
                keepalive.append(leaf)

    def _wrap(fn: Callable) -> Callable:
        # Computed once per class, from the *original* function - inside the
        # wrapper ``cls.__dict__['__call__']`` is this wrapper, not the source.
        combine_op = _combine_op(fn)

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
            if node.parent is not None:
                child_of.setdefault(node.parent, []).append(node.id)
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

            # Did this call combine values with bare array ops? A container
            # that returns an array none of its children produced, while two
            # or more values it *did* see are still unaccounted for, has run
            # something like ``out + x`` between its children. Record the
            # candidate; ``_add_merges`` below decides which are real.
            out_ids = set(_array_ids(out))
            child_ids: set = set()
            for cid in child_of.get(node.id, []):
                child_ids.update(produced_by.get(cid, ()))
            if out_ids and not (out_ids & child_ids):
                merge_candidates.append({
                    "node": node.id,
                    "in_ids": set(_array_ids((a, kw))),
                    "child_ids": child_ids,
                    "out": next(iter(node.outputs), None),
                    "op": combine_op,
                })

            produced_by[node.id] = out_ids
            # A wrapper module often returns its last child's array verbatim.
            # Keep the child as the producer so edges and output attribution
            # point at the module that actually computed the value.
            _register(out, node.id, claim=False)
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
    result = None
    # Node id -1 stands for "the model's own input", so a module consuming the
    # raw input (a residual shortcut, typically) still gets a visible edge.
    # A model called with several arguments gets one pill per argument: leaf i
    # is owned by IN_BASE - i, matching the ids the renderer lays out. Leaf 0
    # keeps INPUT_NODE so single-input graphs are unchanged.
    for i, leaf in enumerate(jax.tree_util.tree_leaves((args, kwargs))):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            _register(leaf, INPUT_NODE if i == 0 else IN_BASE - i)
    try:
        result = model(*args, **kwargs)
        outputs = _as_tensors(result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for cls, fn in patched.items():
            setattr(cls, "__call__", fn)

    # Which module produced each returned array? Recorded here (while the
    # producer map is still populated) so the renderer can draw one output
    # node per result and wire it to the module it came from.
    out_sources: list = []
    if result is not None:
        for leaf in jax.tree_util.tree_leaves(result):
            if not (hasattr(leaf, "shape") and hasattr(leaf, "dtype")):
                continue
            out_sources.append({
                "tensor": Tensor(tuple(leaf.shape), str(leaf.dtype)),
                # ``None`` means the value was built by bare jnp ops after the
                # last module ran, so no module owns it directly.
                "src": producer.get(id(leaf)),
            })

    # own_params = params not attributable to any child module
    children: dict = {}
    for n in nodes:
        if n.parent is not None:
            children.setdefault(n.parent, []).append(n.id)
    for n in nodes:
        child_total = sum(nodes[c].params for c in children.get(n.id, []))
        n.own_params = max(0, n.params - child_total)

    # Residual adds are bare array ops (``out = out + x``), not modules, so
    # nothing above records them and the branches appear to diverge and never
    # rejoin. Synthesize an explicit merge node for each container that
    # combined an incoming value with something its children produced: it is
    # the only part of a residual block that carries no parameters, and
    # without it the picture is simply wrong.
    for cand in merge_candidates:
        host = nodes[cand["node"]]
        op = cand["op"]
        if op is None:
            continue                    # no ``+`` / concat in this __call__
        kids_ids = child_of.get(host.id, [])
        if not kids_ids:
            continue                    # a leaf module, nothing was combined
        # Operands are the children whose output nothing else consumed - the
        # ends of each branch. In ``conv2(conv1(x)) + shortcut(x)`` that is
        # {conv2, shortcut}: conv1's output was eaten by conv2, so conv1 is
        # mid-branch, while the *last-called* child is merely ``shortcut``.
        consumed = {e.src for e in edges if e.dst in kids_ids}
        branch_ends = [k for k in kids_ids if k not in consumed]
        # Plus any value the host received that no child produced (a residual
        # adding the raw input, as in ``out + x``). Skip it when a branch end
        # already consumed that same value - then the branch *is* that path
        # (``shortcut(x)``), and adding the raw input would double-count it.
        in_end_subtree = set(branch_ends)
        for n in nodes:
            if any(_is_descendant(n.id, b, nodes) for b in branch_ends):
                in_end_subtree.add(n.id)
        fed_ends = {e.src for e in edges
                    if e.dst in in_end_subtree and e.src not in kids_ids}
        carried = [src for src in (producer.get(i) for i in cand["in_ids"])
                   if src is not None and src != host.id
                   and src not in kids_ids and src not in fed_ends]
        if len(branch_ends) + len(carried) < 2:
            continue                    # nothing actually converged here
        last_kid = branch_ends[-1] if branch_ends else kids_ids[-1]
        sym = "+" if op == "add" else "⧺"
        merge = Node(
            id=len(nodes),
            path=f"{host.path}.{sym}" if host.path != "<root>" else sym,
            name=sym,
            cls=op,
            depth=host.depth + 1,
            parent=host.id,
            params=0,
            own_params=0,
            inputs=[t for t in (cand["out"],) if t is not None],
            outputs=[t for t in (cand["out"],) if t is not None],
            config={},
            kind="merge",
            op=op,
            # Runs after the host's last child *and everything that child
            # called*, so sort past that whole subtree rather than past the
            # child's own id.
            order=max([last_kid] + [n.id for n in nodes
                                    if _is_descendant(n.id, last_kid, nodes)]),
        )
        nodes.append(merge)
        children.setdefault(host.id, []).append(merge.id)
        tensor = cand["out"] or Tensor((), "")
        # Every branch end feeds the merge. The first (the longest-running
        # path) is the main line; the rest are the shortcuts.
        operands = list(dict.fromkeys(branch_ends + carried))
        for i, src in enumerate(operands):
            edges.append(Edge(src, merge.id, tensor, skip=i > 0))
        # anything that consumed the host's output now consumes the merge
        for e in edges:
            if e.src == host.id and e.dst != merge.id:
                e.src = merge.id
        for o in out_sources:
            if o["src"] == host.id:
                o["src"] = merge.id

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
        output_sources=out_sources,
    )


__all__ = ["trace_model", "Graph", "Node", "Edge", "Tensor"]
