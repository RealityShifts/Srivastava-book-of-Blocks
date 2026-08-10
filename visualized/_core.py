"""Framework-agnostic core of the model tracer.

Everything here is about *graphs*, not about any one array library: the
serializable node/edge/tensor model that ``renderer.py`` consumes, the
source-scanning machinery that recovers bare array ops (a residual ``+``, a
concatenate, an activation) which no runtime hook can see because they are
plain expressions rather than modules, and the post-trace surgery that turns a
raw recording into the graph the renderer draws.

The Flax NNX tracer (``tracer.py``) and the PyTorch tracer (``tracer_torch.py``)
both build the same ``Graph`` from this module, which is why one renderer
serves both. Keep anything importing ``jax`` or ``torch`` out of this file.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from typing import Any, Callable, Optional


# Synthetic node id meaning "the model's own input" (see each tracer).
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
    # Parameter name per input array, read off ``__call__``'s signature, so a
    # pill reads "reference" rather than "input 0".
    input_names: list = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "model_name": self.model_name,
            "total_params": self.total_params,
            "input_tensors": [t.to_dict() for t in self.input_tensors],
            "output_tensors": [t.to_dict() for t in self.output_tensors],
            "input_names": list(self.input_names),
            "error": self.error,
            "output_sources": [
                {"tensor": o["tensor"].to_dict(), "src": o["src"]}
                for o in self.output_sources
            ],
        }

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


# Modules whose functions are array ops worth drawing. A call like
# ``jnp.concat(...)`` or ``nnx.leaky_relu(x)`` is a real step in the forward
# pass, but it is not an ``nnx.Module``, so the runtime trace never sees it.
# Both frameworks' spellings live in one table. The names do not collide, and a
# single list means a model that mixes numpy with either framework is scanned
# the same way. ``torch``/``F``/``nn`` cover PyTorch's usual import aliases.
_OP_ROOTS = ("jnp", "jax", "nnx", "np", "numpy", "lax", "F",
             "torch", "nn", "tf")

# Binary/unary operators, mapped to the symbol drawn inside the op circle.
_BINOPS = {
    ast.Add: ("add", "+"), ast.Sub: ("sub", "−"),
    ast.Mult: ("mul", "×"), ast.Div: ("div", "÷"),
    ast.MatMult: ("matmul", "@"), ast.Pow: ("pow", "^"),
    ast.Mod: ("mod", "%"), ast.FloorDiv: ("floordiv", "//"),
}

# Elementwise ops, whose output shape always equals their input shape. For
# anything outside this set (``mean``, ``reshape``, ``sum``, ``transpose``...)
# the produced shape has to be read off the module that consumed the value,
# since bare ops are not intercepted and never record their own output.
_SHAPE_PRESERVING = frozenset({
    "relu", "leaky_relu", "gelu", "elu", "silu", "swish", "selu", "celu",
    "sigmoid", "tanh", "softplus", "softsign", "hard_sigmoid", "hard_tanh",
    "hard_swish", "hard_silu", "log_sigmoid", "softmax", "log_softmax",
    "abs", "exp", "log", "sqrt", "rsqrt", "square", "sign", "negative",
    "clip", "round", "floor", "ceil", "where", "dropout", "standardize",
})

# Short glyphs for the array functions that come up often; anything else
# falls back to the function's own name.
_OP_SYMBOLS = {
    "concat": "⧺", "concatenate": "⧺", "stack": "⧺",
    "add": "+", "subtract": "−", "multiply": "×",
    "divide": "÷", "true_divide": "÷", "matmul": "@",
    "dot": "@", "einsum": "∑", "sum": "∑", "mean": "μ",
    "reshape": "↳", "transpose": "⇄", "swapaxes": "⇄",
    "split": "✂", "where": "?", "pad": "▣",
    # PyTorch spellings of the same operations.
    "cat": "⧺", "hstack": "⧺", "vstack": "⧺", "sub": "−", "mul": "×",
    "div": "÷", "bmm": "@", "mm": "@", "view": "↳", "permute": "⇄",
    "flatten": "↳", "squeeze": "↓", "unsqueeze": "↑", "chunk": "✂",
    "interpolate": "⤢", "softmax": "σ",
}

# Ops that join several values into one. Both spellings live here: JAX writes
# ``jnp.concatenate``, PyTorch writes ``torch.cat``/``hstack``/``vstack``, and
# the join logic below must recognise either. Listing only the JAX names left
# every ``torch.cat`` drawn as a unary op with a single incoming arrow.
_JOIN_OPS = frozenset({
    "concat", "concatenate", "stack",
    "cat", "hstack", "vstack", "column_stack", "row_stack", "dstack",
})


# Builtins that return a plain Python number, so arithmetic on their result
# is index/shape bookkeeping rather than dataflow.
_SCALAR_CALLS = frozenset({"len", "int", "round", "abs", "min", "max", "sum"})


def _is_scalar_expr(node: ast.AST) -> bool:
    """Does ``node`` obviously evaluate to a plain Python number?

    ``len(x) - 1`` and ``i + 1`` are index arithmetic, not array ops, but the
    AST cannot tell them apart from ``out + x`` by shape alone. Recognising the
    numeric side is enough: a subtraction with a literal ``1`` on the right is
    bookkeeping, whereas a residual combines two names.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(
            node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)):
        return _is_scalar_expr(node.operand)
    if isinstance(node, ast.Call):
        return getattr(node.func, "id", None) in _SCALAR_CALLS
    if isinstance(node, ast.BinOp):
        return (_is_scalar_expr(node.left) or _is_scalar_expr(node.right))
    # ``x.shape[0]``, ``x.ndim`` - attributes that hold sizes, never arrays.
    if isinstance(node, ast.Attribute) and node.attr in ("ndim", "size"):
        return True
    if (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"):
        return True
    return False


def _op_root(node: ast.AST) -> Optional[str]:
    """Leftmost name of a dotted expression: ``jax.nn.relu`` -> ``jax``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _dotted(node: ast.AST) -> str:
    """Render an attribute chain back to source form."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


@dataclasses.dataclass
class _Op:
    """A bare array operation found in a module's ``__call__`` source."""

    op: str          # "concat", "add", "leaky_relu", ...
    sym: str         # glyph drawn in the node circle
    qual: str        # full call as written, e.g. "jnp.concat"
    lineno: int      # source line, used to order ops against module calls
    col: int
    nargs: int       # positional array arguments, so binary ops fan in
    # ``self.<attr>`` calls this op's own statement makes - the op consumes
    # their result directly (``nnx.leaky_relu(self.init(x))``).
    after: tuple = ()
    # The nearest ``self.<attr>`` call *before* this op in source order, for
    # the common two-statement form ``x = layer(x)`` then ``x = act(x)``.
    prev_call: Optional[str] = None
    # True when the op sits inside a loop, so it runs once per iteration and
    # must attach to every child that loop invoked, not just the first.
    in_loop: bool = False
    # AST nesting depth. Within one statement the deepest op evaluates first,
    # which is the order the synthesized nodes must chain in.
    depth: int = 0


def _scan_ops(fn: Optional[Callable]) -> list:
    """Every bare array op in ``fn``, in source order.

    Residual adds, ``jnp.concat``, activations like ``nnx.leaky_relu`` and
    reshapes are all written as plain expressions rather than modules, so
    nothing in a module-level runtime trace records them. Reading the source
    is what makes them visible - and it is exact where guessing from array
    identity is not, since a module that merely returns a fresh array (a
    ``ConvBlock`` and its activation) looks identical at runtime to one that
    combined two branches.
    """
    if fn is None:
        return []
    fn = getattr(fn, "__wrapped__", fn)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return []                        # C-implemented or source unavailable

    # Nodes inside a ``for``/``while`` body run once per iteration.
    looped: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for inner in node.body:
                for sub in ast.walk(inner):
                    looped.add(id(sub))

    # Nesting depth, so ops in one expression can be ordered the way they
    # evaluate: in ``tanh(a(x)) * 2 + y`` the ``+`` is outermost but runs
    # last, while ``tanh`` is deepest and runs first.
    depth_of: dict = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            depth_of[id(child)] = depth_of.get(id(parent), 0) + 1

    # Arithmetic sitting in a subscript, a slice, or a comparison is choosing
    # *which* element to use, never computing one: ``x[i + 1]``,
    # ``out_ch[:-1]``, ``if i != len(x) - 1``. Collect those positions so the
    # scan below can skip them wholesale.
    index_ctx: set = set()
    for node in ast.walk(tree):
        holders = []
        if isinstance(node, ast.Subscript):
            holders.append(node.slice)
        elif isinstance(node, ast.Slice):
            holders.extend(h for h in (node.lower, node.upper, node.step) if h)
        elif isinstance(node, ast.Compare):
            holders.append(node.left)
            holders.extend(node.comparators)
        for holder in holders:
            for sub in ast.walk(holder):
                index_ctx.add(id(sub))

    ops: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            # Index/shape bookkeeping, not dataflow. Either side being a plain
            # number is enough - ``len(x) - 1``, ``i + 1``, ``ch * 2`` - since a
            # real array combine has arrays on both sides.
            if id(node) in index_ctx or _is_scalar_expr(node):
                continue
            name, sym = _BINOPS[type(node.op)]
            ops.append(_Op(name, sym, name, node.lineno, node.col_offset, 2,
                           in_loop=id(node) in looped,
                           depth=depth_of.get(id(node), 0)))
        elif isinstance(node, ast.Call):
            root = _op_root(node.func)
            if root not in _OP_ROOTS:
                continue                 # a module call or an unrelated helper
            attr = getattr(node.func, "attr", None)
            if attr is None or attr.startswith("_"):
                continue
            ops.append(_Op(
                attr, _OP_SYMBOLS.get(attr, attr), _dotted(node.func),
                node.lineno, node.col_offset, len(node.args),
                in_loop=id(node) in looped,
                depth=depth_of.get(id(node), 0)))

    # ``for layer in self.layers:`` binds a submodule to a plain name, so the
    # call inside the body reads ``layer(x)`` and names no attribute. Map each
    # such loop variable back to the container it iterates, so those calls
    # count as calls to ``self.layers``.
    aliases: dict = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and _op_root(node.iter) == "self"):
            aliases[node.target.id] = _dotted(node.iter)
        # ``for i, layer in enumerate(self.layers)`` / ``zip(...)``
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
            it = node.iter
            if isinstance(it, ast.Call) and getattr(
                    it.func, "id", None) in ("enumerate", "zip"):
                for arg in it.args:
                    if _op_root(arg) != "self":
                        continue
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            aliases.setdefault(elt.id, _dotted(arg))

    def _call_target(call: ast.Call) -> Optional[str]:
        """The ``self.<attr>`` a call resolves to, directly or via an alias."""
        if _op_root(call.func) == "self":
            return _dotted(call.func)
        if isinstance(call.func, ast.Name) and call.func.id in aliases:
            return aliases[call.func.id]
        # ``self.layers[i](x)`` - subscripting a container attribute.
        if isinstance(call.func, ast.Subscript):
            inner = call.func.value
            if _op_root(inner) == "self":
                return _dotted(inner)
        return None

    # Every module call, with its source position, so each op can be tied to
    # the calls it sits with or after.
    calls = sorted(
        ((c.lineno, c.col_offset, _call_target(c))
         for c in ast.walk(tree)
         if isinstance(c, ast.Call) and _call_target(c) is not None),
        key=lambda t: (t[0], t[1]),
    )
    for op in ops:
        # Same statement (``nnx.leaky_relu(self.init(x))``): the op consumes
        # that call's result, whichever order the two appear on the line.
        op.after = tuple(q for ln, _, q in calls if ln == op.lineno)
        # Otherwise the nearest module call above it, which is what produced
        # the value the op is being applied to.
        prior = [q for ln, cl, q in calls
                 if (ln, cl) < (op.lineno, op.col) and ln != op.lineno]
        op.prev_call = prior[-1] if prior else None

    # Statements run top to bottom; within one, the deepest op runs first.
    ops.sort(key=lambda o: (o.lineno, -o.depth, o.col))
    return ops


def _concat_axis(a: Tensor, b: Tensor) -> Optional[int]:
    """The single axis on which ``a`` and ``b`` differ, if there is exactly one.

    Two arrays can be concatenated only when they agree on every other axis, so
    a lone mismatch identifies the join axis. Equal shapes are joinable on any
    axis; report the last, which is the overwhelmingly common channel concat.
    """
    if len(a.shape) != len(b.shape) or not a.shape:
        return None
    diff = [i for i, (x, y) in enumerate(zip(a.shape, b.shape)) if x != y]
    if not diff:
        return len(a.shape) - 1
    return diff[0] if len(diff) == 1 else None


def _concat_compatible(a: Tensor, b: Tensor) -> bool:
    """Could ``a`` and ``b`` be operands of the same concatenate?"""
    return _concat_axis(a, b) is not None



# ---------------------------------------------------------------------------
# Post-trace graph surgery
# ---------------------------------------------------------------------------

def finalize_graph(nodes, edges, producer, child_of, out_sources,
                   merge_candidates):
    """Turn a raw trace into the graph the renderer draws.

    Everything here is pure bookkeeping over the recorded nodes and edges -
    attributing parameters, repairing the dataflow chain where a bare array op
    broke it, synthesizing nodes for those ops, and marking skip edges. None of
    it touches an array, which is why the JAX and PyTorch tracers share it
    verbatim rather than growing two copies that drift apart.

    ``nodes``, ``edges`` and ``out_sources`` are mutated in place; the
    deduplicated edge list is returned.
    """
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
    # the previously completed leaf, which is what actually fed it. This runs
    # *before* op synthesis so the ops below splice onto a complete chain.
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

    # Bare array ops - residual adds, ``jnp.concat``, activations, reshapes -
    # are written as plain expressions, not modules, so nothing above records
    # them. Without them a residual's branches appear to diverge and never
    # rejoin, and a block that concatenates its inputs shows two arrows
    # arriving at a Linear with no sign of where they joined. Synthesize a
    # node per op from the source scan and wire it into the dataflow.
    def _subtree_max(nid: int) -> int:
        """Highest node id inside ``nid``'s subtree - when it finished."""
        return max([nid] + [n.id for n in nodes
                            if _is_descendant(n.id, nid, nodes)])

    def _new_op_node(host: Node, op: str, sym: str, qual: str,
                     tensor: Optional[Tensor], order: int,
                     out_tensor: Optional[Tensor] = None) -> Node:
        # ``out_tensor`` differs from ``tensor`` for ops that change shape - a
        # ``jnp.mean`` over the spatial axes consumes (B,H,W,C) and produces
        # (B,C). Bare ops are never intercepted, so the shape they produced is
        # not observed directly; it is read off whatever consumed the value.
        node = Node(
            id=len(nodes),
            path=f"{host.path}.{sym}" if host.path != "<root>" else sym,
            name=sym,
            cls=qual,
            depth=host.depth + 1,
            parent=host.id,
            params=0,
            own_params=0,
            inputs=[t for t in (tensor,) if t is not None],
            outputs=[t for t in (out_tensor or tensor,) if t is not None],
            config={},
            kind="merge",
            op=op,
            order=order,
        )
        nodes.append(node)
        children.setdefault(host.id, []).append(node.id)
        return node

    for cand in merge_candidates:
        host = nodes[cand["node"]]
        kids_ids = list(child_of.get(host.id, []))
        # ``after`` on each op names the ``self.<attr>(...)`` calls made in the
        # same statement, so an op can be placed relative to real children.
        # Map each child node back to the attribute path used to call it.
        kid_by_attr: dict = {}
        for kid in kids_ids:
            attr = nodes[kid].path
            if attr.startswith(host.path + ".") and host.path != "<root>":
                attr = attr[len(host.path) + 1:]
            kid_by_attr.setdefault(f"self.{attr}", []).append(kid)
            # Children held in a list or dict are reached as ``self.layers.0``
            # but called as ``layer(x)`` inside ``for layer in self.layers``.
            # Register the container name too, so a loop body's ops attach to
            # every element the loop invoked.
            container = attr.rsplit(".", 1)[0]
            if container != attr:
                kid_by_attr.setdefault(f"self.{container}", []).append(kid)

        # The tip of each wire as ops are spliced onto it. ``jnp.tanh(a(x)) *
        # 2 + y`` synthesizes three nodes that must chain - tanh off ``a``,
        # then ``*`` off tanh, then ``+`` off ``*`` - rather than all three
        # hanging off ``a``. Keyed by the child they ultimately derive from.
        tip: dict = {}

        for op_info in cand["ops"]:
            op, sym, qual = op_info.op, op_info.sym, op_info.qual
            # Which child ran in the same statement? That anchors the op in
            # execution order and gives it its operand. Failing that, the
            # nearest module call above it produced the value it consumes.
            anchors = [k for call in op_info.after
                       for k in kid_by_attr.get(call, [])]
            if not anchors and op_info.prev_call:
                anchors = list(kid_by_attr.get(op_info.prev_call, []))
            # A call site inside a loop runs once per iteration; each of the
            # children it produced gets its own copy of the op.
            if anchors and not op_info.in_loop:
                anchors = anchors[-1:]

            if op in _JOIN_OPS and not anchors:
                # A join of several *incoming* values, before any child runs -
                # ``jnp.concat([motion, id])`` at the top of an adapter. Its
                # operands are the host's own inputs, and everything the host
                # fed downstream should hang off the join instead.
                # Keep each operand's own tensor alongside its producer, so the
                # incoming arrows report pre-concat shapes rather than the
                # joined width.
                src_tensor: dict = {}
                srcs = []
                for aid, tsr in zip(cand["in_ids"], cand["in_arrays"]):
                    s = producer.get(aid)
                    if s is None or s == host.id or s in kids_ids:
                        continue
                    if s not in src_tensor:
                        src_tensor[s] = tsr
                        srcs.append(s)
                if len(srcs) < 2:
                    continue            # not actually joining two branches
                # The join's result is what the child that consumed it received.
                # Prefer the earliest child *in call order* whose own input is
                # wider than every operand - that is the concatenated array.
                # ``min(kids_ids)`` alone assumes the first child consumes the
                # join, which is not true in general.
                joined = None
                op_widths = [t.shape[-1] for t in src_tensor.values()
                             if t is not None and t.shape]
                total = sum(op_widths) if op_widths else None
                for kid in sorted(kids_ids):
                    cand_t = next(iter(nodes[kid].inputs), None)
                    if cand_t is None or not cand_t.shape:
                        continue
                    if total is not None and cand_t.shape[-1] == total:
                        joined = cand_t
                        break
                if joined is None and kids_ids:
                    joined = next(iter(nodes[min(kids_ids)].inputs), None)
                if joined is None:
                    joined = cand["in_tensors"][0] if cand["in_tensors"] else None
                first_kid = min(kids_ids) if kids_ids else host.id
                node = _new_op_node(host, op, sym, qual, joined,
                                    order=first_kid - 1)
                srcs_set = set(srcs)
                # The children that consumed those raw inputs now consume the
                # join - that is what the source actually does. Rewire before
                # adding the operand edges so they are not themselves caught.
                rewired = False
                for e in edges:
                    if (e.src in srcs_set and e.dst != node.id
                            and (e.dst in kids_ids
                                 or any(_is_descendant(e.dst, k, nodes)
                                        for k in kids_ids))):
                        e.src = node.id
                        rewired = True
                # The first child usually received the joined array directly,
                # so no edge existed to rewire - the value never passed
                # through another module. Wire the join to it explicitly.
                if not rewired and kids_ids:
                    edges.append(Edge(node.id, first_kid,
                                      joined or Tensor((), "")))
                for i, src in enumerate(srcs):
                    edges.append(Edge(
                        src, node.id,
                        src_tensor.get(src) or joined or Tensor((), ""),
                        skip=i > 0))
                continue

            if not kids_ids:
                continue                # a leaf module: no flow to splice into

            merged = False
            if cand["tail_combine"] and (op in _JOIN_OPS
                                         or op in ("add", "sub", "mul", "div")):
                # A trailing combine that closes a residual: ``out + x``.
                # Operands are the children whose output nothing consumed -
                # the ends of each branch. In ``conv2(conv1(x)) + shortcut(x)``
                # that is {conv2, shortcut}: conv1's output was eaten by conv2.
                consumed = {e.src for e in edges if e.dst in kids_ids}
                branch_ends = [k for k in kids_ids if k not in consumed]
                # Plus any value the host received that no child produced (a
                # residual adding the raw input). Skip it when a branch end
                # already consumed that value - then the branch *is* that path
                # (``shortcut(x)``) and adding it would double-count.
                in_end_subtree = set(branch_ends)
                for n in nodes:
                    if any(_is_descendant(n.id, b, nodes) for b in branch_ends):
                        in_end_subtree.add(n.id)
                fed_ends = {e.src for e in edges
                            if e.dst in in_end_subtree
                            and e.src not in kids_ids}
                carried = [s for s in (producer.get(i) for i in cand["in_ids"])
                           if s is not None and s != host.id
                           and s not in kids_ids and s not in fed_ends]
                # Fewer than two operands means nothing converged here, so
                # this is not a residual merge - fall through and draw it as
                # an ordinary elementwise op instead of dropping it.
                if len(branch_ends) + len(carried) >= 2:
                    last_kid = branch_ends[-1] if branch_ends else kids_ids[-1]
                    node = _new_op_node(host, op, sym, qual, cand["out"],
                                        order=_subtree_max(last_kid))
                    tensor = cand["out"] or Tensor((), "")
                    operands = list(dict.fromkeys(branch_ends + carried))
                    for i, src in enumerate(operands):
                        edges.append(Edge(src, node.id, tensor, skip=i > 0))
                    # Whatever consumed the host's output now consumes the op.
                    # The value leaves the host's *last leaf*, not the host node
                    # itself, so matching only ``host.id`` left the original edge
                    # in place: a second arrow ran from deep inside the block
                    # straight to the next module, bypassing the residual and its
                    # activation. Redirect everything leaving the subtree.
                    h_inside = {host.id} | {n.id for n in nodes
                                            if _is_descendant(n.id, host.id, nodes)}
                    for e in edges:
                        if (e.src in h_inside and e.dst != node.id
                                and e.dst not in h_inside):
                            e.src = node.id
                    for o in out_sources:
                        if o["src"] in h_inside:
                            o["src"] = node.id
                    # The combine is now the tip of each branch it closed, so an
                    # op written *after* it (``out = out + x`` then
                    # ``out = leaky_relu(out)``) chains downstream instead of
                    # splicing onto the branch and landing upstream of it.
                    for b in operands:
                        tip[b] = node.id
                    merged = True

            if merged or not anchors:
                continue                # placed, or cannot be placed reliably

            # An elementwise op applied to a child's output - an activation
            # (``x = nnx.leaky_relu(x)`` after ``x = layer(x)``) or a reshape.
            # Splice it onto the wire leaving that child: everything
            # downstream of the child now reads from the op instead.
            for src_kid in anchors:
                # Splice onto the current tip of this wire, so several ops in
                # one statement form a chain instead of a fan.
                prev = tip.get(src_kid, src_kid)
                tensor = next(iter(nodes[prev].outputs), None)
                # A reducing or reshaping op (``jnp.mean(x, axis=(1, 2))``)
                # outputs a different shape than it consumed, and bare ops are
                # never intercepted so that shape was never recorded. The
                # module that consumed the result did record it as its own
                # input - take it from there rather than reporting the
                # unchanged input shape, which would simply be wrong.
                out_t = None
                if op not in _SHAPE_PRESERVING:
                    for e in edges:
                        if e.src != prev:
                            continue
                        cons = next(iter(nodes[e.dst].inputs), None)
                        if cons is not None:
                            out_t = cons
                            break
                node = _new_op_node(host, op, sym, qual, tensor,
                                    order=_subtree_max(src_kid),
                                    out_tensor=out_t)
                # Downstream consumers now read from the op. A container returns
                # the array its last leaf produced, so the recorded edge often
                # leaves that leaf (``changer.0.norm``) rather than the anchor
                # itself (``changer.0``). Matching only the anchor left the op
                # dangling with no output while the real flow bypassed it - the
                # module appeared wired straight through, short-circuiting the
                # join. Redirect anything leaving the anchor's subtree, except
                # edges that stay inside it.
                inside = {prev} | {n.id for n in nodes
                                   if _is_descendant(n.id, prev, nodes)}
                for e in edges:
                    if (e.src in inside and e.dst != node.id
                            and e.dst not in inside):
                        e.src = node.id
                for o in out_sources:
                    if o["src"] in inside:
                        o["src"] = node.id
                edges.append(Edge(prev, node.id, tensor or Tensor((), "")))

                # A join is not unary. ``jnp.concat([out, x[i+1]])`` follows a
                # child, so it lands here rather than in the no-anchor branch
                # above, but it still consumes a second value - typically one of
                # the host's own inputs that no child produced. Without this the
                # node draws a single arrow and the joined operand never
                # connects, which also makes its width look unexplained.
                if op in _JOIN_OPS and tensor is not None:
                    used = {e.src for e in edges if e.dst == node.id}
                    extra = []
                    for aid, tsr in zip(cand["in_ids"], cand["in_arrays"]):
                        s = producer.get(aid)
                        if (s is None or s == host.id or s == node.id
                                or s in used or s in kids_ids):
                            continue
                        # A genuine operand agrees on every axis but the one
                        # being joined. Matching on rank alone is far too loose:
                        # in a pyramid every feature map is rank 4, so all of
                        # them looked like operands and the join lit up three
                        # unrelated inputs instead of the one it consumes.
                        if not _concat_compatible(tsr, tensor):
                            continue
                        extra.append((s, tsr))
                    # ``jnp.concat([out, feat])`` takes exactly one partner. If
                    # several inputs are shape-compatible the trace cannot say
                    # which, and guessing draws confidently wrong arrows - so
                    # wire it only when the choice is unambiguous.
                    if len(extra) == 1:
                        s, tsr = extra[0]
                        edges.append(Edge(s, node.id, tsr, skip=True))
                        used.add(s)
                        # The join widens the concat axis; the recorded output
                        # was read off a consumer and is often the pre-join
                        # shape. Correct it from the operands actually wired.
                        ax = _concat_axis(tsr, tensor)
                        if ax is not None:
                            dims = list(tensor.shape)
                            dims[ax] = tensor.shape[ax] + tsr.shape[ax]
                            node.outputs = [Tensor(tuple(dims), tensor.dtype)]

                tip[src_kid] = node.id

    # An edge carries whatever its source emits. Splicing an op into the flow
    # reassigns ``src`` on the edges downstream of it, but those edges kept the
    # tensor recorded for the *old* producer - so an arrow out of an activation
    # could advertise the shape of the module it points at rather than the value
    # travelling the wire. Re-label from the source now that every op is placed.
    for e in edges:
        if e.src < 0 or e.src >= len(nodes):
            continue                     # an input pill: nothing to read off
        out = nodes[e.src].outputs
        if out:
            e.tensor = out[0]

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

    # Deduplicate: one module pair can exchange several arrays.
    seen = set()
    unique_edges = []
    for e in edges:
        sig = (e.src, e.dst)
        if sig in seen:
            continue
        seen.add(sig)
        unique_edges.append(e)

    # Mark skip connections by fan-out: when one value feeds several consumers,
    # the immediate next one is the "main" path and any consumer further along
    # is a shortcut that bypasses it.
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
    return unique_edges


__all__ = [
    "Tensor", "Node", "Edge", "Graph",
    "INPUT_NODE", "IN_BASE", "finalize_graph",
    "_is_descendant", "_scan_ops", "_concat_axis", "_concat_compatible",
    "_Op", "_BINOPS", "_OP_ROOTS", "_OP_SYMBOLS", "_SHAPE_PRESERVING",
    "_INTERESTING",
]
