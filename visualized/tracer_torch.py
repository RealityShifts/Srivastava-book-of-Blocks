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
    _scan_ops_deep,
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

#: Bare ops worth drawing as their own node, labelled with the op's own name -
#: no glyphs: one symbol per op family made distinct operations look identical
#: (``interpolate`` and ``grid_sample`` both rendered as an arrow, though only
#: one of them changes resolution). A curated allowlist rather than "every
#: torch function": a graph with a node per ``__getitem__`` and ``.to(dtype)``
#: buries the structure it exists to show.
#: Names are ``func.__name__`` as ``TorchFunctionMode`` reports them.
_TRACED_OPS = {
    "interpolate", "grid_sample", "upsample",
    "cat", "concat", "concatenate", "stack",
    "add", "__add__", "sub", "__sub__",
    "mul", "__mul__", "div", "__truediv__",
    "matmul", "__matmul__",
    "mean", "sum",
    "sigmoid", "tanh", "softmax", "logit",
    "relu", "leaky_relu", "gelu", "silu",
    "reshape", "view", "flatten", "permute",
    "transpose", "squeeze", "unsqueeze",
    "chunk", "split", "pad", "clamp",
    "normalize", "layer_norm",
    # Convolution called functionally rather than through nn.Conv2d. A module
    # that keeps its kernel in a *buffer* - AliasFreeActivation's windowed-sinc
    # up/down filters - has no child module for the hooks to see, so without
    # this its entire body is invisible and it renders as a bare constant.
    "conv1d", "conv2d", "conv3d", "conv_transpose2d",
    # Grid builders. These construct a value rather than transform one, so it
    # is tempting to treat them as invisible - but then whatever consumes the
    # grid shows an operand arriving from an anonymous constant, and the
    # coordinate grid is precisely what makes a warp readable.
    "meshgrid", "linspace", "arange",
}


#: Ops that only shuffle metadata. They are followed for *edges* - the value
#: keeps flowing - but never get a node, or every ``.expand`` broadcasting a
#: style code across a feature map becomes a box.
_INVISIBLE_OPS = {
    "expand", "expand_as", "to", "detach", "contiguous", "type_as", "float",
    "half", "clone", "requires_grad_",
    "tensor", "as_tensor", "empty", "zeros", "ones", "full", "zeros_like",
    "ones_like", "__get__", "size", "dim", "item", "numel",
    # Allocation, not computation: never drawn, but must not sever the chain
    # either - AliasFreeActivation builds its zero-stuffed tensor this way.
    "new_zeros", "new_ones", "new_empty", "new_full", "new_tensor",
    # Slices and views re-present an existing value rather than computing a new
    # one. Untagged, ``flow[:, 0]`` severs the flow field's provenance and the
    # add that consumes it appears to take an operand from nowhere.
    "__getitem__", "select", "narrow", "index_select", "unbind", "split_with_sizes",
}


class _OpTracer(torch.overrides.TorchFunctionMode):
    """Records bare tensor ops as graph nodes, wiring edges by tensor identity.

    Every op's result is tagged with the node that produced it, and every op
    reads the tags of its arguments to find its own producers. That is the
    whole mechanism: an edge exists because the *same tensor object* left one
    node and entered another, which is a fact about the run rather than an
    inference from source layout.

    The alternative this replaces - parsing ``forward``'s AST to find bare ops,
    then anchoring each to a nearby module call and reconstructing its shapes
    from whatever neighbour was reachable - is right only while a value flows
    straight from one call to the next. Wherever it does not, the graph goes
    wrong in ways no local fix reaches: a flow field resized inside a decoder
    loop was produced levels above, so "the nearest preceding call" is the
    trunk, and the op ends up carrying the trunk's shapes and, once spliced
    onto the host's inputs, the trunk's edges too - one node with an arrow in
    from every pyramid level and an arrow out to nearly every module below it.

    Tagging cannot express that error. The op sees the exact tensors it was
    handed, so its shapes are what it actually resized and its edges lead to
    whoever actually produced them.

    Module boundaries still come from the forward hooks: ``scope`` is the node
    id of the innermost module executing, so each op lands inside the module
    whose body ran it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.nodes: list = []          # op nodes, in call order
        self.edges: list = []          # (src_node, dst_node, Tensor)
        self.scope: list = []          # module node ids, innermost last
        self.enabled = False
        # Depth guard. ``TorchFunctionMode`` is re-entrant: ``F.interpolate``
        # dispatches to further torch functions, and the mode sees those too.
        # Recording them would nest one op inside another and pair an outer
        # call's input with an inner call's output. Only the outermost call
        # becomes a node; everything under it runs untraced.
        self._depth = 0
        # id(tensor) -> node id that produced it. Keyed by identity, with the
        # tensors kept alive: CPython reuses addresses, so a freed intermediate
        # would otherwise hand its id to an unrelated tensor and fabricate an
        # edge.
        self.tags: dict = {}
        self._keepalive: list = []
        #: Called as ``on_produce(tensor, tmp_id)`` when an op produces a
        #: value. The tracer points the shared producer map at the op, so a
        #: module consuming the result resolves to the op rather than to
        #: whatever module last handled the value.
        self.on_produce = None
        #: node id -> class name, leaf modules only. Lets an op recognise that
        #: it *is* its enclosing module - ``nn.LeakyReLU`` running
        #: ``leaky_relu`` - rather than a bare op written beside one.
        self.leaf_cls: dict = {}

    def tag(self, obj: Any, node_id: int) -> None:
        """Mark every tensor in ``obj`` as produced by ``node_id``."""
        for leaf in _leaves(obj):
            if isinstance(leaf, torch.Tensor):
                self.tags[id(leaf)] = node_id
                self._keepalive.append(leaf)

    def source_of(self, tensor: torch.Tensor):
        return self.tags.get(id(tensor))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        # In-place assignment - ``stuffed[:, :, ::2, ::2] = x``. It does not fit
        # the passthrough below: those forward a tag from the first argument to
        # the *result*, but ``__setitem__`` returns None and the value that
        # matters is the one being assigned. The tag has to move from the
        # source (args[2]) onto the mutated target (args[0]) instead.
        #
        # Without this the target keeps whatever tag its allocation had - none,
        # for a fresh ``new_zeros`` - so the real input is severed and the
        # enclosing module traces as a constant with no incoming edge. This is
        # exactly how AliasFreeActivation's input enters it.
        if self.enabled and name == "__setitem__" and len(args) >= 3:
            out = func(*args, **kwargs)
            target, value = args[0], args[2]
            if isinstance(target, torch.Tensor) and isinstance(value, torch.Tensor):
                src = self.tags.get(id(value))
                if src is not None:
                    self.tag(target, src)
                    if self.on_produce is not None:
                        self.on_produce(target, src)
            return out

        if not self.enabled or name in _INVISIBLE_OPS:
            out = func(*args, **kwargs)
            # A metadata-only op still passes the value along, so the result
            # inherits its input's producer. Without this a ``.expand`` between
            # two real ops severs the chain.
            if name in _INVISIBLE_OPS:
                src = next((self.tags.get(id(a)) for a in _leaves(args)
                            if isinstance(a, torch.Tensor)
                            and id(a) in self.tags), None)
                if src is not None:
                    self.tag(out, src)
            return out

        if name not in _TRACED_OPS or self._depth:
            return func(*args, **kwargs)

        # Producers are read *before* the call: an in-place op would otherwise
        # overwrite the tag on its own input and lose the incoming edge.
        srcs = []
        for a in _leaves(args):
            if not isinstance(a, torch.Tensor):
                continue
            src = self.tags.get(id(a))
            if src is not None:
                srcs.append((src, Tensor(tuple(a.shape), _dtype(a))))

        first = next((a for a in _leaves(args)
                      if isinstance(a, torch.Tensor)), None)
        # Weight math is not dataflow. ``EqualLinear`` scales its weight matrix
        # by a constant on every call, and ``ModulatedConv2d`` reshapes and
        # normalizes its kernel - real torch ops on real tensors, but on
        # *parameters*, so they describe how a layer builds its weights rather
        # than how activations move. Drawing them adds a node per layer whose
        # output is a ``(512, 512)`` kernel with nowhere to go. An op none of
        # whose inputs carry a producer tag is exactly this: every activation
        # traces back to a module output or the model input.
        if first is not None and not srcs and not first.requires_grad:
            pass          # constant folding on plain tensors: still dataflow
        elif not srcs and any(isinstance(a, torch.nn.Parameter)
                              for a in _leaves(args)):
            return func(*args, **kwargs)

        self._depth += 1
        try:
            out = func(*args, **kwargs)
        finally:
            self._depth -= 1
        res = next((t for t in _leaves(out)
                    if isinstance(t, torch.Tensor)), None)
        if res is None or first is None:
            return out
        # A step on the weight path, not the activation path.
        # ``ModulatedConv2d`` reshapes its kernel to (out, in, kh, kw) and adds
        # a style bias before convolving; those are real ops on real tensors,
        # but what they produce is a *filter*, and drawing it puts a node in
        # the diagram whose value never reaches another module. Activations in
        # this codebase are NCHW or (N, C) and always carry the batch axis, so
        # a 4-D result whose leading dim is not the batch is a kernel.
        if (len(res.shape) == 4 and first.shape
                and res.shape[0] != first.shape[0]):
            return out

        # Skip an op that is merely the body of an activation module: the
        # module already has a node of its own, so recording the call inside
        # it draws the same step twice - once as ``LeakyReLU``, once as a
        # ``leaky_relu`` op hanging off it with nothing downstream. Detected by
        # the scope being a leaf module with no children of its own whose class
        # name matches the op.
        scope = self.scope[-1] if self.scope else None
        if scope is not None and scope in self.leaf_cls:
            cls_name = self.leaf_cls[scope].lower().replace("_", "")
            if cls_name == name.lower().replace("_", ""):
                # The module itself is the producer of this value. Tagging it
                # here rather than falling through matters: the module hook
                # that runs on exit only claims tensors nothing has tagged, so
                # a value left untagged by the skip would reach the next
                # module with no producer and both sides lose the edge.
                self.tag(out, scope)
                if self.on_produce is not None:
                    for leaf in _leaves(out):
                        if isinstance(leaf, torch.Tensor):
                            self.on_produce(leaf, scope)
                return out

        # Every operand belongs on a wire. An op against a value nothing
        # produced - ``resize_flow``'s ``resized * scale``, a modulated conv's
        # own weight - would otherwise draw a single arrow in, leaving
        # "times what?" unanswerable from the diagram. Rather than describe the
        # missing side in text on the card, give it a node of its own and a
        # real edge, so the graph stays a graph. Parameters are distinguished
        # from plain constants because a trainable weight and a hard-coded
        # scale factor are different things to a reader.
        extras = []
        tagged = {id(a) for a in _leaves(args) if id(a) in self.tags}
        for a in _leaves(args):
            if id(a) in tagged:
                continue
            extras.append({
                "kind": ("parameter" if isinstance(a, torch.nn.Parameter)
                         else "constant"),
                "tensor": Tensor(tuple(a.shape), _dtype(a)),
                "value": ([round(float(v), 4) for v in a.flatten().tolist()]
                          if a.numel() <= 4 else None),
            })
        # ``_leaves`` keeps only tensors, but a bare Python scalar is an
        # operand too - ``2.0 * sx`` normalizing pixel coordinates - and is the
        # whole answer to "times what?" on that card.
        try:
            flat = pytree.tree_leaves(args)
        except Exception:                  # pragma: no cover - exotic types
            flat = list(args)
        scalars = [a for a in flat
                   if isinstance(a, (int, float)) and not isinstance(a, bool)]
        if scalars:
            # One node for all of a call's scalars, labelled generically -
            # torchvista's "N scalars". Naming each by its value spawned a node
            # called "191" and another called "767" for what are really just
            # ``interpolate``'s size arguments, and a graph full of loose
            # integers hides the ops it is supposed to show. The values live in
            # the node's config, where they are still readable on the card.
            extras.append({
                "kind": "scalar",
                "tensor": Tensor((), "float32"),
                "value": scalars[0] if len(scalars) == 1 else scalars,
                "label": ("scalar" if len(scalars) == 1
                          else f"{len(scalars)} scalars"),
            })

        nid = -(len(self.nodes) + 1)      # provisional; renumbered on merge
        self.nodes.append({
            "extras": extras,
            "tmp_id": nid,
            "op": name,
            "sym": name,
            "scope": self.scope[-1] if self.scope else None,
            "in": Tensor(tuple(first.shape), _dtype(first)),
            "out": Tensor(tuple(res.shape), _dtype(res)),
            "n_in": len(srcs),
        })
        for src, tsr in srcs:
            self.edges.append((src, nid, tsr))
        self.tag(out, nid)
        # Point the *shared* producer map at this op too. The module hooks
        # resolve their inputs through that map, so without this a module
        # consuming an op's result finds no producer at all and the op is left
        # with no consumer - the two halves of the same missing edge.
        if self.on_produce is not None:
            for leaf in _leaves(out):
                if isinstance(leaf, torch.Tensor):
                    self.on_produce(leaf, nid)
        return out


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
    # Observed shapes for the bare ops, so finalize_graph can label them from
    # what actually ran instead of inferring from neighbouring modules.
    # Bare ops become nodes here, wired by tensor identity. The module hooks
    # below push/pop its scope and tag module outputs, so an op consuming a
    # module's result finds that module as its producer.
    op_tracer = _OpTracer()

    def _flops_now() -> int:
        if counter is None:
            return -1
        try:
            return counter.get_total_flops()
        except Exception:              # pragma: no cover - counter API drift
            return -1

    def _op_produced(leaf, tmp_id: int) -> None:
        """An op created ``leaf``; record it as the producer for the hooks.

        Writing into the same map the module hooks read is what makes an op a
        first-class producer: a module consuming the op's result resolves to
        the op itself, not to whatever module last handled the value, and no
        edge has to be rewired afterwards.
        """
        producer[id(leaf)] = tmp_id
        keepalive.append(leaf)

    op_tracer.on_produce = _op_produced

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
        op_tracer.scope.append(node.id)
        if not list(mod.children()):
            op_tracer.leaf_cls[node.id] = type(mod).__name__
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
        if op_tracer.scope:
            op_tracer.scope.pop()
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
            ops_cache[cls] = _scan_ops_deep(cls.forward, cls)
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
        # Same value, tagged for the op tracer: a bare op consuming this
        # module's result must find the module as its producer. claim=False
        # applies here too - re-tagging a passthrough would credit the wrapper
        # instead of the child that computed it.
        for leaf in _leaves(out):
            if isinstance(leaf, torch.Tensor) and id(leaf) not in op_tracer.tags:
                op_tracer.tag(leaf, nid)

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
        src = INPUT_NODE if i == 0 else IN_BASE - i
        _register(leaf, src)
        op_tracer.tag(leaf, src)

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
                with counter, op_tracer:
                    op_tracer.enabled = True
                    result = model(*args, **kwargs)
                    op_tracer.enabled = False
                total_flops = _flops_now()
            else:
                with op_tracer:
                    op_tracer.enabled = True
                    result = model(*args, **kwargs)
                    op_tracer.enabled = False
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

    # Fold the traced bare ops in as real nodes. They were numbered negatively
    # while running - their ids had to exist before the module list was final -
    # so renumber onto the end and rewrite the edges that referenced them.
    tmp_to_id: dict = {}
    for rec in op_tracer.nodes:
        host = rec["scope"]
        host_node = nodes[host] if host is not None and host < len(nodes) else None
        nid = len(nodes)
        tmp_to_id[rec["tmp_id"]] = nid
        nodes.append(Node(
            id=nid,
            path=(f"{host_node.path}.{rec['op']}"
                  if host_node is not None and host_node.path != "<root>"
                  else rec["op"]),
            name=rec["sym"],
            cls=rec["op"],
            depth=(host_node.depth + 1) if host_node is not None else 0,
            parent=host,
            params=0, own_params=0,
            inputs=[rec["in"]], outputs=[rec["out"]],
            config={}, kind="merge", op=rec["op"],
            # Ops own no parameters, and the FLOP counter already attributes
            # their cost to the enclosing module, so leaving these empty keeps
            # the module roll-up exact rather than double-counting.
            order=nid, flops=-1, own_flops=0,
        ))
    # Edges recorded against provisional ids, plus the ones the module hooks
    # wrote while the producer map still held a provisional id.
    # Constants, parameters and scalars become nodes of their own, each with an
    # edge into the op that consumed it. This is what keeps a binary op looking
    # like a binary op: ``grid_x + flow[:, 0]`` shows two arrows in, and
    # ``resized * scale`` shows the scale factor as a labelled source rather
    # than as prose on the card.
    for rec in op_tracer.nodes:
        op_id = tmp_to_id[rec["tmp_id"]]
        host = nodes[op_id].parent
        for extra in rec.get("extras", ()):
            kind = extra["kind"]
            val = extra["value"]
            # Generic label, value in the config: a card reading "constant"
            # with ``value: [2.0, 2.0]`` beneath it stays scannable, where a
            # node *named* "[2.0, 2.0]" competes with the ops for attention.
            label = extra.get("label") or kind
            cid = len(nodes)
            nodes.append(Node(
                id=cid,
                path=(f"{nodes[host].path}.{label}"
                      if host is not None and nodes[host].path != "<root>"
                      else label),
                name=label,
                cls=kind,
                depth=nodes[host].depth + 1 if host is not None else 0,
                parent=host,
                params=0, own_params=0,
                inputs=[], outputs=[extra["tensor"]],
                config=({} if val is None else {"value": val}),
                kind="merge", op=kind,
                order=cid, flops=-1, own_flops=0,
            ))
            edges.append(Edge(cid, op_id, extra["tensor"]))

    for src, dst, tsr in op_tracer.edges:
        s_id = tmp_to_id.get(src, src)
        d_id = tmp_to_id.get(dst, dst)
        if s_id != d_id:
            edges.append(Edge(s_id, d_id, tsr))
    for e in edges:
        e.src = tmp_to_id.get(e.src, e.src)
        e.dst = tmp_to_id.get(e.dst, e.dst)
    for key, val in list(producer.items()):
        if val in tmp_to_id:
            producer[key] = tmp_to_id[val]

    # An op whose result its enclosing module returns has no consumer *inside*
    # that module, so it reads as a branch that stops - even though the value
    # very much continues, along the module's own outgoing edge. Hand those
    # edges over: whatever consumed the module consumed this op's tensor.
    op_ids = set(tmp_to_id.values())
    by_src: dict = {}
    for e in edges:
        by_src.setdefault(e.src, []).append(e)
    for op_id in op_ids:
        node = nodes[op_id]
        host = node.parent
        if host is None or any(e.src == op_id for e in edges):
            continue
        out_shape = tuple(node.outputs[0].shape) if node.outputs else None
        for e in by_src.get(host, []):
            if out_shape is None or tuple(e.tensor.shape) == out_shape:
                e.src = op_id
    for o in out_sources:
        if o["src"] in tmp_to_id:
            o["src"] = tmp_to_id[o["src"]]

    # channel_axis=1: everything here is NCHW, so a concat of two equal-shaped
    # feature maps widens dim 1. The default (-1) is the Flax tracer's NHWC.
    #
    # merge_candidates is deliberately empty: it drives finalize_graph's
    # AST-based op synthesis, which _OpTracer replaces outright. Running both
    # would draw every bare op twice - once as observed, once as reconstructed.
    # The JAX tracer still passes its own candidates and keeps that path.
    unique_edges = finalize_graph(nodes, edges, producer, child_of,
                                  out_sources, [],
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
