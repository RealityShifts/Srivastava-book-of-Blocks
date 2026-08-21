"""Drop the pass-through pills PyTorch draws and Flax does not.

Both tracers hook module calls, but the two frameworks disagree about what a
module *is*. Flax's ``get_activation`` returns a plain function and its "no
norm" case is a bare lambda, so neither appears in a trace. PyTorch spells both
as ``nn.Module`` - ``nn.ReLU``, ``nn.LeakyReLU``, ``nn.Tanh``, ``nn.Identity`` -
so the tracer faithfully draws a pill for each one.

The effect compounds with depth. On a 71 M-parameter generator that is ~125
extra pills carrying no parameters and no shape change - 54 ``Identity`` (one
per ``activation="identity"`` and per ``norm="none"``) and 65 activations. The
graph is still correct, just unreadable next to its Flax twin.

The rule here is deliberately narrower than "hide activations": a node is
removed only when it owns no parameters **and** its output shape equals its
input shape. That is precisely "this node moved no data anywhere". It drops
``Identity`` and every pointwise activation, while keeping something like
``nn.Upsample``, which owns no parameters either but does change the shape and
is therefore telling you something.

Nothing is lost: before an activation is dropped, its class name is folded into
its parent's ``config``, so the details panel for a ``ConvBlock`` still says
which nonlinearity it applies. That makes the pruned PyTorch graph strictly
more informative than the Flax one, which never had that information to show.
"""

from __future__ import annotations

import dataclasses

# Everything torch spells as a module but Flax spells as a function. Restricted
# to shape-preserving, parameter-free layers - see the module docstring.
PASSTHROUGH_CLASSES = frozenset({
    "Identity",
    "ReLU", "ReLU6", "LeakyReLU", "PReLU", "RReLU",
    "GELU", "SiLU", "Mish", "ELU", "CELU", "SELU",
    "Tanh", "Sigmoid", "Hardsigmoid", "Hardswish", "Hardtanh",
    "Softplus", "Softsign", "LogSigmoid",
    "Dropout", "Dropout1d", "Dropout2d", "Dropout3d",
})


def is_passthrough(node) -> bool:
    """True when ``node`` owns no parameters and did not change the shape."""
    if node.kind != "module" or node.cls not in PASSTHROUGH_CLASSES:
        return False
    if node.own_params:
        return False
    # A node with no recorded shapes tells us nothing either way; treat the
    # class allowlist as sufficient there rather than keeping mystery pills.
    if not node.inputs or not node.outputs:
        return True
    return node.inputs[0].shape == node.outputs[0].shape


def prune_passthrough(graph):
    """Return ``graph`` with parameter-free, shape-preserving modules removed.

    Mutates and returns the graph in place. Edges that ran *through* a removed
    node are reconnected to its nearest surviving producer, so the dataflow
    stays connected rather than developing holes.
    """
    by_id = {n.id: n for n in graph.nodes}
    doomed = {n.id for n in graph.nodes if is_passthrough(n)}
    if not doomed:
        return graph

    # Fold each dropped activation's identity into its parent's details panel
    # before it disappears. Identity is a genuine no-op, so it is not recorded.
    for node_id in doomed:
        node = by_id[node_id]
        parent = by_id.get(node.parent)
        if parent is not None and node.cls != "Identity":
            parent.config.setdefault("activation", node.cls)

    incoming: dict = {}
    for edge in graph.edges:
        incoming.setdefault(edge.dst, []).append(edge)

    def surviving_sources(node_id: int, seen: frozenset = frozenset()) -> list:
        """Nearest producers of ``node_id`` that outlive the prune.

        Recursive because dropped nodes chain - a ``ConvBlock`` whose norm is
        ``Identity`` and whose activation is ``ReLU`` puts two back to back.
        ``seen`` guards a malformed cyclic graph rather than recursing forever.
        """
        if node_id in seen:
            return []
        out = []
        for edge in incoming.get(node_id, []):
            if edge.src in doomed:
                out.extend(surviving_sources(edge.src, seen | {node_id}))
            else:
                out.append(edge.src)
        return out

    rewired, seen_pairs = [], set()
    for edge in graph.edges:
        if edge.dst in doomed:
            continue                      # its consumers get rewired instead
        if edge.src not in doomed:
            if (edge.src, edge.dst) not in seen_pairs:
                seen_pairs.add((edge.src, edge.dst))
                rewired.append(edge)
            continue
        for src in surviving_sources(edge.src):
            if (src, edge.dst) in seen_pairs:
                continue
            seen_pairs.add((src, edge.dst))
            # Shape is preserved across a dropped node by construction, so the
            # original edge's tensor still describes what flows along the new one.
            rewired.append(dataclasses.replace(edge, src=src))
    graph.edges = rewired

    # A model returning straight out of an activation (a tanh-terminated
    # generator does) would otherwise point its output pill at a dead node.
    for out in graph.output_sources:
        if out.get("src") in doomed:
            sources = surviving_sources(out["src"])
            out["src"] = sources[0] if sources else None

    # Defensive: these classes have no children in practice, but a graph with a
    # dangling parent pointer renders as a detached subtree.
    for node in graph.nodes:
        while node.parent in doomed:
            node.parent = by_id[node.parent].parent

    graph.nodes = [n for n in graph.nodes if n.id not in doomed]
    return graph
