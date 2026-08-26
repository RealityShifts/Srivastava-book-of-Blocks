# visualized - interactive graphs for Flax NNX models

A TorchVista-style visualizer for this repo's Flax blocks. It runs one real
forward pass, records the module graph with concrete shapes and parameter
counts, and writes a **single self-contained HTML file** - no CDN, no build
step, no server.

```
Blocks/visualized/
    tracer.py     capture the graph from a live forward pass
    renderer.py   graph -> standalone interactive HTML
    visualize.py  visualize() / visualize_notebook() / summary()
    cli.py        python -m Blocks.visualized.cli ...
    examples.py   runnable end-to-end examples
```

## Quick start

```python
import jax
from flax import nnx
from Blocks.visualized import visualize

model = MyModel(rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(0), (1, 128, 128, 3))

visualize(model, x, path="model.html", open_browser=True)
```

In a notebook (renders inline, no file written):

```python
from Blocks.visualized import visualize_notebook
visualize_notebook(model, x)
```

Quick text tree in a terminal:

```python
from Blocks.visualized import summary
print(summary(model, x))
```

From the command line:

```bash
python -m Blocks.visualized.cli Blocks.flax_blocks.core_blocks:ResidualBlock \
    --args 8 16 --input 2,16,16,8 --out res.html --text --open
```

`--args`/`--kwargs` construct the model (`rngs` is supplied automatically),
`--input` gives one shape per input tensor.

## In the page

| Action | Result |
| --- | --- |
| scroll | zoom |
| drag | pan |
| click a node | select it; neighbours stay lit, everything else dims |
| click a group's border or label | collapse that group |
| double-click a collapsed node | expand it |
| `+ depth` / `- depth` | expand or collapse one nesting level at a time |
| `Fit` | re-centre |

Nodes are coloured by module family (conv, norm, attention, linear, ...) and
badged with their parameter count. Dashed **orange** edges are skip /
residual connections; solid edges are ordinary dataflow. The sidebar shows
per-module input/output shapes, parameter totals, share of the whole model,
and constructor config (kernel size, strides, epsilon, ...).

The blue pill at the top is the model input. **Green pills at the bottom are
the model's outputs** - one per returned array, each wired in green to the
module that actually produced it. A model returning a tuple or a list of
feature maps gets one pill per entry, numbered in return order; click any of
them to see its shape and its producing module.

**Bare array ops appear as orange circles.** Anything written as a plain
expression rather than a module - a residual `out = out + x`, a
`jnp.concat([a, b])`, an activation like `nnx.leaky_relu(x)`, a reshape - is
invisible to a module-level trace: the branches would appear to diverge and
never rejoin, and a block that concatenates its inputs would show two arrows
arriving at a `Linear` with no sign of where they joined. Each op therefore
gets a synthesized node carrying no parameters, labelled with a glyph (`+`,
`⧺`, `×`, `∑`, `↳`, ...) or the function's own name, and spliced into the
dataflow where the source runs it. The details panel shows the qualified
call (`jnp.concat`, `nnx.leaky_relu`) and the shape it produced.

For a merge, the skip operand is bowed out sideways so the bypass is visible
rather than hidden under the main chain. Ops are recognised on `jnp`, `jax`,
`nnx`, `lax`, and `numpy`, plus the arithmetic operators.

The view opens at depth 1 - top-level children only. Increase depth as
needed; big models stay readable because you choose how far to unfold.

## How it works, and what that implies

`jax.make_jaxpr` flattens a model to primitives, and NNX emits no
`name_stack`, so a jaxpr cannot say *which module* an op belongs to. Instead
`trace_model` temporarily wraps `__call__` on every `nnx.Module` class
reachable from the model, runs one eager forward pass, and records module
identity, nesting, real shapes, and dataflow. Wrappers are installed on the
class (NNX rejects stray instance attributes) and always removed in a
`finally`.

Consequences worth knowing:

- **Only executed modules appear.** A submodule skipped by a branch in
  `__call__` is absent from the graph - the picture shows what ran, for the
  inputs you gave.
- **Shapes are concrete**, because the pass is eager rather than jitted.
- **A failing model still produces a graph.** If the forward pass raises, the
  partial graph is returned with the error attached and the failing module
  marked in red - which is exactly when the picture is most useful.
- **Dataflow edges come from array identity** (`id()` of the arrays a module
  consumed vs. produced).

  *PyTorch:* bare ops are observed, not reconstructed. A `TorchFunctionMode`
  sees every tensor op as it runs, gives the ones worth drawing a node, and
  tags each result with the node that produced it; the next consumer - op or
  module - resolves its own inputs through the same producer map. An edge
  therefore exists because the *same tensor object* left one node and entered
  another, which is a fact about the run.

  This replaced an AST scan of `forward` that found bare ops in the source,
  anchored each to a nearby module call, and reconstructed its shapes from
  whatever neighbour was reachable. That is right only while a value flows
  straight from one call to the next. Where it does not, the errors were not
  local: a flow field resized inside a decoder loop is produced levels above,
  so "the nearest preceding call" was the trunk, and the op ended up carrying
  the trunk's shapes and - once spliced onto the host's inputs - the trunk's
  edges, one node with an arrow in from every pyramid level and an arrow out
  to nearly every module below it. Tagging cannot express that: the op sees
  the tensors it was actually handed.

  Four things the mode has to get right, each of which otherwise shows up as a
  wrong node: it is re-entrant (`F.interpolate` dispatches to more torch
  functions, so only the outermost call becomes a node); an op that *is* its
  enclosing module (`nn.LeakyReLU` running `leaky_relu`) must tag the module
  and not draw itself twice; weight math is not dataflow (`EqualLinear`
  scaling its kernel, `ModulatedConv2d` reshaping one - real ops whose result
  is a filter, recognised by a 4-D result that does not carry the batch axis);
  and metadata-only ops like `.expand` pass their producer through rather than
  severing the chain.

  An observed op that no drawn consumer takes is an interior step of a larger
  expression - `L2Norm` adds an epsilon, then rsqrts, then multiplies, and only
  ops on the drawn list become nodes - so it is spliced out rather than left
  with an arrow going nowhere.

  *JAX:* values that pass through pure `jnp` operations between modules are new
  arrays, which breaks the chain; those links are recovered from execution
  order and tagged `inferred`, and the source-scan machinery below still
  applies.
- **Op nodes come from the source.** Which array ops a container runs is
  decided by parsing its `__call__`, because array identity alone cannot tell
  a residual add from an ordinary activation - `ConvBlock` also returns a
  fresh array. Each op is then placed against the module calls around it: in
  the same statement (`nnx.leaky_relu(self.init(x))`), or after the nearest
  preceding call (`x = layer(x)` then `x = nnx.leaky_relu(x)`). Ops in a loop
  body attach to every child that loop invoked, and `for layer in self.layers`
  is resolved back to the container so those calls are matched. If the source
  is unavailable (a C-implemented or dynamically generated `__call__`), no op
  nodes are synthesized and those steps will not be drawn.
- **Ops are drawn where they are written, not where they execute.** The
  source scan gives position, not a second runtime trace, so an op guarded by
  a branch that did not run for your inputs may still be drawn.
- Tracing is not thread-safe: it patches classes for the duration of the call.

## Requirements

`jax` and `flax` only, both already required by `flax_blocks`.
`visualize_notebook` additionally needs `IPython`, imported lazily so the
rest of the package works without it.
