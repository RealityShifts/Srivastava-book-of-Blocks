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
  consumed vs. produced). Values that pass through pure `jnp` operations
  between modules are new arrays, so an edge may be attributed to the nearest
  enclosing module rather than the exact tensor.
- Tracing is not thread-safe: it patches classes for the duration of the call.

## Requirements

`jax` and `flax` only, both already required by `flax_blocks`.
`visualize_notebook` additionally needs `IPython`, imported lazily so the
rest of the package works without it.
