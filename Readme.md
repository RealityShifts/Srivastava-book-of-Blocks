# Srivastava Book of Blocks

A catalog of reusable neural-network building blocks (PyTorch and Flax), plus Mermaid architecture diagrams for each public block.

## Layout

| Path | Repo | Contents |
|------|------|----------|
| [`models/blocks/`](models/blocks/) | [Srivastava-book-of-Blocks](https://github.com/RealityShifts/Srivastava-book-of-Blocks) | `pytorch_blocks/`, `flax_blocks/`, smoke tests |
| [`models/diagram/`](models/diagram/) | [Srivastava-book-of-Blocks-diagrams](https://github.com/RealityShifts/Srivastava-book-of-Blocks-diagrams) | Per-block `.md` diagrams, `_generate.py`, spec DSL in `blocks/` |

This checkout also keeps `pytorch_blocks/` and `flax_blocks/` at the repo root (same layout as the blocks submodule) for direct use without entering `models/blocks/`.

## Clone

```bash
git clone --recurse-submodules https://github.com/xlibraries/Srivastava-book-of-Blocks.git
cd Srivastava-book-of-Blocks
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Dependencies

```bash
pip install -r requirements.txt
```

(`torch` for PyTorch blocks; `jax` / `flax` for Flax blocks.)

## Quick check

Smoke tests use package imports (`from . import …`), so run them as modules—not as bare script paths.

From the blocks submodule:

```bash
cd models/blocks
python3 -m pytorch_blocks._smoke_test
python3 -m flax_blocks._smoke_test
```

From the repo root (in-tree copy under `pytorch_blocks/` / `flax_blocks/`):

```bash
python3 -m pytorch_blocks._smoke_test
python3 -m flax_blocks._smoke_test
```

## Diagrams

122 Mermaid `flowchart TD` diagrams across 17 categories live in the **diagrams** submodule (`models/diagram/`).

Regenerate all diagram markdown from specs:

```bash
cd models/diagram
python3 _generate.py
```

Browse generated files under `models/diagram/<category>/` (e.g. `core/DilatedConv2d.md`). See [`models/diagram/README.md`](models/diagram/README.md) for rendering in GitHub, draw.io, Mermaid Live, etc.

## Use as a submodule in another project

**Blocks only:**

```bash
git submodule add https://github.com/RealityShifts/Srivastava-book-of-Blocks.git models/blocks
git submodule update --init --recursive
```

**Diagrams only:**

```bash
git submodule add https://github.com/RealityShifts/Srivastava-book-of-Blocks-diagrams.git models/diagram
git submodule update --init --recursive
```

**Both** (as in this repo): add each submodule, or clone this repository with `--recurse-submodules`.
