"""Command-line entry point: visualize a model without writing a script.

Example::

    python -m Blocks.visualized.cli Blocks.flax_blocks.core_blocks:ResidualBlock \\
        --args 8 16 --input 2,16,16,8 --out res.html --open

``--args``/``--kwargs`` construct the model (``rngs`` is supplied
automatically when the constructor accepts it); ``--input`` gives the shapes
of the example tensors fed to the forward pass.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys

import jax
from flax import nnx

from .visualize import summary, visualize


def _literal(text: str):
    """Best-effort scalar parse for CLI-supplied constructor arguments."""
    import ast
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _load(target: str):
    """Resolve ``package.module:Attribute`` to the object it names."""
    if ":" not in target:
        raise SystemExit(
            f"target must look like 'my.module:ModelClass' (got {target!r})")
    mod_name, attr = target.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise SystemExit(f"cannot import '{mod_name}': {exc}") from None
    try:
        return getattr(mod, attr)
    except AttributeError:
        raise SystemExit(f"'{mod_name}' has no attribute '{attr}'") from None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m Blocks.visualized.cli",
        description="Render an interactive HTML graph of a Flax NNX model.")
    ap.add_argument("target", help="module path, e.g. my.module:ModelClass")
    ap.add_argument("--args", nargs="*", default=[],
                    help="positional constructor arguments")
    ap.add_argument("--kwargs", nargs="*", default=[], metavar="K=V",
                    help="keyword constructor arguments")
    ap.add_argument("--input", nargs="*", default=[], metavar="D,D,D",
                    help="input shapes, e.g. 2,16,16,8 (one per tensor)")
    ap.add_argument("--out", default="model_graph.html", help="output file")
    ap.add_argument("--title", default=None, help="page title")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed")
    ap.add_argument("--open", action="store_true", dest="open_browser",
                    help="open the result in a browser")
    ap.add_argument("--text", action="store_true",
                    help="also print a text summary")
    ns = ap.parse_args(argv)

    obj = _load(ns.target)
    kw = {}
    for item in ns.kwargs:
        if "=" not in item:
            raise SystemExit(f"--kwargs entries need K=V form (got {item!r})")
        key, val = item.split("=", 1)
        kw[key] = _literal(val)

    if isinstance(obj, nnx.Module):
        model = obj                      # already-instantiated model
    else:
        pos = [_literal(a) for a in ns.args]
        # Inspect ``__init__`` rather than the class: the NNX metaclass
        # reports a bare ``(*args, **kwargs)`` signature.
        try:
            target = obj.__init__ if inspect.isclass(obj) else obj
            params = inspect.signature(target).parameters
        except (TypeError, ValueError):
            params = {}
        if "rngs" in params and "rngs" not in kw:
            kw["rngs"] = nnx.Rngs(ns.seed)
        try:
            model = obj(*pos, **kw)
        except TypeError as exc:
            raise SystemExit(
                f"could not construct {ns.target}: {exc}\n"
                f"pass constructor arguments with --args / --kwargs") from None

    if not ns.input:
        raise SystemExit("--input is required, e.g. --input 2,16,16,8")
    key = jax.random.key(ns.seed)
    inputs = []
    for i, spec in enumerate(ns.input):
        try:
            shape = tuple(int(d) for d in spec.replace("(", "")
                          .replace(")", "").split(",") if d.strip())
        except ValueError:
            raise SystemExit(f"bad --input shape {spec!r}") from None
        inputs.append(jax.random.normal(jax.random.fold_in(key, i), shape))

    if ns.text:
        print(summary(model, *inputs))

    out = visualize(model, *inputs, path=ns.out, title=ns.title,
                    open_browser=ns.open_browser)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
