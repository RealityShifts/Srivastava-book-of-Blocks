"""Optional runtime type checking via jaxtyping + beartype.

Set ``CHECK_TYPES=0`` in the environment to disable all runtime checks and
fall back to a pure no-op decorator (zero overhead, annotations remain as
static documentation).

Usage::

    from ._typecheck import typecheck

    @typecheck
    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        ...
"""

from __future__ import annotations

import os
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

ENABLE_RUNTIME_CHECKS = os.getenv("CHECK_TYPES", "1") == "1"

if ENABLE_RUNTIME_CHECKS:
    from beartype import beartype
    from jaxtyping import jaxtyped

    typecheck: Callable[[F], F] = jaxtyped(typechecker=beartype)
else:
    def typecheck(fn: F) -> F:
        return fn


__all__ = ["typecheck", "ENABLE_RUNTIME_CHECKS"]
