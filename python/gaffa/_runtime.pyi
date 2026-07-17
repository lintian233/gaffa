"""Type-only fragment for small runtime bindings.

This module exists only in the stub graph consumed by :mod:`gaffa._core`.
Runtime Python code must import these symbols from ``gaffa._core`` instead.
"""

from typing import Sequence


def vector_add(lhs: Sequence[float], rhs: Sequence[float]) -> list[float]: ...


def cuda_device_count() -> int: ...


def cuda_runtime_version() -> int: ...
