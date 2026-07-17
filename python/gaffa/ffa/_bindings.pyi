"""Type-only fragment for raw FFA bindings.

This module is not importable at runtime. Python code must import public FFA
objects from ``gaffa.ffa`` and must not depend on underscored bindings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FfaPlan:
    """Immutable reusable plan for one FFA period search."""

    task_count: int
    """Number of FFA transform tasks in this plan."""

    width_trials: tuple[int, ...]
    """Global boxcar widths used by raw FFA detection."""

    def __repr__(self) -> str: ...


class FfaPeak:
    """One raw FFA peak before cross-width or cross-DM candidate processing."""

    period: float
    frequency: float
    snr: float
    width: int
    duty_cycle: float
    phase: int
    shift: int
    bins: int
    width_index: int
    period_index: int

    def __repr__(self) -> str: ...


def _make_riptide_ffa_plan(
    *,
    nsamples: int,
    tsamp: float,
    period_min: float,
    period_max: float,
    bins_min: int = 180,
    bins_max: int = 256,
    min_periods: int = 1,
    duty_cycle_max: float = 0.20,
    width_trial_spacing: float = 1.5,
    max_tasks: int = 1_000_000,
) -> FfaPlan:
    """Private binding used by :func:`gaffa.ffa.make_riptide_plan`."""
    ...


def _ffa_search_cpu(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[FfaPeak]:
    """Private CPU binding used by :func:`gaffa.ffa.ffa_search`."""
    ...


def _ffa_search_cuda_host(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    device_id: int = 0,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[FfaPeak]:
    """Private host-input CUDA binding used by :func:`gaffa.ffa.ffa_search`."""
    ...
