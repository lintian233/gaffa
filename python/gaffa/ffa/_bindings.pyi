"""Type-only fragment for raw FFA bindings.

This module is not importable at runtime. Python code must import public FFA
objects from ``gaffa.ffa`` and must not depend on underscored bindings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..peaks._bindings import DmPeak, PeriodicPeak
from ..preprocessing._bindings import PreprocessPlan


class FfaPlan:
    """Immutable reusable plan for one FFA period search."""

    nsamples: int
    """Number of input samples accepted by the plan."""

    tsamp: float
    """Sampling interval in seconds."""

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


class _CudaProgram:
    def __init__(
        self,
        plan: FfaPlan,
        device_id: int,
        series_tile_size: int = 16,
    ) -> None: ...

    @property
    def device_id(self) -> int: ...

    @property
    def tile_capacity(self) -> int: ...

    @property
    def nsamples(self) -> int: ...

    def search(
        self,
        data: NDArray[np.float32],
        *,
        snr_threshold: float = 6.0,
        max_peaks: int = 0,
    ) -> list[PeriodicPeak]: ...

    def search_batch(
        self,
        data: NDArray[np.float32],
        *,
        snr_threshold: float = 6.0,
        max_peaks: int = 0,
    ) -> list[list[PeriodicPeak]]: ...


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


def _search_raw_cpu(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[FfaPeak]:
    """Private CPU binding used by :func:`gaffa.ffa.search_raw`."""
    ...


def _search_raw_cuda_host(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    device_id: int = 0,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[FfaPeak]:
    """Private CUDA binding used by :func:`gaffa.ffa.search_raw`."""
    ...


def _search_periodic_cpu(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int = 0,
) -> list[PeriodicPeak]: ...


def _search_periodic_batch_cpu(
    data: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int = 0,
) -> list[list[PeriodicPeak]]: ...


def _search_dms_cpu(
    data: NDArray[np.uint32] | NDArray[np.float32],
    *,
    tsamp: float,
    dm_low: float,
    dm_step: float,
    dm_index_offset: int,
    plan: FfaPlan,
    preprocess: PreprocessPlan,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[DmPeak]: ...
