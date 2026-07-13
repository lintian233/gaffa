"""Fast-folding-algorithm search primitives for preprocessed time series."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FfaPlan:
    """Immutable reusable plan for one FFA period search."""

    @property
    def task_count(self) -> int:
        """Number of transform tasks generated for the search."""
        ...

    @property
    def width_trials(self) -> tuple[int, ...]:
        """Global boxcar widths used during raw peak detection."""
        ...


class FfaPeak:
    """One raw FFA peak before cross-width/DM candidate processing."""

    period: float
    """Trial period in seconds."""

    frequency: float
    """Trial frequency in Hz."""

    snr: float
    """Raw boxcar signal-to-noise ratio."""

    width: int
    duty_cycle: float
    phase: int
    shift: int
    bins: int
    width_index: int
    period_index: int

    def __repr__(self) -> str: ...


def make_riptide_plan(
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
    """Build a reusable Riptide-compatible FFA plan.

    Parameters
    ----------
    nsamples
        Number of samples accepted by the plan.
    tsamp
        Sampling interval in seconds.
    period_min, period_max
        Requested period search bounds in seconds.
    bins_min, bins_max
        Inclusive profile-bin range.
    min_periods
        Minimum number of observed periods.
    duty_cycle_max
        Maximum global boxcar width fraction at ``bins_min``.
    width_trial_spacing
        Multiplicative boxcar-width spacing.
    max_tasks
        Maximum generated transform task count.
    """
    ...


def ffa_search(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[FfaPeak]:
    """Run CPU FFA peak detection on one preprocessed time series.

    Parameters
    ----------
    time_series
        Non-empty C-contiguous one-dimensional ``float32`` array. The caller
        must provide finite, baseline-corrected, approximately zero-mean,
        unit-variance samples. Input is read without copying.
    plan
        FFA plan whose sample count equals ``time_series.size``.
    snr_threshold
        Finite raw boxcar S/N threshold.
    max_peaks
        Optional positive raw-peak safety limit. A reached limit raises rather
        than truncating the result.

    Returns
    -------
    list[FfaPeak]
        Raw FFA peaks in deterministic descending order.
    """
    ...
