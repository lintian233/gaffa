"""Fast-folding-algorithm search primitives for preprocessed time series."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._bindings import FfaPeak as FfaPeak, FfaPlan as FfaPlan


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
        Requested period-search bounds in seconds.
    bins_min, bins_max
        Inclusive folded-profile bin range.
    min_periods
        Minimum number of observed periods.
    duty_cycle_max
        Maximum global boxcar-width fraction at ``bins_min``.
    width_trial_spacing
        Multiplicative boxcar-width spacing.
    max_tasks
        Maximum generated transform task count.

    Returns
    -------
    FfaPlan
        Immutable plan reusable for series with exactly ``nsamples`` samples.
    """
    ...


def ffa_search(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
    backend: Literal["cpu", "cuda"] = "cpu",
    device_id: int = 0,
) -> list[FfaPeak]:
    """Run raw FFA peak detection on one preprocessed time series.

    Parameters
    ----------
    time_series
        Non-empty C-contiguous one-dimensional ``float32`` array. Samples must
        be finite, baseline-corrected, approximately zero-mean, and unit
        variance. The input is read without copying.
    plan
        FFA plan whose accepted sample count equals ``time_series.size``.
    snr_threshold
        Finite raw boxcar signal-to-noise threshold.
    max_peaks
        Optional positive raw-peak safety limit. Reaching the limit raises
        rather than silently truncating scientific output.
    backend
        ``"cpu"`` or ``"cuda"``. CUDA uploads the host input once and returns
        only compact raw peak records.
    device_id
        CUDA device ordinal. It must remain zero for the CPU backend.

    Returns
    -------
    list[FfaPeak]
        Raw FFA peaks in deterministic descending order.
    """
    ...


__all__ = ["FfaPeak", "FfaPlan", "ffa_search", "make_riptide_plan"]
