"""Fast-folding-algorithm search primitives for preprocessed time series."""

import numpy as np
from numpy.typing import NDArray

from ..dedispersion import DedispersedResult
from ..peaks import DmPeak, PeriodicPeak
from ..preprocessing import PreprocessPlan
from ._bindings import FfaPeak as FfaPeak
from ._bindings import FfaPlan as FfaPlan


class CudaProgram:
    """Reusable native CUDA FFA execution state for host NumPy inputs."""

    def __init__(
        self,
        plan: FfaPlan,
        *,
        device: str = "cuda:0",
        series_tile_size: int = 16,
    ) -> None: ...

    @property
    def device(self) -> str: ...

    @property
    def nsamples(self) -> int: ...

    @property
    def series_tile_size(self) -> int: ...

    def search(
        self,
        time_series: NDArray[np.float32],
        *,
        snr_threshold: float = 6.0,
        max_peaks: int | None = None,
    ) -> list[PeriodicPeak]: ...

    def search_batch(
        self,
        data: NDArray[np.float32],
        *,
        snr_threshold: float = 6.0,
        max_peaks: int | None = None,
    ) -> list[list[PeriodicPeak]]: ...

    def close(self) -> None: ...

    def __enter__(self) -> CudaProgram: ...

    def __exit__(self, exc_type: object, exc_value: object,
                 traceback: object) -> None: ...


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


def search_raw(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
    device: str | None = None,
) -> list[FfaPeak]:
    """Run raw FFA peak detection on one prepared time series.

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
    device
        ``None`` selects CPU. A value such as ``"cuda:0"`` selects CUDA.
        CUDA uploads the host input once and returns only compact raw peak
        records.

    Returns
    -------
    list[FfaPeak]
        Raw FFA peaks in deterministic descending order.
    """
    ...


def search(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
    device: str | None = None,
) -> list[PeriodicPeak]:
    """Search one prepared series with CPU or native CUDA."""
    ...


def search_batch(
    data: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
    device: str | None = None,
    series_tile_size: int = 16,
) -> list[list[PeriodicPeak]]:
    """Search a prepared ``[nseries, nsamples]`` batch."""
    ...


def search_dms_cpu(
    dedispersed: DedispersedResult,
    plan: FfaPlan,
    *,
    preprocess: PreprocessPlan,
    dm_index_offset: int = 0,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[DmPeak]:
    """Preprocess and search one host-resident block of DM time series.

    Parameters
    ----------
    dedispersed
        Host result with shape ``(ndm, nsamples)`` and dtype ``uint32`` or
        ``float32``. Its data is read without copying the complete block.
    plan
        Native FFA plan matching ``dedispersed.nsamples`` and ``tsamp``.
    preprocess
        Plan applied independently to each DM row before FFA search.
    dm_index_offset
        Global DM-trial index assigned to the first row in this block.
    snr_threshold
        Finite raw boxcar signal-to-noise threshold.
    max_peaks
        Optional positive per-DM raw-peak safety limit.

    Returns
    -------
    list[DmPeak]
        Backend-neutral peaks carrying physical DM and global DM index.

    Notes
    -----
    The function searches only the supplied block. It uses the optimized C++
    OpenMP DM loop and does not perform cross-block candidate clustering.
    """
    ...


__all__ = [
    "CudaProgram",
    "FfaPeak",
    "FfaPlan",
    "PeriodicPeak",
    "make_riptide_plan",
    "search",
    "search_batch",
    "search_raw",
    "search_dms_cpu",
]
