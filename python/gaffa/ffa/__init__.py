"""Fast-folding-algorithm search primitives for preprocessed time series."""

from .._core import (
    FfaPeak,
    FfaPlan,
    _ffa_search_cpu,
    _ffa_search_cuda_host,
    _make_riptide_ffa_plan,
)


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
    """Build a reusable Riptide-compatible FFA plan."""

    return _make_riptide_ffa_plan(
        nsamples=nsamples,
        tsamp=tsamp,
        period_min=period_min,
        period_max=period_max,
        bins_min=bins_min,
        bins_max=bins_max,
        min_periods=min_periods,
        duty_cycle_max=duty_cycle_max,
        width_trial_spacing=width_trial_spacing,
        max_tasks=max_tasks,
    )


def ffa_search(
    time_series,
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
    backend: str = "cpu",
    device_id: int = 0,
) -> list[FfaPeak]:
    """Search one preprocessed time series with an FFA plan."""

    if backend == "cpu":
        if device_id != 0:
            raise ValueError("device_id is only valid with backend='cuda'")
        return _ffa_search_cpu(
            time_series,
            plan,
            snr_threshold=snr_threshold,
            max_peaks=max_peaks,
        )
    if backend == "cuda":
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        return _ffa_search_cuda_host(
            time_series,
            plan,
            device_id=device_id,
            snr_threshold=snr_threshold,
            max_peaks=max_peaks,
        )
    raise ValueError("backend must be 'cpu' or 'cuda'")


__all__ = ["FfaPeak", "FfaPlan", "ffa_search", "make_riptide_plan"]
