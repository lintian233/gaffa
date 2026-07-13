"""Fast-folding-algorithm search primitives for preprocessed time series."""

from ._core import FfaPeak, FfaPlan, _ffa_search_cpu, _make_riptide_ffa_plan


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
        Number of samples in every time series searched with this plan.
    tsamp
        Input sampling interval in seconds.
    period_min, period_max
        Requested period search bounds in seconds.
    bins_min, bins_max
        Inclusive range of folded-profile bin counts.
    min_periods
        Minimum number of folded periods required in the observation.
    duty_cycle_max
        Largest global boxcar trial fraction, generated from ``bins_min``.
    width_trial_spacing
        Multiplicative spacing between successive boxcar widths.
    max_tasks
        Safety limit on generated FFA transform tasks.

    Returns
    -------
    FfaPlan
        Immutable plan reusable for time series with exactly ``nsamples``
        samples.
    """

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
) -> list[FfaPeak]:
    """Search one preprocessed time series with a CPU FFA plan.

    Parameters
    ----------
    time_series
        Non-empty, one-dimensional, C-contiguous ``numpy.float32`` series.
        It must already be finite, baseline-corrected, and approximately
        zero-mean with unit sample variance. This function does not convert,
        detrend, normalize, or copy the input.
    plan
        Reusable plan returned by :func:`make_riptide_plan`. Its input length
        must equal ``time_series.size``.
    snr_threshold
        Finite raw boxcar S/N threshold.
    max_peaks
        Optional positive safety limit on raw peaks. ``None`` leaves raw peak
        collection unbounded. Reaching a limit raises instead of truncating
        scientific results.

    Returns
    -------
    list[FfaPeak]
        Significant raw FFA peaks in descending deterministic peak order.

    Notes
    -----
    This is the CPU implementation. CUDA search requires an explicit
    device-resident time-series API and is intentionally not selected here.
    """

    return _ffa_search_cpu(
        time_series,
        plan,
        snr_threshold=snr_threshold,
        max_peaks=max_peaks,
    )


__all__ = ["FfaPeak", "FfaPlan", "ffa_search", "make_riptide_plan"]
