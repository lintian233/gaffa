"""Fast-folding-algorithm search primitives for preprocessed time series."""

from __future__ import annotations

from .._core import (
    DmPeak,
    FfaPeak,
    FfaPlan,
    PeriodicPeak,
    _CudaProgram,
    _make_riptide_ffa_plan,
    _search_dms_cpu,
    _search_periodic_batch_cpu,
    _search_periodic_cpu,
    _search_raw_cpu,
    _search_raw_cuda_host,
)
from ..dedispersion import DedispersedResult
from ..preprocessing import PreprocessPlan


def _parse_device(device: str | None) -> int | None:
    if device is None:
        return None
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("device must be None or a CUDA device such as 'cuda:0'")
    ordinal = device[5:]
    if not ordinal.isdigit():
        raise ValueError("device must be a CUDA device such as 'cuda:0'")
    return int(ordinal)


def _parse_max_peaks(max_peaks: int | None) -> int:
    if max_peaks is None:
        return 0
    if not isinstance(max_peaks, int) or isinstance(max_peaks, bool) or max_peaks <= 0:
        raise ValueError("max_peaks must be positive or None")
    return max_peaks


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


def search_raw(
    time_series,
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
    device: str | None = None,
) -> list[FfaPeak]:
    """Search one prepared time series and return raw FFA peaks.

    ``device=None`` selects the CPU implementation. A value such as
    ``device="cuda:0"`` selects the native CUDA implementation. This is the
    low-level raw-detection API; use :func:`search` for canonical
    :class:`~gaffa.peaks.PeriodicPeak` results.
    """

    device_id = _parse_device(device)
    if max_peaks is not None:
        _parse_max_peaks(max_peaks)
    if device_id is None:
        return _search_raw_cpu(
            time_series,
            plan,
            snr_threshold=snr_threshold,
            max_peaks=max_peaks,
        )
    return _search_raw_cuda_host(
        time_series,
        plan,
        device_id=device_id,
        snr_threshold=snr_threshold,
        max_peaks=max_peaks,
    )


def search(
    time_series,
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
    device: str | None = None,
) -> list[PeriodicPeak]:
    """Search one prepared time series and return canonical periodic peaks.

    ``device=None`` uses the native CPU implementation. A value such as
    ``device="cuda:0"`` performs a one-shot native CUDA search. Repeated CUDA
    tile searches should reuse :class:`CudaProgram` instead.
    """

    device_id = _parse_device(device)
    peak_limit = _parse_max_peaks(max_peaks)
    if device_id is None:
        return _search_periodic_cpu(
            time_series,
            plan,
            snr_threshold=snr_threshold,
            max_peaks=peak_limit,
        )
    assert device is not None
    return CudaProgram(plan, device=device).search(
        time_series,
        snr_threshold=snr_threshold,
        max_peaks=max_peaks,
    )


def search_batch(
    data,
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
    device: str | None = None,
    series_tile_size: int = 16,
) -> list[list[PeriodicPeak]]:
    """Search a prepared ``[nseries, nsamples]`` batch.

    Each returned inner list corresponds to one input row. The native CPU
    batch convenience path is intentionally a thin row-wise wrapper; the
    optimized DM-search CPU path remains :func:`search_dms_cpu`.
    """

    device_id = _parse_device(device)
    peak_limit = _parse_max_peaks(max_peaks)
    if device_id is None:
        return _search_periodic_batch_cpu(
            data,
            plan,
            snr_threshold=snr_threshold,
            max_peaks=peak_limit,
        )
    assert device is not None
    return CudaProgram(
        plan,
        device=device,
        series_tile_size=series_tile_size,
    ).search_batch(
        data,
        snr_threshold=snr_threshold,
        max_peaks=max_peaks,
    )


class CudaProgram:
    """Reusable native CUDA FFA execution state for host NumPy inputs."""

    def __init__(
        self,
        plan: FfaPlan,
        *,
        device: str = "cuda:0",
        series_tile_size: int = 16,
    ) -> None:
        device_id = _parse_device(device)
        if device_id is None:
            raise ValueError("CudaProgram requires a CUDA device")
        if (
            isinstance(series_tile_size, bool)
            or not isinstance(series_tile_size, int)
            or series_tile_size <= 0
        ):
            raise ValueError("series_tile_size must be positive")
        self._program = _CudaProgram(plan, device_id, series_tile_size)

    @property
    def device(self) -> str:
        self._ensure_open()
        return f"cuda:{self._get_program().device_id}"

    @property
    def nsamples(self) -> int:
        self._ensure_open()
        return self._get_program().nsamples

    @property
    def series_tile_size(self) -> int:
        self._ensure_open()
        return self._get_program().tile_capacity

    def search(
        self,
        time_series,
        *,
        snr_threshold: float = 6.0,
        max_peaks: int | None = None,
    ) -> list[PeriodicPeak]:
        """Search one prepared host ``float32`` time series."""

        self._ensure_open()
        return self._get_program().search(
            time_series,
            snr_threshold=snr_threshold,
            max_peaks=_parse_max_peaks(max_peaks),
        )

    def search_batch(
        self,
        data,
        *,
        snr_threshold: float = 6.0,
        max_peaks: int | None = None,
    ) -> list[list[PeriodicPeak]]:
        """Search a prepared host ``[nseries, nsamples]`` batch."""

        self._ensure_open()
        return self._get_program().search_batch(
            data,
            snr_threshold=snr_threshold,
            max_peaks=_parse_max_peaks(max_peaks),
        )

    def close(self) -> None:
        """Release the CUDA Program and its reusable device allocations."""

        self._program = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._program is None:
            raise RuntimeError("CudaProgram is closed")

    def _get_program(self) -> _CudaProgram:
        if self._program is None:
            raise RuntimeError("CudaProgram is closed")
        return self._program


def search_dms_cpu(
    dedispersed: DedispersedResult,
    plan: FfaPlan,
    *,
    preprocess: PreprocessPlan,
    dm_index_offset: int = 0,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[DmPeak]:
    """Preprocess and search one host-resident block of DM time series."""

    if not isinstance(dedispersed, DedispersedResult):
        raise TypeError("dedispersed must be a DedispersedResult")
    if dm_index_offset < 0:
        raise ValueError("dm_index_offset must be non-negative")
    return _search_dms_cpu(
        dedispersed.data,
        tsamp=dedispersed.tsamp,
        dm_low=dedispersed.dm_low,
        dm_step=dedispersed.dm_step,
        dm_index_offset=dm_index_offset,
        plan=plan,
        preprocess=preprocess,
        snr_threshold=snr_threshold,
        max_peaks=max_peaks,
    )


__all__ = [
    "CudaProgram",
    "FfaPeak",
    "FfaPlan",
    "PeriodicPeak",
    "make_riptide_plan",
    "search",
    "search_batch",
    "search_dms_cpu",
    "search_raw",
]
