"""Optional Loki P-FFA search backend.

This module is available when Gaffa is built with ``GAFFA_ENABLE_LOKI=ON``.
It accepts prepared host ``float32`` NumPy arrays and keeps the CUDA execution
state reusable through :class:`PffaProgram`.
"""

from __future__ import annotations

from ..peaks import PeriodicPeak

try:
    from .._loki import PffaPlan as PffaPlan
    from .._loki import _make_pffa_plan, _PffaProgram
except ImportError as error:
    raise ImportError(
        "gaffa.loki requires a build with GAFFA_ENABLE_LOKI=ON and the Loki runtime available"
    ) from error


def _parse_device(device: str) -> int:
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("device must be a CUDA device such as 'cuda:0'")
    ordinal = device[5:]
    if not ordinal.isdigit():
        raise ValueError("device must be a CUDA device such as 'cuda:0'")
    return int(ordinal)


def _parse_peak_limit(max_peaks: int | None, default: int) -> int:
    if max_peaks is None:
        return default
    if isinstance(max_peaks, bool) or not isinstance(max_peaks, int) or max_peaks <= 0:
        raise ValueError("max_peaks must be positive or None")
    return max_peaks


def make_pffa_plan(
    *,
    nsamples: int,
    tsamp: float,
    frequency: tuple[float, float],
    acceleration: tuple[float, float] | None = None,
    jerk: tuple[float, float] | None = None,
    snap: tuple[float, float] | None = None,
    phase_bins_min: int = 256,
    phase_bins_max: int = 256,
    eta: float = 1.0,
    duty_cycle_max: float = 0.20,
    width_spacing: float = 1.5,
    snr_threshold: float = 6.0,
) -> PffaPlan:
    """Create a validated Loki P-FFA plan.

    ``frequency`` and motion ranges are inclusive ``(minimum, maximum)``
    pairs. Loki currently requires ``nsamples`` to be a power of two and does
    not support snap plans through the public region planner.
    """

    return _make_pffa_plan(
        nsamples=nsamples,
        tsamp=tsamp,
        frequency=frequency,
        acceleration=acceleration,
        jerk=jerk,
        snap=snap,
        phase_bins_min=phase_bins_min,
        phase_bins_max=phase_bins_max,
        eta=eta,
        duty_cycle_max=duty_cycle_max,
        width_spacing=width_spacing,
        snr_threshold=snr_threshold,
    )


class PffaProgram:
    """Reusable Loki P-FFA execution state for host NumPy inputs."""

    def __init__(
        self,
        plan: PffaPlan,
        *,
        device: str = "cuda:0",
        max_peaks_per_series: int = 1_000_000,
    ) -> None:
        device_id = _parse_device(device)
        if (
            isinstance(max_peaks_per_series, bool)
            or not isinstance(max_peaks_per_series, int)
            or max_peaks_per_series <= 0
        ):
            raise ValueError("max_peaks_per_series must be positive")
        self._max_peaks_per_series = max_peaks_per_series
        self._program = _PffaProgram(plan, device_id, max_peaks_per_series)

    @property
    def device(self) -> str:
        self._ensure_open()
        return f"cuda:{self._get_program().device_id}"

    @property
    def nsamples(self) -> int:
        self._ensure_open()
        return self._get_program().nsamples

    def search(
        self,
        time_series,
        *,
        max_peaks: int | None = None,
    ) -> list[PeriodicPeak]:
        """Search one prepared host ``float32`` time series."""

        self._ensure_open()
        return self._get_program().search(
            time_series,
            max_peaks=_parse_peak_limit(max_peaks, self._max_peaks_per_series),
        )

    def search_batch(
        self,
        data,
        *,
        max_peaks: int | None = None,
    ) -> list[list[PeriodicPeak]]:
        """Search a prepared host ``[nseries, nsamples]`` batch."""

        self._ensure_open()
        return self._get_program().search_batch(
            data,
            max_peaks=_parse_peak_limit(max_peaks, self._max_peaks_per_series),
        )

    def close(self) -> None:
        """Release the Loki Program and its reusable device allocations."""

        self._program = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._program is None:
            raise RuntimeError("PffaProgram is closed")

    def _get_program(self) -> _PffaProgram:
        if self._program is None:
            raise RuntimeError("PffaProgram is closed")
        return self._program


def search(
    time_series,
    plan: PffaPlan,
    *,
    device: str = "cuda:0",
    max_peaks: int | None = None,
) -> list[PeriodicPeak]:
    """Run one-shot Loki P-FFA search on a prepared host series."""

    return PffaProgram(plan, device=device).search(
        time_series,
        max_peaks=max_peaks,
    )


def search_batch(
    data,
    plan: PffaPlan,
    *,
    device: str = "cuda:0",
    max_peaks: int | None = None,
    max_peaks_per_series: int = 1_000_000,
) -> list[list[PeriodicPeak]]:
    """Run one-shot Loki P-FFA search on a prepared batch."""

    return PffaProgram(
        plan,
        device=device,
        max_peaks_per_series=max_peaks_per_series,
    ).search_batch(data, max_peaks=max_peaks)


__all__ = ["PffaPlan", "PffaProgram", "make_pffa_plan", "search", "search_batch"]
