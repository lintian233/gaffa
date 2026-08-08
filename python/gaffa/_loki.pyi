"""Private pybind11 declarations for the optional Loki extension."""


from typing import Any

from .peaks import PeriodicPeak

class PffaPlan:
    nsamples: int
    tsamp: float



class _PffaProgram:
    def __init__(
        self,
        plan: PffaPlan,
        device_id: int,
        max_peaks_per_series: int,
    ) -> None: ...

    @property
    def device_id(self) -> int: ...

    @property
    def nsamples(self) -> int: ...

    def search(
        self,
        time_series: Any,
        *,
        max_peaks: int,
    ) -> list[PeriodicPeak]: ...

    def search_batch(
        self,
        data: Any,
        *,
        max_peaks: int,
    ) -> list[list[PeriodicPeak]]: ...


def _make_pffa_plan(
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
) -> PffaPlan: ...
