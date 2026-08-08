"""Optional Loki P-FFA Python API."""

from typing import Self

import numpy as np
from numpy.typing import NDArray

from ..peaks import PeriodicPeak

class PffaPlan:
    nsamples: int
    tsamp: float



class PffaProgram:
    def __init__(
        self,
        plan: PffaPlan,
        *,
        device: str = "cuda:0",
        max_peaks_per_series: int = 1_000_000,
    ) -> None: ...

    @property
    def device(self) -> str: ...

    @property
    def nsamples(self) -> int: ...

    def search(
        self,
        time_series: NDArray[np.float32],
        *,
        max_peaks: int | None = None,
    ) -> list[PeriodicPeak]: ...

    def search_batch(
        self,
        data: NDArray[np.float32],
        *,
        max_peaks: int | None = None,
    ) -> list[list[PeriodicPeak]]: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_value: object,
                 traceback: object) -> None: ...


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
) -> PffaPlan: ...


def search(
    time_series: NDArray[np.float32],
    plan: PffaPlan,
    *,
    device: str = "cuda:0",
    max_peaks: int | None = None,
) -> list[PeriodicPeak]: ...


def search_batch(
    data: NDArray[np.float32],
    plan: PffaPlan,
    *,
    device: str = "cuda:0",
    max_peaks: int | None = None,
    max_peaks_per_series: int = 1_000_000,
) -> list[list[PeriodicPeak]]: ...


__all__ = ["PffaPlan", "PffaProgram", "make_pffa_plan", "search", "search_batch"]
