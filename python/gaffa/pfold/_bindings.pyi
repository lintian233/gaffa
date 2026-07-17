"""Type-only fragment for folding bindings.

This module is not importable at runtime. Python code must import public
folding objects from ``gaffa.pfold`` or ``gaffa._core``.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..dedispersion._bindings import DedispersedArray, DedispersedSpectrumArray

FoldCubeArray: TypeAlias = NDArray[np.float32]
FoldExposureArray: TypeAlias = NDArray[np.float64]


class FoldResult:
    """Folded products for a dedispersed dynamic spectrum."""

    cube: FoldCubeArray
    exposure: FoldExposureArray
    profile: NDArray[np.float32]
    freq_phase: NDArray[np.float32]
    time_phase: NDArray[np.float32]
    phase: NDArray[np.float32]
    time: NDArray[np.float64]
    nsubint: int
    nchans: int
    nbin: int
    period: float
    tsamp: float
    tsubint: float

    def __repr__(self) -> str: ...


class FoldedProfile:
    """Exposure-normalized folded profile for a dedispersed 1D time series."""

    profile: NDArray[np.float32]
    exposure: NDArray[np.float64]
    phase: NDArray[np.float32]
    nbin: int
    period: float
    tsamp: float

    def __repr__(self) -> str: ...


def _fold_dedispersed_profile(
    data: DedispersedArray,
    *,
    tsamp: float,
    period: float,
    nbin: int,
    dm_index: int = 0,
) -> FoldedProfile:
    """Private binding used by :func:`gaffa.pfold.fold_profile`."""
    ...


def _fold_dedispersed_spectrum(
    data: DedispersedSpectrumArray,
    *,
    tsamp: float,
    period: float,
    nbin: int,
    tsubint: float,
    output_channels: int,
) -> FoldResult:
    """Private binding used by :func:`gaffa.pfold.fold_spectrum`."""
    ...
