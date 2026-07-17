"""Pulsar folding helpers for dedispersed time series and spectra."""

from __future__ import annotations

from typing import overload

from .._core import (
    DedispersedResult,
    DedispersedSpectrum,
    FoldResult,
    FoldedProfile,
    _fold_dedispersed_profile,
    _fold_dedispersed_spectrum,
)


def fold_profile(
    dedispersed: DedispersedResult,
    *,
    period: float,
    nbin: int,
    dm_index: int = 0,
) -> FoldedProfile:
    """Fold a dedispersed 1D time series into an integrated profile."""

    if dm_index < 0:
        raise ValueError("dm_index must be non-negative")
    return _fold_dedispersed_profile(
        dedispersed.data,
        tsamp=dedispersed.tsamp,
        period=period,
        nbin=nbin,
        dm_index=dm_index,
    )


def fold_spectrum(
    spectrum: DedispersedSpectrum,
    *,
    period: float,
    nbin: int,
    tsubint: float = 10.0,
    nsubband: int | None = None,
) -> FoldResult:
    """Fold a single-DM dedispersed dynamic spectrum."""

    if nsubband is not None and nsubband <= 0:
        raise ValueError("nsubband must be positive")
    output_channels = 0 if nsubband is None else nsubband
    return _fold_dedispersed_spectrum(
        spectrum.data,
        tsamp=spectrum.tsamp,
        period=period,
        nbin=nbin,
        tsubint=tsubint,
        output_channels=output_channels,
    )


@overload
def fold(
    data: DedispersedResult,
    *,
    period: float,
    nbin: int,
    dm_index: int = 0,
) -> FoldedProfile: ...


@overload
def fold(
    data: DedispersedSpectrum,
    *,
    period: float,
    nbin: int,
    tsubint: float = 10.0,
    nsubband: int | None = None,
) -> FoldResult: ...


def fold(
    data: DedispersedResult | DedispersedSpectrum,
    *,
    period: float,
    nbin: int,
    dm_index: int = 0,
    tsubint: float = 10.0,
    nsubband: int | None = None,
) -> FoldedProfile | FoldResult:
    """Fold a dedispersed time series or dynamic spectrum."""

    if isinstance(data, DedispersedResult):
        return fold_profile(data, period=period, nbin=nbin, dm_index=dm_index)
    if isinstance(data, DedispersedSpectrum):
        return fold_spectrum(
            data,
            period=period,
            nbin=nbin,
            tsubint=tsubint,
            nsubband=nsubband,
        )
    raise TypeError("fold expects DedispersedResult or DedispersedSpectrum")


__all__ = [
    "FoldResult",
    "FoldedProfile",
    "fold",
    "fold_profile",
    "fold_spectrum",
]
