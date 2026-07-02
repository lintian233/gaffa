"""Pulsar folding helpers for dedispersed time series and spectra."""

from __future__ import annotations

from typing import overload

from ._core import (
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
    """Fold a dedispersed 1D time series into an integrated profile.

    Parameters
    ----------
    dedispersed
        Host-resident dedispersed time series returned by
        ``gaffa.dedispersion.dedisperse_single_dm`` or another single/multi-DM
        dedispersion function.
    period
        Folding period in seconds.
    nbin
        Number of phase bins.
    dm_index
        Row index in ``dedispersed.data``. Single-DM results use the default 0.

    Returns
    -------
    FoldedProfile
        Integrated profile, exposure, and phase coordinate.
    """

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
    """Fold a single-DM dedispersed dynamic spectrum.

    Parameters
    ----------
    spectrum
        Host-resident dedispersed dynamic spectrum returned by
        ``gaffa.dedispersion.dedisperse_spectrum``.
    period
        Folding period in seconds.
    nbin
        Number of phase bins.
    tsubint
        Subintegration length in seconds.
    nsubband
        Optional number of frequency subbands. When provided, the current
        channel count must be exactly divisible by this value.

    Returns
    -------
    FoldResult
        Folded products. ``cube`` has shape ``(nsubint, nchans, nbin)``;
        ``profile`` has shape ``(nbin,)``; ``freq_phase`` has shape
        ``(nchans, nbin)``; and ``time_phase`` has shape ``(nsubint, nbin)``.
    """

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
    """Fold a dedispersed time series or dynamic spectrum.

    This is a convenience dispatcher. Prefer ``fold_profile`` for 1D
    dedispersed time series and ``fold_spectrum`` for 2D dedispersed spectra
    when writing stable analysis code.
    """

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
