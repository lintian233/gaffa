"""Pulsar folding helpers for dedispersed time series and spectra."""

from __future__ import annotations

from typing import overload

from ._core import DedispersedResult, DedispersedSpectrum, FoldResult, FoldedProfile


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
        Host-resident dedispersed time series.
    period
        Folding period in seconds.
    nbin
        Number of phase bins.
    dm_index
        Row index in ``dedispersed.data``.

    Returns
    -------
    FoldedProfile
        Folded profile, exposure, and phase coordinate.
    """
    ...


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
        Host-resident dedispersed dynamic spectrum.
    period
        Folding period in seconds.
    nbin
        Number of phase bins.
    tsubint
        Subintegration length in seconds.
    nsubband
        Optional number of frequency subbands. The current channel count must
        be exactly divisible by this value.

    Returns
    -------
    FoldResult
        Folded cube and common profile projections.
    """
    ...


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
