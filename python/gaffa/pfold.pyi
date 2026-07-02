"""Pulsar folding helpers for dedispersed dynamic spectra."""

from __future__ import annotations

from ._core import DedispersedSpectrum, FoldResult


def fold(
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
        Optional number of frequency subbands. The current channel count must
        be exactly divisible by this value.

    Returns
    -------
    FoldResult
        Folded cube and common profile projections.
    """
    ...
