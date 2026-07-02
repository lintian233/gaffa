"""Pulsar folding helpers for dedispersed dynamic spectra."""

from __future__ import annotations

from ._core import DedispersedSpectrum, FoldResult, _fold_dedispersed_spectrum


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


__all__ = ["FoldResult", "fold"]
