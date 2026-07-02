from pathlib import Path

import numpy as np
import pytest

from gaffa.dedispersion import dedisperse_spectrum
from gaffa.io import Filterbank
from gaffa._core import _fold_dedispersed_spectrum
from gaffa.pfold import FoldResult, fold


DATA_DIR = Path(__file__).parent / "data"
BASETEST = DATA_DIR / "basetest.fil"


def test_fold_returns_pfold_products() -> None:
    spectrum = dedisperse_spectrum(Filterbank(BASETEST), dm=0.0, backend="cpu")

    result = fold(
        spectrum,
        period=spectrum.tsamp * 4,
        nbin=4,
        tsubint=spectrum.tsamp * 8,
    )

    assert isinstance(result, FoldResult)
    assert result.cube.shape == (result.nsubint, result.nchans, result.nbin)
    assert result.exposure.shape == (result.nsubint, result.nbin)
    assert result.profile.shape == (result.nbin,)
    assert result.freq_phase.shape == (result.nchans, result.nbin)
    assert result.time_phase.shape == (result.nsubint, result.nbin)
    assert result.phase.shape == (result.nbin,)
    assert result.time.shape == (result.nsubint,)
    assert result.nchans == spectrum.nchans
    assert result.nbin == 4
    assert result.cube.dtype == np.float32


def test_fold_can_downsample_frequency() -> None:
    spectrum = dedisperse_spectrum(Filterbank(BASETEST), dm=0.0, backend="cpu")
    nsubband = max(1, spectrum.nchans // 2)

    result = fold(
        spectrum,
        period=spectrum.tsamp * 4,
        nbin=4,
        tsubint=spectrum.tsamp * 8,
        nsubband=nsubband,
    )

    assert result.nchans == nsubband
    assert result.freq_phase.shape == (nsubband, 4)


def test_fold_rejects_non_contiguous_data() -> None:
    spectrum = dedisperse_spectrum(Filterbank(BASETEST), dm=0.0, backend="cpu")

    with pytest.raises(ValueError, match="C-contiguous"):
        _fold_dedispersed_spectrum(
            spectrum.data[:, ::-1],
            tsamp=spectrum.tsamp,
            period=spectrum.tsamp * 4,
            nbin=4,
            tsubint=spectrum.tsamp * 8,
            output_channels=0,
        )


def test_fold_rejects_invalid_nsubband() -> None:
    spectrum = dedisperse_spectrum(Filterbank(BASETEST), dm=0.0, backend="cpu")

    with pytest.raises(ValueError, match="nsubband must be positive"):
        fold(
            spectrum,
            period=spectrum.tsamp * 4,
            nbin=4,
            nsubband=0,
        )
