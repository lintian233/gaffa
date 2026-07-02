from pathlib import Path

import numpy as np
import pytest

from gaffa._core import _fold_dedispersed_profile, _fold_dedispersed_spectrum
from gaffa.dedispersion import dedisperse_single_dm, dedisperse_spectrum
from gaffa.io import Filterbank
from gaffa.pfold import (
    FoldResult,
    FoldedProfile,
    fold,
    fold_profile,
    fold_spectrum,
)


DATA_DIR = Path(__file__).parent / "data"
BASETEST = DATA_DIR / "basetest.fil"


def test_fold_returns_pfold_products() -> None:
    spectrum = dedisperse_spectrum(Filterbank(BASETEST), dm=0.0, backend="cpu")

    result = fold_spectrum(
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


def test_fold_dispatches_spectrum_to_fold_result() -> None:
    spectrum = dedisperse_spectrum(Filterbank(BASETEST), dm=0.0, backend="cpu")

    result = fold(
        spectrum,
        period=spectrum.tsamp * 4,
        nbin=4,
        tsubint=spectrum.tsamp * 8,
    )

    assert isinstance(result, FoldResult)


def test_fold_profile_returns_integrated_profile() -> None:
    dedispersed = dedisperse_single_dm(Filterbank(BASETEST), dm=0.0, backend="cpu")

    result = fold_profile(
        dedispersed,
        period=dedispersed.tsamp * 4,
        nbin=4,
    )

    assert isinstance(result, FoldedProfile)
    assert result.profile.shape == (4,)
    assert result.exposure.shape == (4,)
    assert result.phase.shape == (4,)
    assert result.profile.dtype == np.float32
    assert result.exposure.dtype == np.float64
    assert result.nbin == 4


def test_fold_dispatches_dedispersed_result_to_profile() -> None:
    dedispersed = dedisperse_single_dm(Filterbank(BASETEST), dm=0.0, backend="cpu")

    result = fold(
        dedispersed,
        period=dedispersed.tsamp * 4,
        nbin=4,
    )

    assert isinstance(result, FoldedProfile)


def test_fold_can_downsample_frequency() -> None:
    spectrum = dedisperse_spectrum(Filterbank(BASETEST), dm=0.0, backend="cpu")
    nsubband = max(1, spectrum.nchans // 2)

    result = fold_spectrum(
        spectrum,
        period=spectrum.tsamp * 4,
        nbin=4,
        tsubint=spectrum.tsamp * 8,
        nsubband=nsubband,
    )

    assert result.nchans == nsubband
    assert result.freq_phase.shape == (nsubband, 4)


def test_fold_profile_uses_dm_index() -> None:
    data = np.array(
        [
            [1, 1, 1, 1],
            [2, 4, 6, 8],
        ],
        dtype=np.uint32,
    )

    result = _fold_dedispersed_profile(
        data,
        tsamp=0.25,
        period=1.0,
        nbin=2,
        dm_index=1,
    )

    np.testing.assert_allclose(
        result.profile, np.array([3.0, 7.0], dtype=np.float32)
    )


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


def test_fold_profile_rejects_non_contiguous_data() -> None:
    data = np.arange(16, dtype=np.uint32).reshape(2, 8)

    with pytest.raises(ValueError, match="C-contiguous"):
        _fold_dedispersed_profile(
            data[:, ::-1],
            tsamp=0.25,
            period=1.0,
            nbin=4,
        )


def test_fold_profile_rejects_invalid_dm_index() -> None:
    dedispersed = dedisperse_single_dm(Filterbank(BASETEST), dm=0.0, backend="cpu")

    with pytest.raises(ValueError, match="dm_index must be non-negative"):
        fold_profile(
            dedispersed,
            period=dedispersed.tsamp * 4,
            nbin=4,
            dm_index=-1,
        )
    with pytest.raises(IndexError, match="dm_index is out of range"):
        fold_profile(
            dedispersed,
            period=dedispersed.tsamp * 4,
            nbin=4,
            dm_index=dedispersed.ndm,
        )


def test_fold_rejects_invalid_nsubband() -> None:
    spectrum = dedisperse_spectrum(Filterbank(BASETEST), dm=0.0, backend="cpu")

    with pytest.raises(ValueError, match="nsubband must be positive"):
        fold_spectrum(
            spectrum,
            period=spectrum.tsamp * 4,
            nbin=4,
            nsubband=0,
        )
