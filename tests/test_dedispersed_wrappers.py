import numpy as np
import pytest

from gaffa.dedispersion import DedispersedResult, DedispersedSpectrum
from gaffa.pfold import FoldedProfile, FoldResult, fold_profile, fold_spectrum


def test_dedispersed_result_wraps_uint32_array_without_copy() -> None:
    data = np.array([[1, 2, 3, 4]], dtype=np.uint32)

    result = DedispersedResult(data, tsamp=0.25, dm_low=12.5)

    assert np.shares_memory(result.data, data)
    assert result.data is data
    assert result.shape == (1, 4)
    assert result.dtype == data.dtype
    assert result.nbytes == data.nbytes
    assert result.ndm == 1
    assert result.nsamples == 4
    assert result.tsamp == 0.25
    assert result.dm_low == 12.5
    assert result.dm_step == 0.0
    assert result.backend == "external"


def test_dedispersed_result_can_be_folded() -> None:
    data = np.array([[1, 2, 3, 4]], dtype=np.uint32)
    result = DedispersedResult(data, tsamp=0.25)

    folded = fold_profile(result, period=1.0, nbin=2)

    assert isinstance(folded, FoldedProfile)
    np.testing.assert_allclose(folded.profile, np.array([1.5, 3.5], dtype=np.float32))


def test_dedispersed_result_rejects_inputs_that_would_copy() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        DedispersedResult([[1, 2, 3, 4]], tsamp=0.25)

    data = np.arange(8, dtype=np.uint32).reshape(2, 4)
    with pytest.raises(ValueError, match="C-contiguous"):
        DedispersedResult(data[:, ::-1], tsamp=0.25)

    with pytest.raises(TypeError, match="uint32 or float32"):
        DedispersedResult(np.ones((1, 4), dtype=np.uint16), tsamp=0.25)


def test_dedispersed_spectrum_wraps_array_without_copy() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    spectrum = DedispersedSpectrum(
        data,
        tsamp=0.25,
        dm=42.0,
        chan_begin=10,
    )

    assert np.shares_memory(spectrum.data, data)
    assert spectrum.data is data
    assert spectrum.shape == (2, 2)
    assert spectrum.dtype == data.dtype
    assert spectrum.nbytes == data.nbytes
    assert spectrum.nsamples == 2
    assert spectrum.nchans == 2
    assert spectrum.tsamp == 0.25
    assert spectrum.dm == 42.0
    assert spectrum.chan_begin == 10
    assert spectrum.chan_end == 12
    assert spectrum.backend == "external"


def test_dedispersed_spectrum_can_be_folded() -> None:
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint8)
    spectrum = DedispersedSpectrum(data, tsamp=0.25)

    folded = fold_spectrum(spectrum, period=1.0, nbin=2, tsubint=1.0)

    assert isinstance(folded, FoldResult)
    assert folded.cube.shape == (1, 2, 2)
    np.testing.assert_allclose(folded.profile, np.array([2.5, 6.5], dtype=np.float32))


def test_dedispersed_spectrum_rejects_invalid_metadata_and_inputs() -> None:
    data = np.ones((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="chan_end - chan_begin"):
        DedispersedSpectrum(data, tsamp=0.25, chan_begin=0, chan_end=3)
    with pytest.raises(ValueError, match="dm must be finite and non-negative"):
        DedispersedSpectrum(data, tsamp=0.25, dm=-1.0)
    with pytest.raises(ValueError, match="C-contiguous"):
        DedispersedSpectrum(data[:, ::-1], tsamp=0.25)
    with pytest.raises(TypeError, match="uint8, uint16, or float32"):
        DedispersedSpectrum(np.ones((2, 2), dtype=np.uint32), tsamp=0.25)
