from pathlib import Path

import numpy as np
import pytest

import gaffa
from gaffa.dedispersion import (
    DedispersedResult,
    dedisperse_multi_dm,
    dedisperse_single_dm,
    dedisperse_subband,
)
from gaffa.io import Filterbank


DATA_DIR = Path(__file__).parent / "data"
BASETEST = DATA_DIR / "basetest.fil"


def test_dedisperse_single_dm_returns_numpy_result() -> None:
    fb = Filterbank(BASETEST)

    result = dedisperse_single_dm(fb, dm=0.0, backend="cpu")

    assert isinstance(result, DedispersedResult)
    assert result.backend == "cpu"
    assert result.ndm == 1
    assert result.nsamples == fb.header.nsamples
    assert result.shape == (1, fb.header.nsamples)
    assert result.data.shape == result.shape
    assert result.data.dtype == np.uint32
    assert result.nbytes == result.data.nbytes


def test_dedisperse_multi_dm_matches_single_dm_slices() -> None:
    fb = Filterbank(BASETEST)

    multi = dedisperse_multi_dm(
        fb,
        dm_low=0.0,
        dm_step=1.0,
        ndm=3,
        backend="cpu",
    )

    assert multi.shape == (3, fb.header.nsamples)
    for dm_index in range(multi.ndm):
        single = dedisperse_single_dm(fb, dm=float(dm_index), backend="cpu")
        np.testing.assert_array_equal(multi.data[dm_index], single.data[0])


def test_dedisperse_subband_degenerate_matches_multi_dm() -> None:
    fb = Filterbank(BASETEST)

    multi = dedisperse_multi_dm(
        fb,
        dm_low=0.0,
        dm_step=1.0,
        ndm=4,
        backend="cpu",
    )
    subband = dedisperse_subband(
        fb,
        dm_low=0.0,
        dm_step=1.0,
        ndm=4,
        backend="cpu",
        subband_channels=1,
        ndm_per_nominal=1,
    )

    np.testing.assert_array_equal(subband.data, multi.data)


def test_dedispersion_rejects_unknown_backend() -> None:
    fb = Filterbank(BASETEST)

    with pytest.raises(ValueError, match="unknown dedispersion backend"):
        dedisperse_single_dm(fb, dm=0.0, backend="bad")


def test_dedispersed_data_lifetime_survives_wrapper_scope() -> None:
    def load_data() -> np.ndarray:
        fb = Filterbank(BASETEST)
        return dedisperse_single_dm(fb, dm=0.0, backend="cpu").data

    data = load_data()

    assert data.shape[0] == 1
    assert data[0, 0] >= 0


@pytest.mark.skipif(gaffa.cuda_device_count() == 0, reason="CUDA device is not visible")
def test_dedisperse_single_dm_cuda_returns_host_numpy_result() -> None:
    fb = Filterbank(BASETEST)

    result = dedisperse_single_dm(fb, dm=0.0, backend="cuda")

    assert result.backend == "cuda"
    assert result.shape == (1, fb.header.nsamples)
    assert result.data.dtype == np.uint32
