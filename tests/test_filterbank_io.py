from pathlib import Path

import numpy as np
import pytest

from gaffa.io import Filterbank


DATA_DIR = Path(__file__).parent / "data"
BASETEST = DATA_DIR / "basetest.fil"


def test_filterbank_returns_header_and_numpy_data() -> None:
    fb = Filterbank(BASETEST)

    assert isinstance(fb, Filterbank)
    assert fb.header.nsamples > 0
    assert fb.header.nifs > 0
    assert fb.header.nchans > 0
    assert fb.data.shape == (fb.header.nsamples, fb.header.nifs, fb.header.nchans)


def test_filterbank_dtype_matches_nbits() -> None:
    fb = Filterbank(BASETEST)

    assert fb.header.nbits == 8
    assert fb.data.dtype == np.uint8


def test_filterbank_convenience_properties_follow_data() -> None:
    fb = Filterbank(BASETEST)

    assert fb.shape == fb.data.shape
    assert fb.dtype == fb.data.dtype
    assert fb.nbytes == fb.data.nbytes


def test_filterbank_preserve_file_order() -> None:
    fb = Filterbank(BASETEST, channel_order="preserve_file_order")

    assert fb.header.foff < 0


def test_filterbank_rejects_unknown_channel_order() -> None:
    with pytest.raises(ValueError, match="unknown channel_order"):
        Filterbank(BASETEST, channel_order="bad")


def test_filterbank_rejects_unknown_reverse_backend() -> None:
    with pytest.raises(ValueError, match="unknown reverse_backend"):
        Filterbank(BASETEST, reverse_backend="bad")


def test_filterbank_data_lifetime_survives_wrapper_scope() -> None:
    def load_data() -> np.ndarray:
        return Filterbank(BASETEST).data

    data = load_data()

    assert data.shape[0] > 0
    assert data[0, 0, 0] >= 0
