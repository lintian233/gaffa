import numpy as np
import pytest

from gaffa import ffa


def make_plan(nsamples: int) -> ffa.FfaPlan:
    return ffa.make_riptide_plan(
        nsamples=nsamples,
        tsamp=1.0,
        period_min=2.0,
        period_max=4.0,
        bins_min=2,
        bins_max=2,
    )


def test_make_riptide_plan_exposes_stable_summary() -> None:
    plan = make_plan(8)

    assert isinstance(plan, ffa.FfaPlan)
    assert plan.task_count > 0
    assert plan.width_trials == (1,)


def test_ffa_search_returns_sorted_raw_peaks() -> None:
    series = np.array([0, 0, 0, 5, 0, 0, 0, 5], dtype=np.float32)

    peaks = ffa.ffa_search(series, make_plan(series.size), snr_threshold=0.0)

    assert peaks
    assert all(isinstance(peak, ffa.FfaPeak) for peak in peaks)
    assert all(peaks[index].snr >= peaks[index + 1].snr for index in range(len(peaks) - 1))
    assert peaks[0].period > 0.0
    assert peaks[0].frequency > 0.0
    assert peaks[0].width > 0
    assert peaks[0].bins == 2


def test_ffa_search_returns_empty_above_threshold() -> None:
    series = np.array([0, 0, 0, 5, 0, 0, 0, 5], dtype=np.float32)

    peaks = ffa.ffa_search(series, make_plan(series.size), snr_threshold=1000.0)

    assert peaks == []


def test_ffa_search_rejects_inputs_that_would_copy() -> None:
    plan = make_plan(8)

    with pytest.raises(TypeError, match="numpy.ndarray"):
        ffa.ffa_search([0.0] * 8, plan)
    with pytest.raises(TypeError, match="dtype float32"):
        ffa.ffa_search(np.zeros(8, dtype=np.float64), plan)
    with pytest.raises(ValueError, match="1D"):
        ffa.ffa_search(np.zeros((1, 8), dtype=np.float32), plan)
    with pytest.raises(ValueError, match="C-contiguous"):
        ffa.ffa_search(np.zeros(16, dtype=np.float32)[::2], plan)


def test_ffa_search_rejects_plan_length_mismatch_and_invalid_peak_limit() -> None:
    series = np.zeros(8, dtype=np.float32)

    with pytest.raises(ValueError, match="input_nsamples"):
        ffa.ffa_search(series, make_plan(9))
    with pytest.raises(ValueError, match="positive or None"):
        ffa.ffa_search(series, make_plan(series.size), max_peaks=0)
