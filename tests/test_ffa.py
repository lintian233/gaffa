import numpy as np
import pytest

import gaffa
from gaffa import dedispersion, ffa, peaks, preprocessing


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


def test_search_raw_returns_sorted_raw_peaks() -> None:
    series = np.array([0, 0, 0, 5, 0, 0, 0, 5], dtype=np.float32)

    peaks = ffa.search_raw(series, make_plan(series.size), snr_threshold=0.0)

    assert peaks
    assert all(isinstance(peak, ffa.FfaPeak) for peak in peaks)
    assert all(peaks[index].snr >= peaks[index + 1].snr for index in range(len(peaks) - 1))
    assert peaks[0].period > 0.0
    assert peaks[0].frequency > 0.0
    assert peaks[0].width > 0
    assert peaks[0].bins == 2


def test_search_raw_returns_empty_above_threshold() -> None:
    series = np.array([0, 0, 0, 5, 0, 0, 0, 5], dtype=np.float32)

    peaks = ffa.search_raw(series, make_plan(series.size), snr_threshold=1000.0)

    assert peaks == []


def test_search_returns_canonical_periodic_peaks() -> None:
    series = np.array([0, 0, 0, 5, 0, 0, 0, 5], dtype=np.float32)

    result = ffa.search(series, make_plan(series.size), snr_threshold=0.0)

    assert result
    assert all(isinstance(peak, peaks.PeriodicPeak) for peak in result)
    assert result[0].motion.reference_time_seconds == 4.0
    assert result[0].motion.frequency_hz > 0.0


def test_search_batch_preserves_series_boundaries() -> None:
    series = np.array([0, 0, 0, 5, 0, 0, 0, 5], dtype=np.float32)
    batch = np.stack([series, series])

    result = ffa.search_batch(batch, make_plan(series.size), snr_threshold=0.0)

    assert len(result) == 2
    assert all(result_row for result_row in result)
    assert all(
        isinstance(peak, peaks.PeriodicPeak)
        for result_row in result
        for peak in result_row
    )


def test_search_raw_rejects_inputs_that_would_copy() -> None:
    plan = make_plan(8)

    with pytest.raises(TypeError, match="numpy.ndarray"):
        ffa.search_raw([0.0] * 8, plan)
    with pytest.raises(TypeError, match="dtype float32"):
        ffa.search_raw(np.zeros(8, dtype=np.float64), plan)
    with pytest.raises(ValueError, match="1D"):
        ffa.search_raw(np.zeros((1, 8), dtype=np.float32), plan)
    with pytest.raises(ValueError, match="C-contiguous"):
        ffa.search_raw(np.zeros(16, dtype=np.float32)[::2], plan)


def test_search_raw_rejects_plan_length_mismatch_and_invalid_peak_limit() -> None:
    series = np.zeros(8, dtype=np.float32)

    with pytest.raises(ValueError, match="input_nsamples"):
        ffa.search_raw(series, make_plan(9))
    with pytest.raises(ValueError, match="positive or None"):
        ffa.search_raw(series, make_plan(series.size), max_peaks=0)


def test_search_raw_rejects_unknown_device_and_old_arguments() -> None:
    series = np.zeros(8, dtype=np.float32)
    plan = make_plan(series.size)

    with pytest.raises(ValueError, match="cuda:0"):
        ffa.search_raw(series, plan, device="opencl:0")
    with pytest.raises(ValueError, match="cuda:0"):
        ffa.search_raw(series, plan, device="cuda:x")
    with pytest.raises(TypeError):
        ffa.search_raw(series, plan, backend="cuda")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ffa.search_raw(series, plan, device_id=1)  # type: ignore[call-arg]


def test_search_rejects_invalid_device() -> None:
    series = np.zeros(8, dtype=np.float32)
    plan = make_plan(series.size)

    with pytest.raises(ValueError, match="cuda:0"):
        ffa.search(series, plan, device="gpu:0")
    with pytest.raises(ValueError, match="cuda:0"):
        ffa.search(series, plan, device="cuda:x")


def test_cuda_program_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        ffa.CudaProgram(make_plan(8), device=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="series_tile_size"):
        ffa.CudaProgram(make_plan(8), series_tile_size=True)  # type: ignore[arg-type]


@pytest.mark.skipif(gaffa.cuda_device_count() == 0, reason="CUDA device required")
def test_search_raw_cuda_matches_cpu() -> None:
    series = np.array([0, 0, 0, 5, 0, 0, 0, 5], dtype=np.float32)
    plan = make_plan(series.size)

    cpu_peaks = ffa.search_raw(series, plan, snr_threshold=1.0)
    cuda_peaks = ffa.search_raw(
        series, plan, snr_threshold=1.0, device="cuda:0"
    )

    assert len(cuda_peaks) == len(cpu_peaks)
    for cpu_peak, cuda_peak in zip(cpu_peaks, cuda_peaks, strict=True):
        assert cuda_peak.period == pytest.approx(cpu_peak.period)
        assert cuda_peak.frequency == pytest.approx(cpu_peak.frequency)
        assert cuda_peak.snr == pytest.approx(cpu_peak.snr)
        assert cuda_peak.width == cpu_peak.width
        assert cuda_peak.phase == cpu_peak.phase
        assert cuda_peak.shift == cpu_peak.shift
        assert cuda_peak.bins == cpu_peak.bins


@pytest.mark.skipif(gaffa.cuda_device_count() == 0, reason="CUDA device required")
def test_search_cuda_and_program_return_canonical_peaks() -> None:
    series = np.array([0, 0, 0, 5, 0, 0, 0, 5], dtype=np.float32)
    plan = make_plan(series.size)

    cpu_result = ffa.search(series, plan, snr_threshold=1.0)
    one_shot = ffa.search(series, plan, snr_threshold=1.0, device="cuda:0")
    with ffa.CudaProgram(plan, device="cuda:0", series_tile_size=2) as program:
        repeated = program.search(series, snr_threshold=1.0)
        batch = program.search_batch(
            np.stack([series, series]), snr_threshold=1.0
        )

    assert len(one_shot) == len(cpu_result)
    assert len(repeated) == len(cpu_result)
    assert len(batch) == 2
    assert batch[0] and batch[1]
    assert one_shot[0].motion.frequency_hz == pytest.approx(
        cpu_result[0].motion.frequency_hz
    )
    assert repeated[0].snr == pytest.approx(cpu_result[0].snr)

    with pytest.raises(RuntimeError, match="closed"):
        program.search(series)


def test_search_dms_cpu_returns_physical_peaks_for_one_block() -> None:
    data = np.array(
        [
            [0, 0, 0, 5, 0, 0, 0, 5],
            [0, 0, 0, 7, 0, 0, 0, 7],
        ],
        dtype=np.float32,
    )
    block = dedispersion.DedispersedResult(
        data, tsamp=1.0, dm_low=10.0, dm_step=0.5
    )
    preprocess_plan = preprocessing.make_riptide_plan(
        tsamp=1.0,
        running_median_width_seconds=3.0,
        running_median_min_points=3,
    )

    result = ffa.search_dms_cpu(
        block,
        make_plan(block.nsamples),
        preprocess=preprocess_plan,
        dm_index_offset=20,
        snr_threshold=0.0,
    )

    assert result
    assert all(isinstance(peak, peaks.DmPeak) for peak in result)
    assert {peak.dm for peak in result} == {10.0, 10.5}
    assert {peak.dm_index for peak in result} == {20, 21}
    assert all(peak.peak.motion.reference_time_seconds == 4.0 for peak in result)


def test_search_dms_cpu_rejects_mismatched_contracts() -> None:
    block = dedispersion.DedispersedResult(
        np.ones((2, 8), dtype=np.float32),
        tsamp=1.0,
        dm_low=10.0,
        dm_step=0.5,
    )
    preprocess_plan = preprocessing.make_riptide_plan(
        tsamp=1.0,
        running_median_width_seconds=3.0,
        running_median_min_points=3,
    )

    with pytest.raises(ValueError, match="tsamp"):
        ffa.search_dms_cpu(
            block,
            ffa.make_riptide_plan(
                nsamples=8,
                tsamp=0.5,
                period_min=2.0,
                period_max=4.0,
                bins_min=2,
                bins_max=2,
            ),
            preprocess=preprocess_plan,
        )
    with pytest.raises(ValueError, match="non-negative"):
        ffa.search_dms_cpu(
            block,
            make_plan(8),
            preprocess=preprocess_plan,
            dm_index_offset=-1,
        )
