import os

import numpy as np
import pytest
from gaffa.peaks import PeriodicPeak

import gaffa

_loki_build_dir = os.environ.get("GAFFA_LOKI_BUILD_DIR")
if _loki_build_dir:
    gaffa.__path__.append(_loki_build_dir)

try:
    from gaffa import loki
except ImportError:
    pytest.skip("Loki Python extension is not enabled", allow_module_level=True)


def make_plan(nsamples: int = 1 << 18) -> loki.PffaPlan:
    return loki.make_pffa_plan(
        nsamples=nsamples,
        tsamp=64e-6,
        frequency=(100.0, 110.0),
        phase_bins_min=64,
        phase_bins_max=64,
        snr_threshold=3.0,
    )


def make_signal(nsamples: int = 1 << 18) -> np.ndarray:
    time = np.arange(nsamples, dtype=np.float64) * 64e-6
    return np.sin(2.0 * np.pi * 105.0 * time).astype(np.float32)


def sort_peaks(peaks: list[PeriodicPeak]) -> list[PeriodicPeak]:
    return sorted(
        peaks,
        key=lambda peak: (
            peak.motion.frequency_hz,
            peak.motion.acceleration_m_per_s2,
            peak.motion.jerk_m_per_s3,
            peak.motion.snap_m_per_s4,
            peak.phase_bins,
            peak.boxcar_width_bins,
            peak.snr,
        ),
    )


def test_make_pffa_plan_exposes_stable_metadata() -> None:
    plan = make_plan(1 << 12)

    assert isinstance(plan, loki.PffaPlan)
    assert plan.nsamples == 1 << 12
    assert plan.tsamp == pytest.approx(64e-6)
    assert repr(plan) == "<PffaPlan nsamples=4096>"


def test_make_pffa_plan_accepts_taylor_ranges() -> None:
    plan = loki.make_pffa_plan(
        nsamples=1 << 12,
        tsamp=64e-6,
        frequency=(100.0, 110.0),
        acceleration=(-2.0, 2.0),
        jerk=(-0.5, 0.5),
        phase_bins_min=64,
        phase_bins_max=64,
    )

    assert plan.nsamples == 1 << 12


def test_make_pffa_plan_rejects_malformed_ranges() -> None:
    with pytest.raises(ValueError, match="pair"):
        loki.make_pffa_plan(
            nsamples=1 << 12,
            tsamp=64e-6,
            frequency=(100.0,),
        )
    with pytest.raises(ValueError):
        loki.make_pffa_plan(
            nsamples=1000,
            tsamp=64e-6,
            frequency=(100.0, 110.0),
        )


def test_pffa_program_rejects_invalid_configuration_without_cuda_work() -> None:
    plan = make_plan(1 << 12)

    with pytest.raises(ValueError, match="cuda:0"):
        loki.PffaProgram(plan, device="cuda:x")
    with pytest.raises(ValueError, match="positive"):
        loki.PffaProgram(plan, max_peaks_per_series=0)
    with pytest.raises(ValueError, match="positive"):
        loki.PffaProgram(plan, max_peaks_per_series=True)  # type: ignore[arg-type]


@pytest.mark.skipif(gaffa.cuda_device_count() == 0, reason="CUDA device required")
def test_pffa_program_rejects_invalid_host_contracts() -> None:
    plan = make_plan()
    signal = make_signal()

    with loki.PffaProgram(plan) as program:
        with pytest.raises(TypeError, match="numpy.ndarray"):
            program.search(signal.tolist())
        with pytest.raises(TypeError, match="float32"):
            program.search(signal.astype(np.float64))
        with pytest.raises(ValueError, match="1D"):
            program.search(signal.reshape(1, -1))
        with pytest.raises(ValueError, match="C-contiguous"):
            program.search(signal[::2])
        with pytest.raises(ValueError, match="must match"):
            program.search(np.zeros(signal.size // 2, dtype=np.float32))
        with pytest.raises(ValueError, match="1D series or 2D batch"):
            program.search_batch(signal.reshape(1, 1, -1))


@pytest.mark.skipif(gaffa.cuda_device_count() == 0, reason="CUDA device required")
def test_module_search_matches_reusable_program() -> None:
    plan = make_plan()
    signal = make_signal()
    batch = np.stack([signal, signal])

    one_shot = loki.search(signal, plan)
    with loki.PffaProgram(plan) as program:
        reusable = program.search(signal)
        reusable_batch = program.search_batch(batch)

    assert one_shot
    assert len(one_shot) == len(reusable)
    assert all(isinstance(peak, PeriodicPeak) for peak in one_shot)
    one_shot = sort_peaks(one_shot)
    reusable = sort_peaks(reusable)
    for expected, actual in zip(one_shot, reusable, strict=True):
        assert actual.motion.frequency_hz == pytest.approx(
            expected.motion.frequency_hz
        )
        assert actual.snr == pytest.approx(expected.snr, rel=1e-4, abs=1e-5)
        assert actual.phase_bins == expected.phase_bins
        assert actual.boxcar_width_bins == expected.boxcar_width_bins
    assert len(reusable_batch) == 2
    assert reusable_batch[0] and reusable_batch[1]


@pytest.mark.skipif(gaffa.cuda_device_count() == 0, reason="CUDA device required")
def test_pffa_program_close_releases_python_handle() -> None:
    program = loki.PffaProgram(make_plan())
    program.close()

    with pytest.raises(RuntimeError, match="closed"):
        program.search(make_signal())
    with pytest.raises(RuntimeError, match="closed"):
        _ = program.device


def test_pffa_program_searches_host_series_and_batch() -> None:
    if gaffa.cuda_device_count() == 0:
        pytest.skip("CUDA device required")

    plan = make_plan()
    signal = make_signal()

    with loki.PffaProgram(plan, device="cuda:0") as program:
        peaks = program.search(signal)
        batch = program.search_batch(np.stack([signal, signal]))

    assert peaks
    assert all(isinstance(peak, PeriodicPeak) for peak in peaks)
    assert len(batch) == 2
    assert batch[0] and batch[1]
