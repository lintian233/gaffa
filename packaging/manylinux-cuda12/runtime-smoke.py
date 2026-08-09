from __future__ import annotations

import subprocess

import numpy as np

import gaffa
from gaffa import ffa, loki


def signal(nsamples: int, tsamp: float, frequency: float) -> np.ndarray:
    time = np.arange(nsamples, dtype=np.float64) * tsamp
    return np.sin(2.0 * np.pi * frequency * time).astype(np.float32)


def main() -> None:
    if gaffa.cuda_device_count() == 0:
        raise RuntimeError("runtime smoke test requires a visible CUDA device")

    nsamples = 1 << 18
    tsamp = 64e-6
    data = signal(nsamples, tsamp, 105.0)

    native_plan = ffa.make_riptide_plan(
        nsamples=nsamples,
        tsamp=tsamp,
        period_min=0.008,
        period_max=0.012,
        bins_min=64,
        bins_max=64,
    )
    ffa.search(data, native_plan, device="cuda:0", snr_threshold=3.0)

    loki_plan = loki.make_pffa_plan(
        nsamples=nsamples,
        tsamp=tsamp,
        frequency=(100.0, 110.0),
        phase_bins_min=64,
        phase_bins_max=64,
        snr_threshold=3.0,
    )
    peaks = loki.search(data, loki_plan, device="cuda:0", max_peaks=100_000)
    if not peaks:
        raise RuntimeError("Loki smoke search returned no peaks")

    subprocess.run(["gaffa_search", "--help"], check=True)
    print(f"CUDA wheel smoke test passed with {len(peaks)} Loki peaks")


if __name__ == "__main__":
    main()
