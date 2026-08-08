"""Static-only fixture for Gaffa's stable Python-facing API."""

import numpy as np
from gaffa.dedispersion import DedispersedResult
from gaffa.ffa import (
    CudaProgram,
    FfaPeak,
    FfaPlan,
    PeriodicPeak,
    make_riptide_plan,
    search,
    search_batch,
    search_dms_cpu,
    search_raw,
)
from gaffa.peaks import DmPeak
from gaffa.pfold import FoldedProfile, fold_profile
from gaffa.preprocessing import make_riptide_plan as make_preprocess_plan
from numpy.typing import NDArray

from gaffa.loki import PffaPlan, PffaProgram, make_pffa_plan


def use_public_api(samples: DedispersedResult) -> tuple[FfaPlan, list[FfaPeak], FoldedProfile]:
    plan = make_riptide_plan(
        nsamples=samples.nsamples,
        tsamp=samples.tsamp,
        period_min=0.1,
        period_max=1.0,
    )
    peaks = search_raw(samples.data[0].astype("float32"), plan)
    profile = fold_profile(samples, period=0.5, nbin=128)
    return plan, peaks, profile


def use_dm_search(samples: DedispersedResult) -> list[DmPeak]:
    plan = make_riptide_plan(
        nsamples=samples.nsamples,
        tsamp=samples.tsamp,
        period_min=0.1,
        period_max=1.0,
    )
    preprocess = make_preprocess_plan(tsamp=samples.tsamp)
    return search_dms_cpu(samples, plan, preprocess=preprocess)


def use_periodic_search(
    series: NDArray[np.float32], plan: FfaPlan
) -> tuple[list[PeriodicPeak], list[list[PeriodicPeak]]]:
    peaks = search(series, plan)
    batch_peaks = search_batch(series[None, :], plan)
    return peaks, batch_peaks


def use_cuda_program(
    series: NDArray[np.float32], plan: FfaPlan
) -> tuple[list[PeriodicPeak], list[list[PeriodicPeak]]]:
    with CudaProgram(plan, device="cuda:0") as program:
        peaks = program.search(series)
        batch_peaks = program.search_batch(series[None, :])
    return peaks, batch_peaks


def use_loki_program(
    series: NDArray[np.float32],
) -> tuple[PffaPlan, list[PeriodicPeak]]:
    plan = make_pffa_plan(
        nsamples=series.shape[0],
        tsamp=1.0,
        frequency=(1.0, 2.0),
    )
    with PffaProgram(plan, device="cuda:0") as program:
        peaks = program.search(series)
    return plan, peaks
