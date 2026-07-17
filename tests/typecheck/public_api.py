"""Static-only fixture for Gaffa's stable Python-facing API."""

from gaffa.dedispersion import DedispersedResult
from gaffa.ffa import FfaPeak, FfaPlan, ffa_search, make_riptide_plan
from gaffa.pfold import FoldedProfile, fold_profile


def use_public_api(samples: DedispersedResult) -> tuple[FfaPlan, list[FfaPeak], FoldedProfile]:
    plan = make_riptide_plan(
        nsamples=samples.nsamples,
        tsamp=samples.tsamp,
        period_min=0.1,
        period_max=1.0,
    )
    peaks = ffa_search(samples.data[0].astype("float32"), plan)
    profile = fold_profile(samples, period=0.5, nbin=128)
    return plan, peaks, profile
