"""Reusable time-series preprocessing plans."""

from ._bindings import PreprocessPlan as PreprocessPlan


def make_riptide_plan(
    *,
    tsamp: float,
    running_median_width_seconds: float = 5.0,
    running_median_min_points: int = 101,
    normalise: bool = True,
    reject_constant: bool = True,
) -> PreprocessPlan:
    """Build the standard running-median and normalisation plan.

    Parameters
    ----------
    tsamp
        Sampling interval in seconds.
    running_median_width_seconds
        Requested centered running-median width in seconds.
    running_median_min_points
        Odd minimum number of low-resolution samples used by the approximate
        running-median path.
    normalise
        Append zero-mean, unit-variance normalisation after detrending.
    reject_constant
        Reject constant input when normalisation is enabled.
    """
    ...


__all__ = ["PreprocessPlan", "make_riptide_plan"]
