"""Reusable time-series preprocessing plans."""

from .._core import PreprocessPlan, _make_riptide_preprocess_plan


def make_riptide_plan(
    *,
    tsamp: float,
    running_median_width_seconds: float = 5.0,
    running_median_min_points: int = 101,
    normalise: bool = True,
    reject_constant: bool = True,
) -> PreprocessPlan:
    """Build the standard running-median and normalisation plan."""

    return _make_riptide_preprocess_plan(
        tsamp=tsamp,
        running_median_width_seconds=running_median_width_seconds,
        running_median_min_points=running_median_min_points,
        normalise=normalise,
        reject_constant=reject_constant,
    )


__all__ = ["PreprocessPlan", "make_riptide_plan"]
