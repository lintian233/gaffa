"""Type-only fragment for preprocessing bindings."""


class PreprocessPlan:
    """Immutable-by-contract ordered preprocessing definition."""

    step_count: int
    """Number of operations in the plan."""

    def __repr__(self) -> str: ...


def _make_riptide_preprocess_plan(
    *,
    tsamp: float,
    running_median_width_seconds: float = 5.0,
    running_median_min_points: int = 101,
    normalise: bool = True,
    reject_constant: bool = True,
) -> PreprocessPlan: ...
