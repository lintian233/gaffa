"""Python-facing dedispersion API.

The functions in this module return host-resident NumPy arrays. Device-resident
CUDA result objects are intentionally not part of the public Python API yet.
"""

from gaffa._core import (
    DedispersedResult,
    dedisperse_multi_dm,
    dedisperse_single_dm,
    dedisperse_subband,
)

__all__: list[str]
