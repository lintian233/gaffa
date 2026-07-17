"""Python-facing dedispersion API.

The functions in this module return host-resident NumPy arrays. Device-resident
CUDA result objects are intentionally not part of the public Python API yet.
"""

from .._core import (
    DedispersedResult,
    DedispersedSpectrum,
    dedisperse_multi_dm,
    dedisperse_single_dm,
    dedisperse_spectrum,
    dedisperse_subband,
)

__all__ = [
    "DedispersedResult",
    "DedispersedSpectrum",
    "dedisperse_multi_dm",
    "dedisperse_single_dm",
    "dedisperse_spectrum",
    "dedisperse_subband",
]
