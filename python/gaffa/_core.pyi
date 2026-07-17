"""Type surface for the compiled :mod:`gaffa._core` extension.

This module is the physical pybind extension boundary. Domain declarations live
next to their public Python packages so the binding surface does not become the
documentation home for every Gaffa subsystem. Most users should import from
``gaffa.io``, ``gaffa.dedispersion``, ``gaffa.ffa``, or ``gaffa.pfold``.
"""

from ._runtime import (
    cuda_device_count as cuda_device_count,
    cuda_runtime_version as cuda_runtime_version,
    vector_add as vector_add,
)
from .dedispersion._bindings import (
    Backend as Backend,
    DedispersedArray as DedispersedArray,
    DedispersedResult as DedispersedResult,
    DedispersedSpectrum as DedispersedSpectrum,
    DedispersedSpectrumArray as DedispersedSpectrumArray,
    dedisperse_multi_dm as dedisperse_multi_dm,
    dedisperse_single_dm as dedisperse_single_dm,
    dedisperse_spectrum as dedisperse_spectrum,
    dedisperse_subband as dedisperse_subband,
)
from .ffa._bindings import (
    FfaPeak as FfaPeak,
    FfaPlan as FfaPlan,
    _ffa_search_cpu as _ffa_search_cpu,
    _ffa_search_cuda_host as _ffa_search_cuda_host,
    _make_riptide_ffa_plan as _make_riptide_ffa_plan,
)
from .io._bindings import (
    ChannelOrder as ChannelOrder,
    ChannelOrderPolicy as ChannelOrderPolicy,
    Filterbank as Filterbank,
    FilterbankArray as FilterbankArray,
    FilterbankHeader as FilterbankHeader,
    PathLikeStr as PathLikeStr,
    ReverseBackend as ReverseBackend,
    ReverseBackendName as ReverseBackendName,
)
from .pfold._bindings import (
    FoldCubeArray as FoldCubeArray,
    FoldExposureArray as FoldExposureArray,
    FoldResult as FoldResult,
    FoldedProfile as FoldedProfile,
    _fold_dedispersed_profile as _fold_dedispersed_profile,
    _fold_dedispersed_spectrum as _fold_dedispersed_spectrum,
)

__all__ = [
    "Backend",
    "ChannelOrder",
    "ChannelOrderPolicy",
    "DedispersedArray",
    "DedispersedResult",
    "DedispersedSpectrum",
    "DedispersedSpectrumArray",
    "FfaPeak",
    "FfaPlan",
    "Filterbank",
    "FilterbankArray",
    "FilterbankHeader",
    "FoldCubeArray",
    "FoldExposureArray",
    "FoldResult",
    "FoldedProfile",
    "PathLikeStr",
    "ReverseBackend",
    "ReverseBackendName",
    "_ffa_search_cpu",
    "_ffa_search_cuda_host",
    "_fold_dedispersed_profile",
    "_fold_dedispersed_spectrum",
    "_make_riptide_ffa_plan",
    "cuda_device_count",
    "cuda_runtime_version",
    "dedisperse_multi_dm",
    "dedisperse_single_dm",
    "dedisperse_spectrum",
    "dedisperse_subband",
    "vector_add",
]
