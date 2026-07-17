"""Type-only fragment for dedispersion bindings.

This module is not importable at runtime. Python code must import these objects
from ``gaffa.dedispersion`` or ``gaffa._core``.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..io._bindings import Filterbank

Backend: TypeAlias = Literal["cpu", "cuda"]
DedispersedArray: TypeAlias = NDArray[np.uint32] | NDArray[np.float32]
DedispersedSpectrumArray: TypeAlias = (
    NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32]
)


class DedispersedResult:
    """Host-resident dedispersed time series.

    ``data`` has shape ``(ndm, nsamples)``. Integer filterbank input produces
    ``uint32`` output and ``float32`` input produces ``float32`` output. The
    time axis is valid-only, so trailing samples requiring an out-of-range
    channel delay are omitted.
    """

    data: DedispersedArray
    """Dedispersed time series with shape ``(ndm, nsamples)``."""

    backend: str
    """Backend or provenance label that produced the result."""

    dm_low: float
    """First dispersion measure represented by the result."""

    dm_step: float
    """DM spacing between adjacent rows; single-DM results use zero."""

    ndm: int
    """Number of DM rows in ``data``."""

    nsamples: int
    """Number of time samples per DM row."""

    tsamp: float
    """Sampling interval in seconds."""

    def __init__(
        self,
        data: DedispersedArray,
        *,
        tsamp: float,
        dm_low: float = 0.0,
        dm_step: float = 0.0,
        backend: str = "external",
    ) -> None:
        """Wrap a C-contiguous ``(ndm, nsamples)`` array without copying.

        ``data`` must have dtype ``uint32`` or ``float32``. ``dm_low`` and
        ``dm_step`` must be non-negative.
        """
        ...

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def dtype(self) -> np.dtype[np.generic]: ...

    @property
    def nbytes(self) -> int: ...

    def __repr__(self) -> str: ...


class DedispersedSpectrum:
    """Host-resident single-DM aligned dynamic spectrum.

    ``data`` has shape ``(nsamples, nchans)`` and a valid-only time axis. This
    materializes channel-resolved output and is intended for workflows that
    explicitly need an aligned spectrum.
    """

    data: DedispersedSpectrumArray
    """Aligned spectrum with shape ``(nsamples, nchans)``."""

    backend: str
    """Backend or provenance label that produced the result."""

    dm: float
    """Non-negative dispersion measure used for channel alignment."""

    nsamples: int
    """Number of valid time samples."""

    nchans: int
    """Number of selected frequency channels."""

    tsamp: float
    """Sampling interval in seconds."""

    chan_begin: int
    """Inclusive source-channel start."""

    chan_end: int
    """Exclusive source-channel end."""

    def __init__(
        self,
        data: DedispersedSpectrumArray,
        *,
        tsamp: float,
        dm: float = 0.0,
        chan_begin: int = 0,
        chan_end: int | None = None,
        backend: str = "external",
    ) -> None:
        """Wrap a C-contiguous ``(nsamples, nchans)`` array without copying.

        ``data`` must have dtype ``uint8``, ``uint16``, or ``float32``. If
        omitted, ``chan_end`` defaults to ``chan_begin + data.shape[1]``.
        """
        ...

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def dtype(self) -> np.dtype[np.generic]: ...

    @property
    def nbytes(self) -> int: ...

    def __repr__(self) -> str: ...


def dedisperse_spectrum(
    filterbank: Filterbank,
    *,
    dm: float,
    backend: Backend = "cpu",
    device_id: int = 0,
    threads_per_block: int = 256,
    time_tile_samples: int = 81920,
) -> DedispersedSpectrum:
    """Dedisperse one DM and return a host-resident aligned dynamic spectrum.

    Parameters
    ----------
    filterbank
        Loaded filterbank with exactly one IF/polarisation stream.
    dm
        Non-negative dispersion measure used to align channels.
    backend
        ``"cpu"`` or ``"cuda"``. CUDA never silently falls back to CPU.
    device_id
        CUDA device ordinal used by the CUDA backend.
    threads_per_block
        CUDA thread-block size used by the CUDA backend.
    time_tile_samples
        CUDA time-tile setting. The current spectrum CUDA path materializes
        the full output, so this does not reduce returned-spectrum memory.

    Returns
    -------
    DedispersedSpectrum
        Host-resident valid-only aligned spectrum with shape
        ``(nsamples, nchans)``.

    Notes
    -----
    Unlike :func:`dedisperse_single_dm`, this does not sum over channels.
    Memory use is proportional to ``nsamples * nchans * dtype.itemsize``.
    """
    ...


def dedisperse_single_dm(
    filterbank: Filterbank,
    *,
    dm: float,
    backend: Backend = "cpu",
    device_id: int = 0,
    threads_per_block: int = 256,
    time_tile_samples: int = 81920,
) -> DedispersedResult:
    """Dedisperse one DM into a host-resident valid-only time series.

    Integer filterbank input produces ``uint32`` output; ``float32`` input
    produces ``float32`` output. The result has shape ``(1, nsamples)``.
    ``backend``, ``device_id``, ``threads_per_block``, and
    ``time_tile_samples`` have the same semantics as :func:`dedisperse_spectrum`.
    """
    ...


def dedisperse_multi_dm(
    filterbank: Filterbank,
    *,
    dm_low: float,
    dm_step: float,
    ndm: int,
    backend: Backend = "cpu",
    device_id: int = 0,
    threads_per_block: int = 256,
    time_tile_samples: int = 81920,
) -> DedispersedResult:
    """Dedisperse a contiguous DM grid into host-resident time series.

    ``dm_step`` must be positive. Output row ``i`` represents
    ``dm_low + i * dm_step``. Every row shares one valid-only length determined
    by the largest delay in the requested DM/channel grid.
    """
    ...


def dedisperse_subband(
    filterbank: Filterbank,
    *,
    dm_low: float,
    dm_step: float,
    ndm: int,
    backend: Backend = "cuda",
    subband_channels: int = 32,
    ndm_per_nominal: int = 32,
    device_id: int = 0,
    threads_per_block: int = 256,
    time_tile_samples: int = 81920,
) -> DedispersedResult:
    """Dedisperse a contiguous DM grid with the subband method.

    This is the preferred CUDA-facing many-DM API. ``subband_channels`` groups
    adjacent frequency channels and ``ndm_per_nominal`` controls sharing of the
    nominal subband stage. The result is host-resident and valid-only.
    """
    ...
