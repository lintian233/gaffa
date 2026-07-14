"""Type stubs for the compiled gaffa extension module.

Most users should import from public Python modules such as ``gaffa.io`` and
``gaffa.dedispersion`` instead of importing ``gaffa._core`` directly.
"""

from __future__ import annotations

from enum import Enum
from os import PathLike
from typing import Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

PathLikeStr: TypeAlias = str | PathLike[str]
Backend: TypeAlias = Literal["cpu", "cuda"]
ChannelOrder: TypeAlias = Literal["frequency_ascending", "preserve_file_order"]
ReverseBackendName: TypeAlias = Literal["auto", "cpu_scalar", "cpu_openmp"]
FilterbankArray: TypeAlias = (
    NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32]
)
DedispersedArray: TypeAlias = NDArray[np.uint32] | NDArray[np.float32]
DedispersedSpectrumArray: TypeAlias = (
    NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32]
)
FoldCubeArray: TypeAlias = NDArray[np.float32]
FoldExposureArray: TypeAlias = NDArray[np.float64]


class ChannelOrderPolicy(Enum):
    """Channel ordering policy used when reading filterbank files."""

    FrequencyAscending: ChannelOrderPolicy
    """Normalize channels so frequency increases along the channel axis."""

    PreserveFileOrder: ChannelOrderPolicy
    """Keep the channel order stored in the input file."""


class ReverseBackend(Enum):
    """CPU backend used for channel reversal during filterbank loading."""

    Auto: ReverseBackend
    """Choose the channel reversal implementation automatically."""

    CpuScalar: ReverseBackend
    """Use the scalar CPU reversal path."""

    CpuOpenmp: ReverseBackend
    """Use the OpenMP CPU reversal path when OpenMP is available."""


class FilterbankHeader:
    """Parsed SIGPROC filterbank header.

    The numeric fields mirror the filterbank header. Frequencies are in MHz,
    ``tsamp`` is in seconds, and ``frequency_table`` contains one frequency per
    channel in the same order as ``Filterbank.data``.
    """

    header_size: int
    """Header size in bytes."""

    nsamples: int
    """Number of time samples in the data section."""

    telescope_id: int
    machine_id: int
    data_type: int
    barycentric: int
    pulsarcentric: int
    ibeam: int
    nbeams: int
    npuls: int
    nbins: int
    nbits: int
    """Number of bits per input sample: currently 8, 16, or 32."""

    nifs: int
    """Number of IF/polarisation streams. Dedispersion currently requires 1."""

    nchans: int
    """Number of frequency channels."""

    az_start: float
    za_start: float
    src_raj: float
    src_dej: float
    tstart: float
    tsamp: float
    """Sampling interval in seconds."""

    fch1: float
    """Frequency of the first stored channel in MHz."""

    foff: float
    """Frequency offset between adjacent channels in MHz."""

    refdm: float
    period: float
    rawdatafile: str
    source_name: str
    frequency_table: list[float]
    """Per-channel frequencies in MHz, aligned with ``Filterbank.data``."""

    uses_frequency_table: bool
    """Whether the file provided an explicit frequency table."""

    def __repr__(self) -> str: ...


class Filterbank:
    """In-memory SIGPROC filterbank.

    ``data`` is a NumPy array with shape ``(nsamples, nifs, nchans)``. By
    default, channels are normalized to ascending frequency order. Use
    ``channel_order="preserve_file_order"`` when the on-disk channel order must
    be kept exactly.
    """

    header: FilterbankHeader
    """Parsed filterbank header."""

    data: FilterbankArray
    """Sample array with shape ``(nsamples, nifs, nchans)``."""

    def __init__(
        self,
        path: PathLikeStr,
        *,
        channel_order: ChannelOrder = "frequency_ascending",
        reverse_backend: ReverseBackendName = "auto",
        io_buffer_bytes: int = 67_108_864,
        openmp_min_rows: int = 4096,
    ) -> None:
        """Read a SIGPROC filterbank file into host memory.

        Parameters
        ----------
        path
            Input filterbank path.
        channel_order
            ``"frequency_ascending"`` normalizes the channel axis to ascending
            frequency. ``"preserve_file_order"`` keeps the file order.
        reverse_backend
            Backend used when channel reversal is needed.
        io_buffer_bytes
            Host I/O buffer size used while reading sample data.
        openmp_min_rows
            Minimum row count before the OpenMP reversal path is used.

        Raises
        ------
        ValueError
            If an option value is not recognized.
        RuntimeError
            If the file cannot be read or the header/data layout is invalid.
        """
        ...

    @property
    def shape(self) -> tuple[int, int, int]:
        """Alias for ``data.shape``."""
        ...

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Alias for ``data.dtype``."""
        ...

    @property
    def nbytes(self) -> int:
        """Alias for ``data.nbytes``."""
        ...

    def __repr__(self) -> str: ...


class DedispersedResult:
    """Host-resident dedispersed time series.

    ``data`` has shape ``(ndm, nsamples)``. Integer filterbank inputs produce
    ``uint32`` output; ``float32`` inputs produce ``float32`` output. The time
    axis contains only samples for which every selected channel contributes a
    valid input sample, so ``nsamples`` can be smaller than the source
    filterbank length by the global maximum dispersion delay.
    """

    data: DedispersedArray
    """Dedispersed time series with shape ``(ndm, nsamples)``."""

    backend: str
    """Backend or provenance label that produced the result."""

    dm_low: float
    """First DM represented by the result."""

    dm_step: float
    """DM spacing between adjacent rows. Single-DM results use 0."""

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
        """Wrap an existing dedispersed time-series array without copying.

        Parameters
        ----------
        data
            C-contiguous array with shape ``(ndm, nsamples)`` and dtype
            ``uint32`` or ``float32``.
        tsamp
            Sampling interval in seconds.
        dm_low
            First DM represented by ``data``. Must be non-negative.
        dm_step
            DM spacing between adjacent rows. Must be non-negative. Single-DM
            arrays usually use 0.
        backend
            Provenance label. External arrays default to ``"external"``.
        """
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Alias for ``data.shape``."""
        ...

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Alias for ``data.dtype``."""
        ...

    @property
    def nbytes(self) -> int:
        """Alias for ``data.nbytes``."""
        ...

    def __repr__(self) -> str: ...


class DedispersedSpectrum:
    """Host-resident single-DM aligned dynamic spectrum.

    ``data`` has shape ``(nsamples, nchans)`` and keeps the input sample dtype.
    The time axis is valid-only: trailing samples that would require
    out-of-range delayed input samples are omitted. This API materializes the
    full aligned spectrum in host memory and is intended for diagnostics,
    visualization, and workflows that explicitly need channel-resolved output.
    """

    data: DedispersedSpectrumArray
    """Aligned dynamic spectrum with shape ``(nsamples, nchans)``."""

    backend: str
    """Backend or provenance label that produced the result."""

    dm: float
    """Dispersion measure used for channel alignment. Must be non-negative."""

    nsamples: int
    """Number of valid time samples."""

    nchans: int
    """Number of selected frequency channels."""

    tsamp: float
    """Sampling interval in seconds."""

    chan_begin: int
    """Inclusive start channel in the source filterbank."""

    chan_end: int
    """Exclusive end channel in the source filterbank."""

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
        """Wrap an existing aligned dynamic spectrum without copying.

        Parameters
        ----------
        data
            C-contiguous array with shape ``(nsamples, nchans)`` and dtype
            ``uint8``, ``uint16``, or ``float32``.
        tsamp
            Sampling interval in seconds.
        dm
            Dispersion measure used for channel alignment.
        chan_begin
            Inclusive source-channel start.
        chan_end
            Exclusive source-channel end. Defaults to
            ``chan_begin + data.shape[1]``.
        backend
            Provenance label. External arrays default to ``"external"``.
        """
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Alias for ``data.shape``."""
        ...

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Alias for ``data.dtype``."""
        ...

    @property
    def nbytes(self) -> int:
        """Alias for ``data.nbytes``."""
        ...

    def __repr__(self) -> str: ...


class FoldResult:
    """Folded products for a dedispersed dynamic spectrum.

    The main cube has shape ``(nsubint, nchans, nbin)``. Projected products are
    provided for common plotting and inspection workflows.
    """

    cube: FoldCubeArray
    """Folded data with shape ``(nsubint, nchans, nbin)``."""

    exposure: FoldExposureArray
    """Phase exposure with shape ``(nsubint, nbin)``."""

    profile: NDArray[np.float32]
    """Frequency- and time-averaged profile with shape ``(nbin,)``."""

    freq_phase: NDArray[np.float32]
    """Frequency-phase image with shape ``(nchans, nbin)``."""

    time_phase: NDArray[np.float32]
    """Time-phase image with shape ``(nsubint, nbin)``."""

    phase: NDArray[np.float32]
    """Phase coordinate with shape ``(nbin,)``."""

    time: NDArray[np.float64]
    """Subintegration center time in seconds with shape ``(nsubint,)``."""

    nsubint: int
    """Number of subintegrations."""

    nchans: int
    """Number of folded frequency channels or subbands."""

    nbin: int
    """Number of phase bins."""

    period: float
    """Folding period in seconds."""

    tsamp: float
    """Input sampling interval in seconds."""

    tsubint: float
    """Actual subintegration length in seconds after rounding to samples."""

    def __repr__(self) -> str: ...


class FfaPlan:
    """Private core representation of an immutable FFA search plan.

    Use :func:`gaffa.ffa.make_riptide_plan` to construct plans and
    :func:`gaffa.ffa.ffa_search` to execute them.
    """

    task_count: int
    """Number of FFA transform tasks in this plan."""

    width_trials: tuple[int, ...]
    """Global boxcar widths used by raw FFA detection."""

    def __repr__(self) -> str: ...


class FfaPeak:
    """Private core representation of one raw FFA detection peak."""

    period: float
    frequency: float
    snr: float
    width: int
    duty_cycle: float
    phase: int
    shift: int
    bins: int
    width_index: int
    period_index: int

    def __repr__(self) -> str: ...


def _make_riptide_ffa_plan(
    *,
    nsamples: int,
    tsamp: float,
    period_min: float,
    period_max: float,
    bins_min: int = 180,
    bins_max: int = 256,
    min_periods: int = 1,
    duty_cycle_max: float = 0.20,
    width_trial_spacing: float = 1.5,
    max_tasks: int = 1_000_000,
) -> FfaPlan:
    """Private binding used by :func:`gaffa.ffa.make_riptide_plan`."""
    ...


def _ffa_search_cpu(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[FfaPeak]:
    """Private binding used by :func:`gaffa.ffa.ffa_search`."""
    ...


def _ffa_search_cuda_host(
    time_series: NDArray[np.float32],
    plan: FfaPlan,
    *,
    device_id: int = 0,
    snr_threshold: float = 6.0,
    max_peaks: int | None = None,
) -> list[FfaPeak]:
    """Private host-input CUDA binding used by :func:`gaffa.ffa.ffa_search`."""
    ...


class FoldedProfile:
    """Folded profile for a dedispersed 1D time series.

    The profile is exposure-normalized where phase bins received samples and
    zero elsewhere.
    """

    profile: NDArray[np.float32]
    """Integrated folded profile with shape ``(nbin,)``."""

    exposure: NDArray[np.float64]
    """Phase exposure with shape ``(nbin,)``."""

    phase: NDArray[np.float32]
    """Phase coordinate with shape ``(nbin,)``."""

    nbin: int
    """Number of phase bins."""

    period: float
    """Folding period in seconds."""

    tsamp: float
    """Input sampling interval in seconds."""

    def __repr__(self) -> str: ...


def _fold_dedispersed_profile(
    data: DedispersedArray,
    *,
    tsamp: float,
    period: float,
    nbin: int,
    dm_index: int = 0,
) -> FoldedProfile:
    """Private binding used by ``gaffa.pfold.fold_profile``."""
    ...


def _fold_dedispersed_spectrum(
    data: DedispersedSpectrumArray,
    *,
    tsamp: float,
    period: float,
    nbin: int,
    tsubint: float,
    output_channels: int,
) -> FoldResult:
    """Private binding used by ``gaffa.pfold.fold``."""
    ...


def dedisperse_spectrum(
    filterbank: Filterbank,
    *,
    dm: float,
    backend: Backend = "cpu",
    device_id: int = 0,
    threads_per_block: int = 256,
    time_tile_samples: int = 81920,
) -> DedispersedSpectrum:
    """Dedisperse one DM and return the aligned dynamic spectrum.

    Parameters
    ----------
    filterbank
        Loaded filterbank. The current implementation requires
        ``filterbank.header.nifs == 1``.
    dm
        Non-negative dispersion measure used to align channels.
    backend
        Execution backend. ``"cuda"`` requires a visible CUDA device and does
        not fall back to CPU.
    device_id
        CUDA device ordinal used when ``backend="cuda"``.
    threads_per_block
        CUDA thread block size used when ``backend="cuda"``.
    time_tile_samples
        CUDA time-tile length for tiled algorithms. The current spectrum CUDA
        path materializes the full aligned spectrum and does not use this value
        to reduce output memory.

    Returns
    -------
    DedispersedSpectrum
        Host-resident aligned dynamic spectrum with
        ``data.shape == (nsamples, nchans)``.

    Raises
    ------
    ValueError
        If the filterbank shape, backend, or DM plan is invalid.
    RuntimeError
        If CUDA execution fails.

    Notes
    -----
    Unlike ``dedisperse_single_dm``, this function does not sum over channels.
    Integer input dtypes are preserved. Memory use is proportional to
    ``nsamples * nchans * dtype.itemsize``.
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
    """Dedisperse one DM from a loaded filterbank.

    Parameters
    ----------
    filterbank
        Loaded filterbank. The current implementation requires
        ``filterbank.header.nifs == 1``.
    dm
        Dispersion measure to evaluate.
    backend
        Execution backend. ``"cuda"`` requires a visible CUDA device and does
        not fall back to CPU.
    device_id
        CUDA device ordinal used when ``backend="cuda"``.
    threads_per_block
        CUDA thread block size used when ``backend="cuda"``.
    time_tile_samples
        CUDA time-tile length used when ``backend="cuda"``.

    Returns
    -------
    DedispersedResult
        Host-resident result with ``data.shape == (1, nsamples)``.

    Raises
    ------
    ValueError
        If the filterbank shape, backend, or DM plan is invalid.
    RuntimeError
        If CUDA execution fails.

    Notes
    -----
    Integer filterbank inputs produce ``uint32`` output. ``float32`` input
    produces ``float32`` output. The returned time axis is valid-only: trailing
    samples that would require out-of-range delayed input samples are omitted.
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
    """Dedisperse a contiguous DM grid from a loaded filterbank.

    Parameters
    ----------
    filterbank
        Loaded filterbank. The current implementation requires
        ``filterbank.header.nifs == 1``.
    dm_low
        First dispersion measure in the grid.
    dm_step
        Positive spacing between adjacent DM trials.
    ndm
        Number of DM trials.
    backend
        Execution backend. ``"cuda"`` requires a visible CUDA device and does
        not fall back to CPU.
    device_id
        CUDA device ordinal used when ``backend="cuda"``.
    threads_per_block
        CUDA thread block size used when ``backend="cuda"``.
    time_tile_samples
        CUDA time-tile length used when ``backend="cuda"``.

    Returns
    -------
    DedispersedResult
        Host-resident result with ``data.shape == (ndm, nsamples)``. Output row
        ``i`` corresponds to ``dm_low + i * dm_step``.

    Raises
    ------
    ValueError
        If the filterbank shape, backend, or DM grid is invalid.
    RuntimeError
        If CUDA execution fails.

    Notes
    -----
    Integer filterbank inputs produce ``uint32`` output. ``float32`` input
    produces ``float32`` output. All DM rows share the same valid-only length,
    determined by the largest delay over the requested DM/channel grid.
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

    Parameters
    ----------
    filterbank
        Loaded filterbank. The current implementation requires
        ``filterbank.header.nifs == 1``.
    dm_low
        First dispersion measure in the grid.
    dm_step
        Positive spacing between adjacent DM trials.
    ndm
        Number of DM trials.
    backend
        Execution backend. ``"cuda"`` requires a visible CUDA device and does
        not fall back to CPU.
    subband_channels
        Number of adjacent frequency channels grouped into each subband.
    ndm_per_nominal
        Number of adjacent DM trials sharing one nominal subband stage.
    device_id
        CUDA device ordinal used when ``backend="cuda"``.
    threads_per_block
        CUDA thread block size used when ``backend="cuda"``.
    time_tile_samples
        CUDA time-tile length used when ``backend="cuda"``.

    Returns
    -------
    DedispersedResult
        Host-resident result with ``data.shape == (ndm, nsamples)``.

    Raises
    ------
    ValueError
        If the filterbank shape, backend, DM grid, or subband options are
        invalid.
    RuntimeError
        If CUDA execution fails.

    Notes
    -----
    This is the preferred CUDA-facing many-DM API. Integer filterbank inputs
    produce ``uint32`` output. ``float32`` input produces ``float32`` output.
    All DM rows share the same valid-only length, determined by the largest
    delay over the requested DM/channel grid.
    """
    ...


def vector_add(lhs: Sequence[float], rhs: Sequence[float]) -> list[float]:
    """Add two equally sized float sequences with the CUDA demo kernel.

    Parameters
    ----------
    lhs, rhs
        Input sequences with equal length.

    Returns
    -------
    list[float]
        Elementwise sum.
    """
    ...


def cuda_device_count() -> int:
    """Return the number of CUDA devices visible to the runtime.

    Returns
    -------
    int
        Visible CUDA device count.
    """
    ...


def cuda_runtime_version() -> int:
    """Return the CUDA runtime version.

    Returns
    -------
    int
        CUDA runtime version encoded as an integer, for example ``12080``.
    """
    ...
