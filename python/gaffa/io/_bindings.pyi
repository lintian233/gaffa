"""Type-only fragment for filterbank I/O bindings.

This module is not importable at runtime. Python code must import these objects
from ``gaffa.io`` or ``gaffa._core``.
"""

from __future__ import annotations

from enum import Enum
from os import PathLike
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

PathLikeStr: TypeAlias = str | PathLike[str]
ChannelOrder: TypeAlias = Literal["frequency_ascending", "preserve_file_order"]
ReverseBackendName: TypeAlias = Literal["auto", "cpu_scalar", "cpu_openmp"]
FilterbankArray: TypeAlias = (
    NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32]
)


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
    nsamples: int
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

    ``data`` has shape ``(nsamples, nifs, nchans)``. By default, channels are
    normalized to ascending frequency order.
    """

    header: FilterbankHeader
    data: FilterbankArray

    def __init__(
        self,
        path: PathLikeStr,
        *,
        channel_order: ChannelOrder = "frequency_ascending",
        reverse_backend: ReverseBackendName = "auto",
        io_buffer_bytes: int = 67_108_864,
        openmp_min_rows: int = 4096,
    ) -> None: ...

    @property
    def shape(self) -> tuple[int, int, int]: ...

    @property
    def dtype(self) -> np.dtype[np.generic]: ...

    @property
    def nbytes(self) -> int: ...

    def __repr__(self) -> str: ...
