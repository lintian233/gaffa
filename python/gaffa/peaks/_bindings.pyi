"""Type-only fragment for backend-neutral periodic peak bindings."""

from enum import Enum


class MotionOrder(Enum):
    FREQUENCY: MotionOrder
    ACCELERATION: MotionOrder
    JERK: MotionOrder
    SNAP: MotionOrder


class PeriodicMotion:
    order: MotionOrder
    reference_time_seconds: float
    frequency_hz: float
    acceleration_m_per_s2: float
    jerk_m_per_s3: float
    snap_m_per_s4: float

    def __repr__(self) -> str: ...


class PeriodicPeak:
    motion: PeriodicMotion
    phase_bin: int | None
    phase_bins: int
    boxcar_width_bins: int
    duty_cycle: float
    snr: float
    period_seconds: float

    def __repr__(self) -> str: ...


class DmPeak:
    dm: float
    dm_index: int
    peak: PeriodicPeak

    def __repr__(self) -> str: ...
