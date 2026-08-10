"""Telemetry instruments: memory + energy sampling with pluggable backends."""

from .resources import (
    EnergyBackend,
    MacPowermetricsBackend,
    NullEnergyBackend,
    calculate_resource_differences,
    monitor_resources,
)
from .powermetrics_stream import (
    EnergyWindow,
    PowerSample,
    PowermetricsRecorder,
)

__all__ = [
    "EnergyBackend",
    "MacPowermetricsBackend",
    "NullEnergyBackend",
    "calculate_resource_differences",
    "monitor_resources",
    "EnergyWindow",
    "PowerSample",
    "PowermetricsRecorder",
]
