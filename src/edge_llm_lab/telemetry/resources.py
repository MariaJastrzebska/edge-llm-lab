"""Resource (RAM/swap/CPU) and energy sampling.

Ported from ``utils/base_eval.py::BaseEvaluation.monitor_resources`` /
``calculate_resource_differences``, with the energy source turned into a
pluggable backend so the lab can run beyond the original macOS-only
powermetrics path (Linux RAPL / Jetson tegrastats backends implement the same
one-method interface).

KNOWN LIMITATION (carried over, to fix in the paper-grade layer): a backend
sample is an INSTANTANEOUS power reading (mW), not integrated energy (J).
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional, Protocol


class EnergyBackend(Protocol):
    """One instantaneous power sample. Values in mW; None when unavailable."""

    def sample(self) -> dict: ...


class NullEnergyBackend:
    """No energy measurement available on this host."""

    def sample(self) -> dict:
        return {"cpu_power_mw": None, "gpu_power_mw": None, "total_power_mw": None}


class MacPowermetricsBackend:
    """macOS powermetrics (requires passwordless sudo for powermetrics)."""

    def sample(self) -> dict:
        cpu_power: Optional[float] = None
        gpu_power: Optional[float] = None
        try:
            cmd = "sudo -n powermetrics -n 1 -i 1000"
            output = subprocess.check_output(
                cmd, shell=True, text=True, timeout=10, stderr=subprocess.DEVNULL
            )
            for line in output.split("\n"):
                if "CPU Power:" in line and "mW" in line:
                    try:
                        cpu_power = float(line.split("CPU Power:")[1].split("mW")[0].strip())
                    except (IndexError, ValueError):
                        pass
                if "GPU Power:" in line and "mW" in line:
                    try:
                        gpu_power = float(line.split("GPU Power:")[1].split("mW")[0].strip())
                    except (IndexError, ValueError):
                        pass
        except Exception:
            return NullEnergyBackend().sample()

        total = None
        if cpu_power is not None and gpu_power is not None:
            total = cpu_power + gpu_power
        elif cpu_power is not None:
            total = cpu_power
        return {
            "cpu_power_mw": cpu_power,
            "gpu_power_mw": gpu_power,
            "total_power_mw": total,
        }


def default_energy_backend() -> EnergyBackend:
    if platform.system() == "Darwin":
        return MacPowermetricsBackend()
    return NullEnergyBackend()


def _cpu_freq():
    """psutil.cpu_freq() jako dict albo None.

    Na Apple Silicon psutil podnosi RuntimeError ("'pmgr' entry was not found
    in AppleARMIODevice service") zamiast zwrócić None — a to jest dokładnie
    ten sprzęt, na którym laboratorium mierzy. Częstotliwość jest metadaną
    opisową, więc jej brak nie może wywracać zbierania telemetrii.
    """
    import psutil

    try:
        freq = psutil.cpu_freq()
    except (RuntimeError, NotImplementedError, OSError):
        return None
    return freq._asdict() if freq else None


def monitor_resources(energy_backend: Optional[EnergyBackend] = None) -> dict:
    """Sample memory, device info, and one energy reading."""
    import psutil

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()

    device_info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": _cpu_freq(),
    }

    memory_info = {
        "ram_total_gb": memory.total / (1024**3),
        "ram_used_gb": memory.used / (1024**3),
        "swap_used_gb": swap.used / (1024**3),
    }

    backend = energy_backend or default_energy_backend()
    energy_info = backend.sample()

    return {"memory": memory_info, "energy": energy_info, "device": device_info}


def calculate_resource_differences(start_resources: dict, end_resources: dict) -> Optional[dict]:
    """Calculate differences between start and end resources."""
    if not start_resources or not end_resources:
        return None

    start_memory = start_resources.get("memory", {})
    end_memory = end_resources.get("memory", {})
    start_energy = start_resources.get("energy", {})
    end_energy = end_resources.get("energy", {})

    memory_diff = {
        "ram_delta_gb": end_memory.get("ram_used_gb", 0) - start_memory.get("ram_used_gb", 0),
        "swap_delta_gb": end_memory.get("swap_used_gb", 0) - start_memory.get("swap_used_gb", 0),
        "ram_start_gb": start_memory.get("ram_used_gb", 0),
        "ram_end_gb": end_memory.get("ram_used_gb", 0),
        "swap_start_gb": start_memory.get("swap_used_gb", 0),
        "swap_end_gb": end_memory.get("swap_used_gb", 0),
    }

    start_cpu = start_energy.get("cpu_power_mw")
    end_cpu = end_energy.get("cpu_power_mw")
    start_gpu = start_energy.get("gpu_power_mw")
    end_gpu = end_energy.get("gpu_power_mw")

    energy_diff = {
        "cpu_power_start_mw": start_cpu,
        "cpu_power_end_mw": end_cpu,
        "gpu_power_start_mw": start_gpu,
        "gpu_power_end_mw": end_gpu,
        "cpu_power_delta_mw": (end_cpu - start_cpu)
        if start_cpu is not None and end_cpu is not None
        else None,
        "gpu_power_delta_mw": (end_gpu - start_gpu)
        if start_gpu is not None and end_gpu is not None
        else None,
    }

    return {"memory": memory_diff, "energy": energy_diff}
