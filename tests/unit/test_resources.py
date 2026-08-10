"""Tests for the ported resource/energy telemetry."""

from edge_llm_lab.telemetry import (
    NullEnergyBackend,
    calculate_resource_differences,
    monitor_resources,
)


def test_monitor_resources_shape_with_null_energy_backend():
    sample = monitor_resources(energy_backend=NullEnergyBackend())
    assert set(sample.keys()) == {"memory", "energy", "device"}
    assert sample["memory"]["ram_total_gb"] > 0
    assert sample["energy"]["cpu_power_mw"] is None
    assert sample["device"]["cpu_count"] >= 1


def test_calculate_resource_differences_deltas():
    start = {
        "memory": {"ram_used_gb": 4.0, "swap_used_gb": 1.0},
        "energy": {"cpu_power_mw": 1000.0, "gpu_power_mw": 500.0},
    }
    end = {
        "memory": {"ram_used_gb": 6.5, "swap_used_gb": 1.25},
        "energy": {"cpu_power_mw": 4000.0, "gpu_power_mw": 2500.0},
    }
    diff = calculate_resource_differences(start, end)
    assert diff["memory"]["ram_delta_gb"] == 2.5
    assert diff["memory"]["swap_delta_gb"] == 0.25
    assert diff["energy"]["cpu_power_delta_mw"] == 3000.0
    assert diff["energy"]["gpu_power_delta_mw"] == 2000.0


def test_calculate_resource_differences_handles_missing_energy():
    start = {"memory": {"ram_used_gb": 4.0}, "energy": {"cpu_power_mw": None}}
    end = {"memory": {"ram_used_gb": 5.0}, "energy": {"cpu_power_mw": None}}
    diff = calculate_resource_differences(start, end)
    assert diff["energy"]["cpu_power_delta_mw"] is None
    assert diff["memory"]["ram_delta_gb"] == 1.0


def test_calculate_resource_differences_none_input():
    assert calculate_resource_differences({}, {"memory": {}}) is None
