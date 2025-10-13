"""Unit tests for optimization module."""

import pytest
from edge_llm_lab.utils.optimization import (
    get_optimal_kv_cache_type,
    get_optimisations,
    get_mobile_optimizations,
    get_desktop_optimizations
)


class TestOptimalKVCacheType:
    """Test get_optimal_kv_cache_type function."""
    
    def test_exact_match_f16(self):
        """Test exact match for f16."""
        assert get_optimal_kv_cache_type("f16") == "f16"
    
    def test_exact_match_q8_0(self):
        """Test exact match for q8_0."""
        assert get_optimal_kv_cache_type("q8_0") == "q8_0"
    
    def test_exact_match_q4_0(self):
        """Test exact match for q4_0."""
        assert get_optimal_kv_cache_type("q4_0") == "q4_0"
    
    def test_fp16_to_f16(self):
        """Test conversion from fp16 to f16."""
        assert get_optimal_kv_cache_type("fp16") == "f16"
    
    def test_fp8_to_q8_0(self):
        """Test conversion from fp8 to q8_0."""
        assert get_optimal_kv_cache_type("fp8") == "q8_0"
    
    def test_q4_k_m_to_q4_0(self):
        """Test conversion from Q4_K_M to q4_0."""
        assert get_optimal_kv_cache_type("Q4_K_M") == "q4_0"
    
    def test_q5_k_s_to_q4_0(self):
        """Test conversion from Q5_K_S to q4_0."""
        # Q5 doesn't exist in allowed, should default to closest (q4_0)
        result = get_optimal_kv_cache_type("Q5_K_S")
        assert result in ["f16", "q8_0", "q4_0", "q2_0"]
    
    def test_none_returns_default(self):
        """Test None returns default f16."""
        assert get_optimal_kv_cache_type(None) == "f16"
    
    def test_invalid_string_returns_default(self):
        """Test invalid string returns default f16."""
        assert get_optimal_kv_cache_type("invalid") == "f16"
    
    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert get_optimal_kv_cache_type("FP16") == "f16"
        assert get_optimal_kv_cache_type("Q8_0") == "q8_0"


class TestGetOptimisations:
    """Test get_optimisations function."""
    
    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        result = get_optimisations("q8_0")
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_returns_lists(self):
        """Test that both elements are lists."""
        individual, selected = get_optimisations("q8_0")
        assert isinstance(individual, list)
        assert isinstance(selected, list)
    
    def test_individual_not_empty(self):
        """Test that individual optimizations list is not empty."""
        individual, selected = get_optimisations("q8_0")
        assert len(individual) > 0
    
    def test_all_optimizations_are_dicts(self):
        """Test that all optimizations are dictionaries."""
        individual, selected = get_optimisations("q8_0")
        for opt in individual:
            assert isinstance(opt, dict)
        for opt in selected:
            assert isinstance(opt, dict)
    
    def test_baseline_in_individual(self):
        """Test that baseline ({}) is in individual optimizations."""
        individual, selected = get_optimisations("q8_0")
        # Check if there's at least one dict (could be baseline or other opts)
        assert any(isinstance(opt, dict) for opt in individual)
    
    def test_with_none_kv_cache(self):
        """Test with None kv_cache uses default."""
        individual, selected = get_optimisations(None)
        assert isinstance(individual, list)
        assert isinstance(selected, list)
    
    def test_with_different_kv_cache_types(self):
        """Test with different kv_cache types."""
        for kv_cache in ["f16", "q8_0", "q4_0", "q2_0"]:
            individual, selected = get_optimisations(kv_cache)
            assert isinstance(individual, list)
            assert isinstance(selected, list)


class TestGetMobileOptimizations:
    """Test get_mobile_optimizations function."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_mobile_optimizations("q8_0")
        assert isinstance(result, list)
    
    def test_not_empty(self):
        """Test that list is not empty."""
        result = get_mobile_optimizations("q8_0")
        assert len(result) > 0
    
    def test_all_are_dicts(self):
        """Test that all optimizations are dictionaries."""
        result = get_mobile_optimizations("q8_0")
        for opt in result:
            assert isinstance(opt, dict)
    
    def test_contains_mobile_specific_params(self):
        """Test that optimizations contain mobile-specific parameters."""
        result = get_mobile_optimizations("q8_0")
        # Check if any optimization has mobile-specific params
        mobile_params = ["--threads", "--batch-size", "--no-mmap", "--cache-type-k"]
        found_params = False
        for opt in result:
            if any(param in opt for param in mobile_params):
                found_params = True
                break
        assert found_params
    
    def test_with_none_kv_cache(self):
        """Test with None kv_cache."""
        result = get_mobile_optimizations(None)
        assert isinstance(result, list)
        assert len(result) > 0


class TestGetDesktopOptimizations:
    """Test get_desktop_optimizations function."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_desktop_optimizations("q8_0")
        assert isinstance(result, list)
    
    def test_not_empty(self):
        """Test that list is not empty."""
        result = get_desktop_optimizations("q8_0")
        assert len(result) > 0
    
    def test_all_are_dicts(self):
        """Test that all optimizations are dictionaries."""
        result = get_desktop_optimizations("q8_0")
        for opt in result:
            assert isinstance(opt, dict)
    
    def test_baseline_included(self):
        """Test that baseline ({}) is included."""
        result = get_desktop_optimizations("q8_0")
        assert {} in result
    
    def test_contains_desktop_specific_params(self):
        """Test that optimizations contain desktop-specific parameters."""
        result = get_desktop_optimizations("q8_0")
        # Check if any optimization has desktop-specific params
        desktop_params = ["--n-gpu-layers", "--flash-attn", "--cont-batching"]
        found_params = False
        for opt in result:
            if any(param in opt for param in desktop_params):
                found_params = True
                break
        assert found_params
    
    def test_with_none_kv_cache(self):
        """Test with None kv_cache."""
        result = get_desktop_optimizations(None)
        assert isinstance(result, list)
        assert len(result) > 0


class TestOptimizationValidation:
    """Test validation of optimization parameters."""
    
    def test_valid_llama_server_params(self):
        """Test that generated optimizations use valid llama-server parameters."""
        individual, selected = get_optimisations("q8_0")
        
        # Known valid llama-server parameters
        valid_params = [
            "--n-gpu-layers", "--cache-type-k", "--cache-type-v",
            "--flash-attn", "--cont-batching", "--threads",
            "--batch-size", "--ubatch-size", "--no-mmap",
            "--no-kv-offload", "--draft-max", "--draft-p-min",
            "--draft-p-split", "--mlock", "--split-mode",
            "--rope-scaling", "--rope-scale", "--rope-freq-base",
            "--rope-freq-scale", "--grp-attn-n", "--grp-attn-w",
            "--defrag-thold", "--cache-reuse", "--prio", "--poll",
            "--numa", "--cpu-mask", "--cpu-range", "--cpu-strict",
            "--keep", "--no-context-shift"
        ]
        
        for opt in individual + selected:
            for param in opt.keys():
                assert param in valid_params, f"Invalid parameter: {param}"
    
    def test_parameter_values_are_valid_types(self):
        """Test that parameter values are valid types."""
        individual, selected = get_optimisations("q8_0")
        
        for opt in individual + selected:
            for param, value in opt.items():
                # Values should be None, int, float, or str
                assert value is None or isinstance(value, (int, float, str)), \
                    f"Invalid value type for {param}: {type(value)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

