"""
Optimization configurations for llama-server inference.
Based on thesis_generators/referenced_clean.py from fed-mobile project.
"""

from typing import Dict, List, Tuple, Optional
import re


def get_optimal_kv_cache_type(quantization: Optional[str] = None) -> str:
    """
    Determine optimal KV cache type based on model quantization level.
    
    Args:
        quantization: Model quantization level (e.g., "fp16", "q8_0", "q4_0")
        
    Returns:
        str: Optimal KV cache type (one of: "f16", "q8_0", "q4_0", "q2_0")
        
    Example:
        >>> get_optimal_kv_cache_type("fp16")
        'f16'
        >>> get_optimal_kv_cache_type("q8_0")
        'q8_0'
        >>> get_optimal_kv_cache_type("Q4_K_M")
        'q4_0'
    """
    allowed = ["f16", "q8_0", "q4_0", "q2_0"]

    if quantization is None:
        return "f16"  # Fallback

    normalized_quantization = str(quantization).lower()
    
    if normalized_quantization in allowed:
        return normalized_quantization
    
    # Extract number from input (e.g., "16" from "fp16")
    number_match = re.search(r'\d+', normalized_quantization)
    if not number_match:
        return "f16"  # Fallback if no number found
    input_number = int(number_match.group())  # e.g., 16 as int
    
    # Compare numerically with numbers in allowed
    for option in allowed:
        option_number_match = re.search(r'\d+', option)
        if option_number_match:
            option_number = int(option_number_match.group())  # e.g., 16 from "f16"
            if input_number == option_number:
                return option
    
    return "f16"  # Fallback if no match


def get_optimisations(kv_cache: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Get optimization configurations for llama-server testing.
    
    Returns two lists:
    1. Individual optimizations (test mode) - for systematic testing
    2. Selected optimizations (production mode) - best performing combinations
    
    Args:
        kv_cache: KV cache type to use. If None, uses "f16" as default.
        
    Returns:
        Tuple of (individual_optimizations, selected_optimizations)
        Each optimization is a Dict with llama-server parameters.
        
    Example:
        >>> individual, selected = get_optimisations("q8_0")
        >>> print(len(individual))  # Number of test optimizations
        >>> print(selected[0])  # First production optimization
        
    Usage in evaluation:
        ```python
        # For testing all optimizations
        individual, selected = get_optimisations(kv_cache="q8_0")
        evaluator.pipeline_eval_model(mode="logs_only", optimisations=individual)
        
        # For production use (best combinations only)
        evaluator.pipeline_eval_model(mode="logs_and_viz", optimisations=selected)
        ```
    """
    if kv_cache is None:
        kv_cache = "f16"
    
    # PHASE 1: Individual optimizations (for testing)
    # Each optimization is tested separately to understand its impact
    individual_optimizations = [
        {},  # Baseline - no optimizations
        
        # ========== CURRENTLY TESTING ==========
        {"--n-gpu-layers": 99},  # All layers on GPU (safe: if model has <99 layers)
        {"--n-gpu-layers": -1},  # All layers on GPU (auto-detect: may be faster)
        
        # ========== COMMENTED OUT - ALREADY TESTED ==========
        
        # Basic KV Cache optimizations - each separately
        # {"--cache-type-k": kv_cache},
        # {"--cache-type-v": kv_cache},
        # {"--cache-type-k": kv_cache, "--cache-type-v": kv_cache},  # KV Cache
        
        # {"--flash-attn": None},  # Flash Attention
        # {"--cont-batching": None},  # Continuous Batching
        
        # Mobile memory and performance optimizations
        # {"--no-kv-offload": None},  # Disable KV offload for mobile without GPU
        # {"--no-mmap": None},  # Disable memory mapping for constrained memory
        # {"--threads": 1},  # Single thread for very constrained devices
        # {"--threads": 2},  # Conservative mobile
        # {"--threads": 4},  # Standard mobile
        # {"--batch-size": 4, "--ubatch-size": 4},  # Very small batches
        # {"--batch-size": 8, "--ubatch-size": 8},  # Small mobile batches
        # {"--batch-size": 16, "--ubatch-size": 16},  # Standard mobile batches
        # {"--batch-size": 32, "--ubatch-size": 32},  # Larger batches for edge
        
        # Speculative decoding for mobile (if you have draft model)
        # {"--draft-max": 2},  # Minimal speculative
        # {"--draft-max": 3},  # Standard speculative
        # {"--draft-max": 4},  # Aggressive speculative
        # {"--draft-max": 5},  # Very aggressive speculative
        # {"--draft-max": 3, "--draft-p-min": 0.5},  # Speculative with probability
        # {"--draft-max": 3, "--draft-p-min": 0.7},  # More conservative
        # {"--draft-max": 3, "--draft-p-split": 0.3},  # Split probability
        
        # Two-component combinations - KV Cache + basic
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--flash-attn": None
        # },
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--cont-batching": None
        # },
        # {
        #     "--flash-attn": None,
        #     "--cont-batching": None
        # },
        
        # Combinations with thread and batch optimization
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--threads": 4,
        #     "--batch-size": 16,
        #     "--ubatch-size": 16
        # },
        # {
        #     "--flash-attn": None,
        #     "--threads": 4,
        #     "--batch-size": 32,
        #     "--ubatch-size": 32
        # },
        # {
        #     "--cont-batching": None,
        #     "--threads": 2,
        #     "--batch-size": 8,
        #     "--ubatch-size": 8
        # },
        
        # Memory-constrained combinations for mobile
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--no-mmap": None,
        #     "--threads": 2
        # },
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--no-kv-offload": None,
        #     "--threads": 1
        # },
        # {
        #     "--no-mmap": None,
        #     "--no-kv-offload": None,
        #     "--threads": 2,
        #     "--batch-size": 8,
        #     "--ubatch-size": 8
        # },
        
        # Speculative combinations with optimizations (if draft model available)
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--draft-max": 3
        # },
        # {
        #     "--flash-attn": None,
        #     "--draft-max": 3
        # },
        # {
        #     "--cont-batching": None,
        #     "--draft-max": 3
        # },
        
        # Three-way combinations - basic optimizations
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--flash-attn": None,
        #     "--cont-batching": None
        # },
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--flash-attn": None,
        #     "--threads": 4
        # },
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--cont-batching": None,
        #     "--threads": 4
        # },
        
        # Ultra-conservative mobile (very constrained devices)
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--threads": 1,
        #     "--batch-size": 4,
        #     "--ubatch-size": 4,
        #     "--no-mmap": None,
        #     "--no-kv-offload": None
        # },
        
        # Standard mobile optimization
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--flash-attn": None,
        #     "--threads": 4,
        #     "--batch-size": 16,
        #     "--ubatch-size": 16
        # },
        
        # Memory-priority mobile
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--no-mmap": None,
        #     "--threads": 2,
        #     "--batch-size": 8,
        #     "--ubatch-size": 8
        # },
        
        # Android CPU optimized
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--cont-batching": None,
        #     "--no-kv-offload": None,
        #     "--threads": 4
        # },
        
        # Edge device (Raspberry Pi style)
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--threads": 2,
        #     "--batch-size": 4,
        #     "--ubatch-size": 4,
        #     "--no-mmap": None
        # },
        
        # High-performance edge
        # {
        #     "--flash-attn": None,
        #     "--cont-batching": None,
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--threads": 8,
        #     "--batch-size": 32,
        #     "--ubatch-size": 32
        # },
        
        # Full mobile combinations with speculative (when draft model available)
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--flash-attn": None,
        #     "--draft-max": 3,
        #     "--threads": 4
        # },
        # {
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--cont-batching": None,
        #     "--draft-max": 3,
        #     "--threads": 4
        # },
        
        # All optimizations together - maximum performance
        # {
        #     "--flash-attn": None,
        #     "--cont-batching": None,
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--threads": 4,
        #     "--batch-size": 16,
        #     "--ubatch-size": 16
        # },
        
        # All optimizations + speculative
        # {
        #     "--flash-attn": None,
        #     "--cont-batching": None,
        #     "--cache-type-k": kv_cache,
        #     "--cache-type-v": kv_cache,
        #     "--draft-max": 3,
        #     "--threads": 4,
        #     "--batch-size": 16,
        #     "--ubatch-size": 16
        # }
    ]
    
    # PHASE 2: Selected combinations (for production)
    # These are the best-performing combinations based on previous testing
    selected_optimisation = [
        {
            # Best configuration goes here
            # Example: {"--cache-type-k": kv_cache, "--cache-type-v": kv_cache, "--flash-attn": None}
        },
    ]
    
    print(f" Optimization plan:")
    print(f"    Individual optimizations (test mode): {len(individual_optimizations)}")
    print(f"    Selected optimizations (production mode): {len(selected_optimisation)}")
    
    return individual_optimizations, selected_optimisation


def get_mobile_optimizations(kv_cache: Optional[str] = None) -> List[Dict]:
    """
    Get optimizations specifically for mobile devices.
    
    Args:
        kv_cache: KV cache type to use
        
    Returns:
        List of optimization dictionaries optimized for mobile
    """
    if kv_cache is None:
        kv_cache = "f16"
    
    return [
        # Ultra-conservative for very constrained devices
        {
            "--cache-type-k": kv_cache,
            "--cache-type-v": kv_cache,
            "--threads": 1,
            "--batch-size": 4,
            "--ubatch-size": 4,
            "--no-mmap": None,
            "--no-kv-offload": None
        },
        # Standard mobile
        {
            "--cache-type-k": kv_cache,
            "--cache-type-v": kv_cache,
            "--threads": 2,
            "--batch-size": 8,
            "--ubatch-size": 8,
            "--no-mmap": None
        },
        # High-performance mobile
        {
            "--cache-type-k": kv_cache,
            "--cache-type-v": kv_cache,
            "--flash-attn": None,
            "--threads": 4,
            "--batch-size": 16,
            "--ubatch-size": 16
        },
    ]


def get_desktop_optimizations(kv_cache: Optional[str] = None) -> List[Dict]:
    """
    Get optimizations specifically for desktop devices.
    
    Args:
        kv_cache: KV cache type to use
        
    Returns:
        List of optimization dictionaries optimized for desktop
    """
    if kv_cache is None:
        kv_cache = "f16"
    
    return [
        # Baseline
        {},
        # Standard desktop
        {
            "--cache-type-k": kv_cache,
            "--cache-type-v": kv_cache,
            "--flash-attn": None,
            "--cont-batching": None
        },
        # High-performance desktop
        {
            "--flash-attn": None,
            "--cont-batching": None,
            "--cache-type-k": kv_cache,
            "--cache-type-v": kv_cache,
            "--threads": 8,
            "--batch-size": 32,
            "--ubatch-size": 32
        },
        # GPU-optimized (for Mac with Metal or NVIDIA)
        {
            "--n-gpu-layers": -1,  # All layers on GPU
            "--cache-type-k": kv_cache,
            "--cache-type-v": kv_cache,
            "--flash-attn": None
        },
    ]

