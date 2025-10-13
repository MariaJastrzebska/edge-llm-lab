#!/usr/bin/env python3
"""
Demo script showing how to use optimization presets.
This is a standalone demo that doesn't require a model to run.
"""

import sys
from pathlib import Path

# Add the framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from edge_llm_lab.utils.optimization import (
    get_optimisations,
    get_optimal_kv_cache_type,
    get_mobile_optimizations,
    get_desktop_optimizations
)


def print_optimizations(opts, title):
    """Helper to print optimizations in a readable format."""
    print(f"\n{title}")
    print("-" * 60)
    for i, opt in enumerate(opts, 1):
        if not opt:
            print(f"   {i}. Baseline (no optimizations)")
        else:
            opt_items = []
            for k, v in opt.items():
                if v is None:
                    opt_items.append(k)
                else:
                    opt_items.append(f"{k}={v}")
            print(f"   {i}. {', '.join(opt_items)}")


def main():
    """Demonstrate optimization presets."""
    
    print("🔧 Edge LLM Lab - Optimization System Demo")
    print("=" * 60)
    
    # Example 1: Determine optimal KV cache type
    print("\n📊 Example 1: Determine Optimal KV Cache Type")
    print("-" * 60)
    
    quantization_examples = ["fp16", "fp8", "q8_0", "q4_0", "q2_0", "Q4_K_M", "Q5_K_S"]
    
    for quant in quantization_examples:
        kv_cache = get_optimal_kv_cache_type(quant)
        print(f"   {quant:12} → {kv_cache}")
    
    # Example 2: Get all optimizations
    print("\n📋 Example 2: All Available Optimizations")
    print("-" * 60)
    
    kv_cache = "q8_0"  # Example quantization
    individual, selected = get_optimisations(kv_cache)
    
    print(f"\n   Test Mode (individual optimizations): {len(individual)} configs")
    print(f"   Production Mode (selected best): {len(selected)} configs")
    
    # Example 3: Desktop optimizations
    print("\n🖥️  Example 3: Desktop Optimizations")
    desktop_opts = get_desktop_optimizations(kv_cache)
    print_optimizations(desktop_opts, "   Desktop Configurations:")
    
    # Example 4: Mobile optimizations  
    print("\n📱 Example 4: Mobile Optimizations")
    mobile_opts = get_mobile_optimizations(kv_cache)
    print_optimizations(mobile_opts, "   Mobile Configurations:")
    
    # Example 5: Show how to use in evaluation
    print("\n💡 Example 5: How to Use in Evaluation")
    print("-" * 60)
    print("""
   # Option 1: Test all individual optimizations
   individual, selected = get_optimisations("q8_0")
   evaluator.pipeline_eval_model(
       mode='logs_only',  # Just generate logs, no visualizations
       optimisations=individual
   )
   
   # Option 2: Use production configurations
   evaluator.pipeline_eval_model(
       mode='logs_and_viz',  # Generate logs and visualizations
       optimisations=selected
   )
   
   # Option 3: Use device-specific optimizations
   desktop_opts = get_desktop_optimizations("q8_0")
   evaluator.pipeline_eval_model(
       mode='logs_and_viz',
       optimisations=desktop_opts
   )
   
   # Option 4: Test a single custom optimization
   custom_opt = [{"--flash-attn": None, "--cont-batching": None}]
   evaluator.pipeline_eval_model(
       mode='logs_only',
       optimisations=custom_opt
   )
   
   # Option 5: Baseline (no optimizations)
   evaluator.pipeline_eval_model(
       mode='logs_and_viz',
       optimisations=[{}]  # Empty dict = no optimizations
   )
    """)
    
    # Example 6: Explain optimization parameters
    print("\n📖 Example 6: Optimization Parameters Explained")
    print("-" * 60)
    print("""
   Common llama-server optimization parameters:
   
   GPU Offloading:
      --n-gpu-layers=N     Offload N layers to GPU (-1 = all, 0 = none)
      
   KV Cache:
      --cache-type-k=TYPE  KV cache quantization for K (f16, q8_0, q4_0)
      --cache-type-v=TYPE  KV cache quantization for V (f16, q8_0, q4_0)
      
   Performance:
      --flash-attn         Enable Flash Attention (faster, less memory)
      --cont-batching      Enable continuous batching (better throughput)
      
   Threading & Batching:
      --threads=N          Number of CPU threads (1-32)
      --batch-size=N       Batch size for prompt processing
      --ubatch-size=N      Micro-batch size
      
   Memory Management:
      --no-mmap            Disable memory mapping (mobile friendly)
      --no-kv-offload      Keep KV cache in RAM (mobile without GPU)
      --mlock              Lock model in RAM (prevent swapping)
      
   Speculative Decoding:
      --draft-max=N        Max tokens for speculative decoding (2-5)
      --draft-p-min=P      Minimum probability threshold (0.0-1.0)
    """)
    
    print("\n✅ Demo complete!")
    print("\n💡 Next steps:")
    print("   1. Check examples/desktop/basic_evaluation.py for full example")
    print("   2. Modify src/edge_llm_lab/utils/optimization.py to add custom configs")
    print("   3. Run your own evaluations with different optimization presets")


if __name__ == "__main__":
    main()

