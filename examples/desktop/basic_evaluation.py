#!/usr/bin/env python3
"""
Basic example of using Edge LLM Lab framework for evaluation.

This example shows how to:
1. Load agent configuration
2. Use optimization presets
3. Run evaluation with different optimizations
4. Compare results

Usage:
    python basic_evaluation.py --model llama3:8b --agent data_collection_agent
    python basic_evaluation.py --model llama3:8b --agent data_collection_agent --mode test
"""

import argparse
import os
import sys
from pathlib import Path

# Add the framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from edge_llm_lab.core.future_agent_config import AgentConfig
from edge_llm_lab.evaluation.referenced_evaluator import ReferencedEvaluator
from edge_llm_lab.utils.optimization import (
    get_optimisations, 
    get_optimal_kv_cache_type,
    get_mobile_optimizations,
    get_desktop_optimizations
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Edge LLM Lab evaluation with optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python basic_evaluation.py --model llama3:8b --agent data_collection_agent
  
  # Run with specific optimization mode
  python basic_evaluation.py --model llama3:8b --agent data_collection_agent --mode desktop
  
  # Run baseline only
  python basic_evaluation.py --model llama3:8b --agent data_collection_agent --mode baseline
  
  # Run with custom config
  python basic_evaluation.py --model llama3:8b --agent my_agent --config path/to/config.yaml
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama3:8b",
        help="Model name to evaluate (e.g., llama3:8b, mistral:7b)"
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        default="data_collection_agent",
        help="Agent key from configuration"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation config YAML (default: examples/desktop/config/evaluation_config.yaml)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "test", "production", "desktop", "mobile", "baseline"],
        default="interactive",
        help="Optimization mode to use"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        default="q8_0",
        help="Model quantization level for KV cache optimization (e.g., q8_0, q4_0, f16)"
    )
    
    parser.add_argument(
        "--optimizations",
        type=str,
        default=None,
        help="Comma-separated list of optimization IDs or path to optimizations config file"
    )
    
    parser.add_argument(
        "--list-optimizations",
        action="store_true",
        help="List all available optimizations and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running evaluation"
    )
    
    return parser.parse_args()


def load_optimizations_from_file(filepath):
    """Load optimizations from YAML or JSON file."""
    import json
    import yaml
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Optimizations file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        if filepath.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif filepath.suffix == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Expect format: {"optimizations": [...]}
    if isinstance(data, dict) and 'optimizations' in data:
        return data['optimizations']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid optimizations file format. Expected list or dict with 'optimizations' key")


def list_all_optimizations(kv_cache):
    """List all available optimizations with IDs."""
    individual_opts, selected_opts = get_optimisations(kv_cache)
    desktop_opts = get_desktop_optimizations(kv_cache)
    mobile_opts = get_mobile_optimizations(kv_cache)
    
    print("\n🔧 Available Optimization Sets:")
    print("=" * 80)
    
    print("\n📊 Test Mode (Individual):")
    for i, opt in enumerate(individual_opts):
        opt_str = ", ".join([f"{k}={v}" if v is not None else k for k, v in opt.items()])
        print(f"   [{i}] {opt_str if opt_str else 'Baseline'}")
    
    print("\n🎯 Production Mode (Selected):")
    for i, opt in enumerate(selected_opts):
        opt_str = ", ".join([f"{k}={v}" if v is not None else k for k, v in opt.items()])
        print(f"   [{i}] {opt_str if opt_str else 'Baseline'}")
    
    print("\n🖥️  Desktop Optimizations:")
    for i, opt in enumerate(desktop_opts):
        opt_str = ", ".join([f"{k}={v}" if v is not None else k for k, v in opt.items()])
        print(f"   [{i}] {opt_str if opt_str else 'Baseline'}")
    
    print("\n📱 Mobile Optimizations:")
    for i, opt in enumerate(mobile_opts):
        opt_str = ", ".join([f"{k}={v}" if v is not None else k for k, v in opt.items()])
        print(f"   [{i}] {opt_str if opt_str else 'Baseline'}")
    
    print("\n" + "=" * 80)
    print("\n💡 Usage:")
    print("   --mode test              # Use all test optimizations")
    print("   --mode desktop           # Use desktop optimizations")
    print("   --optimizations 0,2,5    # Use specific optimizations by ID")
    print("   --optimizations file.yaml # Load from config file")


def main():
    """Run evaluation with optimizations."""
    args = parse_args()
    
    # Get optimal KV cache type first (needed for listing)
    kv_cache = get_optimal_kv_cache_type(args.quantization)
    
    # Handle --list-optimizations
    if args.list_optimizations:
        list_all_optimizations(kv_cache)
        return
    
    print("🔧 Edge LLM Lab - Optimization Example")
    print("=" * 60)
    
    # Load agent configuration
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / "config" / "evaluation_config.yaml"
    
    print(f"\n📁 Loading configuration from: {config_path}")
    agent_config = AgentConfig.load_from_yaml(str(config_path))
    
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Agent: {args.agent}")
    print(f"   Quantization: {args.quantization}")
    print(f"   Mode: {args.mode}")
    print(f"   Optimal KV cache type: {kv_cache}")
    
    # Get all optimization sets
    individual_opts, selected_opts = get_optimisations(kv_cache)
    desktop_opts = get_desktop_optimizations(kv_cache)
    mobile_opts = get_mobile_optimizations(kv_cache)
    
    # Select optimizations based on --optimizations parameter
    if args.optimizations:
        print(f"\n   📝 Custom optimizations specified: {args.optimizations}")
        
        # Check if it's a file path
        if Path(args.optimizations).exists():
            print(f"   📂 Loading optimizations from file...")
            try:
                optimizations = load_optimizations_from_file(args.optimizations)
                mode_name = f"Custom from file ({len(optimizations)} configs)"
            except Exception as e:
                print(f"   ❌ Error loading optimizations file: {e}")
                return
        else:
            # Parse as comma-separated indices
            try:
                indices = [int(x.strip()) for x in args.optimizations.split(',')]
                print(f"   📊 Using optimization IDs: {indices}")
                
                # Get optimizations from the mode
                if args.mode == "test":
                    base_opts = individual_opts
                elif args.mode == "production":
                    base_opts = selected_opts
                elif args.mode == "desktop":
                    base_opts = desktop_opts
                elif args.mode == "mobile":
                    base_opts = mobile_opts
                else:
                    base_opts = individual_opts  # default
                
                optimizations = [base_opts[i] for i in indices if i < len(base_opts)]
                mode_name = f"Custom selection ({len(optimizations)} configs)"
                
                if len(optimizations) != len(indices):
                    print(f"   ⚠️  Warning: Some indices were out of range")
                
            except ValueError as e:
                print(f"   ❌ Error parsing optimization indices: {e}")
                print(f"   💡 Use --list-optimizations to see available IDs")
                return
    
    # Select optimizations based on mode
    elif args.mode == "interactive":
        print("\n   Select optimization mode:")
        print("   1. Test mode (all individual optimizations)")
        print("   2. Production mode (selected best combinations)")
        print("   3. Desktop optimizations")
        print("   4. Mobile optimizations")
        print("   5. Baseline only")
        
        choice = input("\n   Enter choice (1-5) [default: 5]: ").strip() or "5"
        
        if choice == "1":
            optimizations = individual_opts
            mode_name = "Test mode"
        elif choice == "2":
            optimizations = selected_opts
            mode_name = "Production mode"
        elif choice == "3":
            optimizations = desktop_opts
            mode_name = "Desktop optimizations"
        elif choice == "4":
            optimizations = mobile_opts
            mode_name = "Mobile optimizations"
        else:
            optimizations = [{}]
            mode_name = "Baseline only"
    else:
        # Non-interactive mode
        mode_map = {
            "test": (individual_opts, "Test mode"),
            "production": (selected_opts, "Production mode"),
            "desktop": (desktop_opts, "Desktop optimizations"),
            "mobile": (mobile_opts, "Mobile optimizations"),
            "baseline": ([{}], "Baseline only")
        }
        optimizations, mode_name = mode_map[args.mode]
    
    print(f"\n   ✅ Selected: {mode_name} ({len(optimizations)} configurations)")
    
    # Show optimizations
    print(f"\n   Optimizations to test:")
    for i, opt in enumerate(optimizations[:5], 1):  # Show first 5
        opt_str = ", ".join([f"{k}={v}" if v is not None else k for k, v in opt.items()])
        print(f"      {i}. {opt_str if opt_str else 'Baseline (no optimizations)'}")
    if len(optimizations) > 5:
        print(f"      ... and {len(optimizations) - 5} more")
    
    if args.dry_run:
        print(f"\n   🏁 Dry run complete. Use --dry-run=false to run evaluation.")
        return
    
    # Initialize evaluator
    print(f"\n   📦 Initializing evaluator...")
    try:
        evaluator = ReferencedEvaluator(
            model_name=args.model,
            agent_config=agent_config,
            agent_key=args.agent
        )
        print(f"   ✅ Evaluator initialized successfully")
        
        # Show evaluation plan
        print(f"\n   📊 Evaluation plan:")
        print(f"      • Model: {args.model}")
        print(f"      • Agent: {args.agent}")
        print(f"      • Optimizations: {len(optimizations)}")
        print(f"      • Mode: logs_only")
        
        # Run evaluation
        print(f"\n   🚀 Starting evaluation...")
        # Uncomment to actually run:
        # evaluator.pipeline_eval_model(mode='logs_only', use_cache=True, optimisations=optimizations)
        print(f"   💡 Uncomment the line above to run actual evaluation")
        
    except Exception as e:
        print(f"   ⚠️  Could not initialize evaluator: {e}")
        print(f"   💡 Make sure:")
        print(f"      • Model '{args.model}' is available via Ollama")
        print(f"      • Agent '{args.agent}' exists in configuration")
    
    print(f"\n✅ Example complete!")
    print(f"\n💡 Next steps:")
    print(f"   • Run with --help to see all options")
    print(f"   • Use --mode desktop for desktop optimizations")
    print(f"   • Use --mode mobile for mobile optimizations")
    print(f"   • Check logs in output/ directory")


if __name__ == "__main__":
    main()
