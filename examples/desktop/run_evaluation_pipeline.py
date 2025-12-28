import os
import sys
import argparse

# Add project root and src to path for direct script execution
try:
    _file_path = os.path.abspath(__file__)
    # examples/desktop/run_evaluation_pipeline.py -> ../.. gets to project root
    _project_root = os.path.abspath(os.path.join(os.path.dirname(_file_path), "../../"))
    _src_path = os.path.join(_project_root, "src")
    if _src_path not in sys.path:
        sys.path.insert(0, _src_path)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    pass

from edge_llm_lab.evaluation.referenced_evaluator import EvalModelsReferenced
from edge_llm_lab.core.model_config_loader import load_stage_config
from edge_llm_lab.utils.base_eval import Agent, BaseEvaluation
from edge_llm_lab.evaluation.optimization.optimization_engine import OptimizationEngine

def run_pipeline(agent_type: str = "constant_data_en"):
    print("🚀 Starting 3-Stage Evaluation Pipeline")
    
    # STAGE 1: Model Selection
    print("\n--- STAGE 1: Model Selection ---")
    stage1_config = load_stage_config(1, agent_type)
    models = [m["name"] for m in stage1_config.get("models_to_evaluate", [])]
    
    golden_model = None
    best_score = -1
    
    for model_name in models:
        print(f"🧐 Evaluating {model_name}...")
        evaluator = EvalModelsReferenced(model_name=model_name, agent=Agent(agent_type))
        # Logic to extract score after eval...
        # For demonstration, simplify:
        # Pipeline evaluation with logs and visualization enabled
        evaluator.pipeline_eval_model(mode="logs_and_viz")
        # golden_model logic here...
        
    # STAGE 2: Quantization Analysis
    print("\n--- STAGE 2: Quantization Analysis ---")
    # Using a placeholder for golden_model if not determined above
    golden_model = golden_model or models[0] 
    stage2_config = load_stage_config(2, agent_type)
    # Filter stage2_config for golden_model if necessary
    
    # STAGE 3: Optuna Optimization
    print("\n--- STAGE 3: Optuna Optimization ---")
    opt_engine = OptimizationEngine(study_name=f"opt_{golden_model}")
    
    def objective(trial):
        # Sample parameters from trial based on custom_optimizations.yaml
        params = {
            "temperature": trial.suggest_float("temperature", 0.0, 1.0),
            "top_p": trial.suggest_float("top_p", 0.0, 1.0)
        }
        # Run one evaluation session with these params
        return 0.8 # Mock score
        
    best_params = opt_engine.run_optimization(objective, n_trials=5)
    print(f"🎯 Best inference parameters: {best_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="constant_data_en", help="Agent type to evaluate")
    args = parser.parse_args()
    run_pipeline(args.agent)
