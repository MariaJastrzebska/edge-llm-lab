import os
import sys
import argparse
from datetime import datetime

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
from edge_llm_lab.core.model_config_loader import load_stage_config, delete_model
from edge_llm_lab.utils.base_eval import Agent, BaseEvaluation
from edge_llm_lab.evaluation.optimization.optimization_engine import OptimizationEngine

def run_pipeline(agent_type: str = "constant_data_en", mode: str = "logs_and_viz"):
    print("🚀 Starting 3-Stage Evaluation Pipeline")
    
    # STAGE 1: Model Selection
    print("\n--- STAGE 1: Model Selection ---")
    stage1_config = load_stage_config(1, agent_type)
    models = [m["name"] for m in stage1_config.get("models_to_evaluate", [])]
    
    last_successful_model = None
    golden_model = None
    golden_model = None
    best_score = -1
    
    for i, model_name in enumerate(models):
        print(f"\n🧐 Evaluating {model_name} ({i+1}/{len(models)})...")
        
        # Check model availability and auto-pull if needed (skip in viz_only mode)
        if mode != "viz_only" and not BaseEvaluation.check_model_availability(model_name, install_choice='y'):
            print(f"⏭️ Skipping {model_name}")
            continue
        
        try:
            evaluator = EvalModelsReferenced(model_name=model_name, agent=Agent(agent_type))

            # Pipeline evaluation with logs and visualization enabled
            evaluator.pipeline_eval_model(mode=mode, stage_name="stage_1_selection", generate_comparison=True, generate_per_round=False)
            
            # Simple logic to track the best model (Golden Model)
            # In a real scenario, we'd extract the actual score from the logs
            # For now, we'll use the last successful model as a candidate
            golden_model = model_name 
            
            # Sequential Cleanup: Delete the PREVIOUS successful model
            if last_successful_model and last_successful_model != model_name:
                print(f"🧹 Sequential Cleanup: Removing previous successful model {last_successful_model}")
                #delete_model(last_successful_model)
            
            last_successful_model = model_name
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            # If current model failed, we DON'T delete the previous one yet 
            # as it might still be needed for comparison or fallback.
            continue
            
    # Generate Global Comparison after Stage 1
    if last_successful_model and mode != "logs_only":
         print("\n📊 Generating Global Comparison for Stage 1...")
         try:
             summary_evaluator = EvalModelsReferenced(model_name=last_successful_model, agent=Agent(agent_type))
             summary_evaluator.pipeline_eval_model(
                 mode="viz_only", 
                 stage_name="stage_1_summary",
                 generate_per_round=False,  # ✅ Generate per-round plots
                 generate_per_model=True,  # ✅ Generate aggregated plots (radar, GPT score, etc.)
                 generate_comparison=True,
                 neptune_tags_list=["all_models_summary"]
             )
         except Exception as e:
             print(f"❌ Error generating summary plots: {e}")
            
    # # STAGE 2: Quantization Analysis
    # print("\n--- STAGE 2: Quantization Analysis ---")
    # # Using the best determined model from Stage 1
    # if not golden_model and models:
    #     golden_model = models[0]
        
    # print(f"🏆 Golden Model for Quantization: {golden_model}")
    # stage2_config = load_stage_config(2, agent_type)
    # # logic for quantization comparison...
    # if stage2_config and "models_to_evaluate" in stage2_config:
    #      q_models = [m["name"] for m in stage2_config["models_to_evaluate"]]
    #      for q_model_name in q_models:
    #          print(f"\n📏 Evaluating Quantized Model: {q_model_name}")
             
    #          # Check model availability and auto-pull if needed
    #          if not BaseEvaluation.check_model_availability(q_model_name, install_choice='y'):
    #              print(f"⏭️ Skipping {q_model_name}")
    #              continue
             
    #          try:
    #              evaluator = EvalModelsReferenced(model_name=q_model_name, agent=Agent(agent_type))
    #              evaluator.pipeline_eval_model(mode=mode, stage_name="stage_2_quantization", optimisations_choice="test")
    #          except Exception as e:
    #              print(f"❌ Error evaluating {q_model_name}: {e}")
    
    # # STAGE 3: Optuna Optimization
    # print("\n--- STAGE 3: Optuna Optimization ---")
    # opt_engine = OptimizationEngine(study_name=f"opt_{golden_model}")
    
    # def objective(trial):
    #     # Sample parameters from trial
    #     params = {
    #         "temperature": trial.suggest_float("temperature", 0.0, 1.0),
    #         "top_p": trial.suggest_float("top_p", 0.0, 1.0)
    #     }
    #     # In a real run, this would call evaluator.run_inference with these params
    #     return 0.8 # Mock score
        
    # best_params = opt_engine.run_optimization(objective, n_trials=5)
    # print(f"🎯 Best inference parameters for {golden_model}: {best_params}")

    # # Final Cleanup
    # if last_successful_model:
    #     print(f"🧹 Final Cleanup: Removing last model {last_successful_model}")
    #     #delete_model(last_successful_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="constant_data_en", help="Agent type to evaluate")
    parser.add_argument("--mode", default="logs_and_viz", choices=["logs_only", "logs_and_viz", "viz_only"], help="Evaluation mode")
    args = parser.parse_args()
    run_pipeline(args.agent, args.mode)
