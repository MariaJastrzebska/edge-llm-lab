#!/usr/bin/env python3
"""
Example of how to use edge-llm-lab in fed-mobile project.
This shows integration with existing medical evaluation setup.
"""

import sys
from pathlib import Path

# Add the framework to path (in real usage, this would be installed via pip)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from edge_llm_lab.core.future_base_evaluation import BaseEvaluation
from edge_llm_lab.core.future_agent_config import ConfigLoader
from edge_llm_lab.evaluation.referenced_evaluator import ReferencedEvaluator
from edge_llm_lab.evaluation.unreferenced_evaluator import UnreferencedEvaluator


def main():
    """Example of using edge-llm-lab in fed-mobile context."""
    
    # Load medical configuration
    config_path = Path(__file__).parent / "medical_config.yaml"
    config = ConfigLoader.load_from_yaml(str(config_path))
    
    # Initialize evaluator for constant data collection
    model_name = "granite3.1-dense:2b"  # Example medical model
    agent_type = "constant_data_en"  # Medical agent from config
    
    evaluator = ReferencedEvaluator(
        model_name=model_name,
        agent_type=agent_type,
        config=config
    )
    
    print(f"🚀 Medical Evaluation Setup")
    print(f"📋 Model: {model_name}")
    print(f"🏥 Agent: {evaluator.agent_config.name}")
    print(f"🖥️  Device: {evaluator.device_info['platform']}")
    
    # Example medical evaluation
    patient_input = "Hi, I'm Sarah, I'm 32 years old, 5'6\" tall, I was born on March 15th, 1992, and I'm a software engineer."
    
    print(f"\n📝 Patient Input: {patient_input}")
    
    # Load medical prompt
    prompt = evaluator.load_prompt()
    print(f"📋 Medical Prompt loaded from: {evaluator.agent_config.prompt_path}")
    
    # Example reference response (what we expect)
    reference_response = {
        "name": "Sarah",
        "age": 32,
        "height": "5'6\"",
        "birth_date": "March 15th, 1992",
        "profession": "software engineer"
    }
    
    print(f"📤 Reference Response: {reference_response}")
    
    # Mock model response
    model_response = {
        "name": "Sarah",
        "age": 32,
        "height": "5'6\"",
        "birth_date": "March 15th, 1992",
        "profession": "software engineer"
    }
    
    # Evaluate with reference
    print(f"\n📊 Evaluating against medical reference...")
    metrics = evaluator.evaluate_with_reference(
        model_response=model_response,
        reference_response=reference_response,
        user_input=patient_input
    )
    
    print(f"\n{metrics}")
    
    # Save medical evaluation log
    log_data = {
        "model_name": model_name,
        "agent_type": agent_type,
        "domain": "medical",
        "patient_input": patient_input,
        "model_response": model_response,
        "reference_response": reference_response,
        "metrics": metrics.__dict__,
        "device_info": evaluator.device_info,
        "medical_config": {
            "prompt_path": evaluator.agent_config.prompt_path,
            "schema_path": evaluator.agent_config.schema_path
        }
    }
    
    evaluator.save_evaluation_log(log_data, "medical_evaluation_demo")
    
    print("\n✅ Medical evaluation complete!")
    print("📁 Check the output/logs directory for results.")
    
    # Example of using unreferenced evaluation for medical QA
    print(f"\n🔍 Testing unreferenced evaluation...")
    
    qa_evaluator = UnreferencedEvaluator(
        model_name=model_name,
        agent_type="symptom",  # Different medical agent
        config=config
    )
    
    # Medical question without reference
    medical_question = "What are the common symptoms of diabetes?"
    mock_answer = "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision."
    
    qa_metrics = qa_evaluator.evaluate_with_llm_judge(
        model_response=mock_answer,
        user_input=medical_question,
        evaluation_criteria="medical accuracy and completeness"
    )
    
    print(f"📊 Medical QA Evaluation:")
    print(f"   Factual Accuracy: {qa_metrics.factual_accuracy:.2f}")
    print(f"   Relevance: {qa_metrics.relevance_score:.2f}")
    print(f"   Completeness: {qa_metrics.completeness_score:.2f}")
    print(f"   LLM Judge Confidence: {qa_metrics.additional_metrics.get('llm_judge_confidence', 0):.2f}")


if __name__ == "__main__":
    main()
