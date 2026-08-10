#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skrypt do konwersji modelu do formatu Core ML z automatycznym tworzeniem podkatalogów na podstawie nazwy modelu.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import torch
import numpy as np
import coremltools as ct

# Wyłącz MPS i CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# Wymuś natychmiastowe wyświetlanie komunikatów
sys.stdout.reconfigure(line_buffering=True)
print("Skrypt uruchomiony - sprawdzam czy widzisz ten komunikat")

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info("Logger skonfigurowany")

# Ustawienie ścieżki projektu
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Importy
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    logger.error(f"Błąd importu: {e}")
    sys.exit(1)

# Token Hugging Face (opcjonalne)
try:
    from model_preparation.config.config import get_settings
    settings = get_settings()
    hf_token = settings.get_token("huggingface")
except ImportError:
    logger.warning("Module 'model_preparation' not found. Using default token.")
    hf_token = None

if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    from huggingface_hub import login
    login(token=hf_token)
    logger.info("Logged in to Hugging Face Hub")
else:
    logger.warning("No token found. Ensure model access without token.")

# Ustawienie CPU
default_device = torch.device("cpu")
logger.info("Używam wyłącznie CPU dla stabilności")

def export_optimized_model(model_id, output_path, context_size=1024, batch_size=1):
    """Export PyTorch model to Core ML w formacie float32."""
    logger.info(f"Loading model {model_id}...")
    
    device = default_device
    logger.info(f"Używam urządzenia: {device}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            token=hf_token,
        ).to(device)
        logger.info(f"Model załadowany, liczba parametrów: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} miliardów")
        
        logger.info("Preparing example inputs...")
        example_input = torch.zeros((batch_size, context_size), dtype=torch.long, device=device)
        
        class SimpleLlamaWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, input_ids):
                attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True
                )
                return outputs.logits
        
        wrapped_model = SimpleLlamaWrapper(model).to(device)
        logger.info("Tracing model...")
        traced_model = torch.jit.trace(wrapped_model, example_input)
        
        logger.info("Konwertuję model do Core ML w formacie float32...")
        mlmodel = ct.convert(
            model=traced_model,
            source="pytorch",
            inputs=[ct.TensorType(name="input_ids", shape=(batch_size, context_size), dtype=np.int32)],
            outputs=[ct.TensorType(name="logits", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
            skip_model_load=True
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Konwersja zakończona! Zapisuję model do {output_path}...")
        mlmodel.save(output_path)
        logger.info(f"Rozmiar modelu: {os.path.getsize(output_path) / (1024*1024*1024):.2f} GB")
        return mlmodel
    
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        raise

def quantize_model(model_path, output_path, quantize_type="int4"):
    """Quantize istniejący model Core ML do wybranego formatu."""
    logger.info(f"Loading model from {model_path}...")
    mlmodel = ct.models.MLModel(model_path)
    
    logger.info(f"Quantizing model to {quantize_type}...")
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=quantize_type,
        granularity="per_block",
        block_size=32,
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    try:
        mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)
        logger.info(f"Successfully quantized to {quantize_type}")
    except ValueError as e:
        logger.error(f"Error during quantization to {quantize_type}: {e}")
        raise
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving quantized model to {output_path}...")
    mlmodel.save(output_path)
    logger.info(f"Rozmiar modelu: {os.path.getsize(output_path) / (1024*1024*1024):.2f} GB")
    return mlmodel

def main():
    parser = argparse.ArgumentParser(description="Convert Llama model to Core ML with model-specific subdirectories")
    parser.add_argument("--model_id", eval_type=str, default="ContactDoctor/Bio-Medical-Llama-3-8B", 
                        help="Hugging Face model ID")
    parser.add_argument("--output_dir", eval_type=str, default="downloads", 
                        help="Base directory for Core ML models (relative to project root or absolute)")
    parser.add_argument("--context_size", eval_type=int, default=1024, 
                        help="Maximum context size")
    parser.add_argument("--quantize", action="store_true", 
                        help="Quantize model to specified type")
    parser.add_argument("--quantize_type", eval_type=str, default="int4",
                        help="Quantize model to specified type (int4, int8)")
    parser.add_argument("--test_mode", action="store_true", 
                        help="Use smaller model (TinyLlama) for testing")
    parser.add_argument("--input_model", eval_type=str, default=None, 
                        help="Path to existing .mlpackage model to quantize")
    
    args = parser.parse_args()
    
    # Ustal pełną ścieżkę output_dir
    if os.path.isabs(args.output_dir):
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.path.join(project_root, "models", args.output_dir)
    
    # Ustawienia modelu i ścieżek
    if args.test_mode and not args.input_model:
        logger.info("Używam mniejszego modelu TinyLlama do testów")
        args.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Pobierz skróconą nazwę modelu (ostatnia część ID)
    model_name = args.model_id.split("/")[-1]
    
    # Ścieżki oparte na nazwie modelu w output_dir
    optimized_path = os.path.join(base_output_dir, model_name, "optimized", f"{model_name}_optimized.mlpackage")
    quantized_path = os.path.join(base_output_dir, model_name, "quantized", args.quantize_type, f"{model_name}_{args.quantize_type}.mlpackage")
    
    # Konwersja lub ładowanie modelu
    if args.input_model:
        logger.info(f"Używam istniejącego modelu: {args.input_model}")
        model_path = args.input_model
    else:
        logger.info(f"Rozpoczynam konwersję modelu {args.model_id}")
        mlmodel = export_optimized_model(
            args.model_id,
            optimized_path,
            context_size=args.context_size
        )
        model_path = optimized_path
    
    # Kwantyzacja, jeśli wybrano
    if args.quantize:
        logger.info(f"Rozpoczynam kwantyzację modelu do {args.quantize_type}...")
        mlmodel_quantized = quantize_model(model_path, quantized_path, quantize_type=args.quantize_type)
        logger.info(f"Rozmiar oryginalnego modelu: {os.path.getsize(model_path) / (1024*1024*1024):.2f} GB")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()