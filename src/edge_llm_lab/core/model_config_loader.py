#!/usr/bin/env python3
"""
Model Configuration Loader
Ładuje konfigurację modeli dla różnych agentów z plików YAML
Pobiera dynamicznie metadane modeli z Ollamy
"""
import os
import yaml
import subprocess
import json
import glob
from typing import List, Dict, Any, Optional
import ollama
import sys
import dotenv
dotenv.load_dotenv()

# Initialize ollama client with custom host if provided
OLLAMA_HOST = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_HOST)

def delete_all_models():
    """Delete all models currently installed in Ollama.

    """

    try:
        # List all installed models
        models = client.list()['models']
        print(f"Models: {models}")
        if not models:
            print("No models to delete.")
            return
        
        print("Deleting the following models:")
        for model in models:
            model_name = model['model']
            print(f" - {model_name}")
            client.delete(model_name)
            print(f"Deleted {model_name}")
        
        print("All models deleted successfully.")
    except Exception as e:
        print(f"Error deleting models: {e}")

def delete_model(model_name):
    """Delete a specific model from Ollama."""
    try:
        print(f"🗑️ Deleting model: {model_name}")
        client.delete(model_name)
        print(f"✅ Deleted {model_name}")
        return True
    except Exception as e:
        print(f"⚠️ Error deleting model {model_name}: {e}")
        return False

def pull_model_with_progress(model_name):
    """Pull a model with progress display using streaming."""
    try:
        print(f"Pulling model: {model_name}")
        # Stream the pull process to show progress
        for progress in client.pull(model_name, stream=True):
            # Extract relevant progress information
            status = progress.get('status', '')
            completed = progress.get('completed', 0)
            total = progress.get('total', 0)
            
            # Calculate and display progress percentage if applicable
            if total > 0:
                percentage = (completed / total) * 100
                sys.stdout.write(f"\rStatus: {status} [{percentage:.2f}%]")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\rStatus: {status}")
                sys.stdout.flush()
        
        print("\nModel pulled successfully.")
    except Exception as e:
        print(f"\nError pulling model: {e}")


def load_models_for_agent(agent_type: str, base_path: str = None) -> List[str]:
    """
    Ładuje listę aktywnych modeli dla danego agenta z pliku config.yaml
    
    Args:
        agent_type (str): Typ agenta (constant_data, fluctuating_data, periodic_data, symptom)
        base_path (str): Ścieżka bazowa do folderów source (opcjonalne)
        
    Returns:
        List[str]: Lista nazw modeli do ewaluacji
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_file = os.path.abspath(os.path.join(base_path, "..", "..", "examples", "desktop", "input", "agents", agent_type, "evaluation_config", "config.yaml"))

    
    if not os.path.exists(config_file):
        print(f"⚠️ Config file not found: {config_file}")
        print(f"💡 Create config file with models_to_evaluate section")
        return []
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        models_config = config.get('models_to_evaluate', [])
        
        # Filtruj modele (wszystkie dict modele są aktywne, bo usunęliśmy enabled)
        active_models = []
        for model in models_config:
            if isinstance(model, dict):
                # Wszystkie modele dict są dostępne do testowania
                active_models.append(model['name'])
            elif isinstance(model, str):
                # Backward compatibility - jeśli to tylko string
                active_models.append(model)
        
        print(f"📋 Loaded {len(active_models)} active models for {agent_type}:")
        for model in active_models:
            print(f"  • {model}")
            
        return active_models
        
    except Exception as e:
        print(f"❌ Error loading config for {agent_type}: {e}")
        return []

def get_agent_config(agent_type: str, base_path: str = None) -> Dict[str, Any]:
    """
    Ładuje konfigurację agenta z pliku config.yaml
    
    Args:
        agent_type (str): Typ agenta
        base_path (str): Ścieżka bazowa do folderów source (opcjonalne)
        
    Returns:
        Dict[str, Any]: Konfiguracja agenta
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_file = os.path.abspath(os.path.join(base_path, "..", "..", "examples", "desktop", "input", "agents", agent_type, "evaluation_config", "config.yaml"))
    
    if not os.path.exists(config_file):
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('agent_config', {})
        
    except Exception as e:
        print(f"❌ Error loading agent config for {agent_type}: {e}")
        return {}

def load_stage_config(stage_id: int, agent_type: str = "constant_data_en") -> Dict[str, Any]:
    """Loads configuration for a specific evaluation stage."""
    from edge_llm_lab.core.config.unified_config import UnifiedConfig
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    u_config = UnifiedConfig(root)
    pipeline = u_config.load_pipeline_config()
    
    stage = next((s for s in pipeline["stages"] if s["id"] == stage_id), None)
    if not stage:
        return {}
        
    # Handle path override for different agents if necessary
    path = stage["config_path"]
    if agent_type != "constant_data_en" and "constant_data_en" in path:
         path = path.replace("constant_data_en", agent_type)
         
    return u_config.load_yaml(path)

def list_available_models_for_agent(agent_type: str, base_path: str = None) -> Dict[str, Any]:
    """
    Ładuje wszystkie modele (aktywne i nieaktywne) dla danego agenta
    
    Args:
        agent_type (str): Typ agenta
        base_path (str): Ścieżka bazowa do folderów source (opcjonalne)
        
    Returns:
        Dict[str, Any]: Słownik z wszystkimi modelami i ich statusem
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_file = os.path.abspath(os.path.join(base_path, "..", "..", "examples", "desktop", "input", "agents", agent_type, "evaluation_config", "config.yaml"))
    
    if not os.path.exists(config_file):
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        models_config = config.get('models_to_evaluate', [])
        
        models_info = {}
        for model in models_config:
            if isinstance(model, dict):
                model_name = model['name']
                models_info[model_name] = {
                    'description': model.get('description', ''),
                    'status': model.get('status', 'unknown'),
                    'notes': model.get('notes', ''),
                    'active': model.get('status') == 'active'
                }
        
        return models_info
        
    except Exception as e:
        print(f"❌ Error loading models info for {agent_type}: {e}")
        return {}

# def get_ollama_model_info(model_name: str) -> Optional[Dict[str, Any]]:
#     """
#     Pobiera informacje o modelu z Ollamy (rozmiar, parametry, etc.)
    
#     Args:
#         model_name (str): Nazwa modelu w Ollamie
        
#     Returns:
#         Optional[Dict[str, Any]]: Metadane modelu lub None jeśli błąd
#     """
#     try:
#         # Pobierz listę modeli z `ollama list --json`
#         result = subprocess.run(['ollama', 'list'], 
#                               capture_output=True, text=True, timeout=10)
        
#         if result.returncode != 0:
#             print(f"⚠️ Error running 'ollama list': {result.stderr}")
#             return None
            
#         # Parse text output (NAME, ID, SIZE, MODIFIED)
#         lines = result.stdout.strip().split('\n')
        
#         # Skip header line
#         for line in lines[1:]:
#             if not line.strip():
#                 continue
            
#             parts = line.split()
#             if len(parts) >= 5:  # NAME ID SIZE UNIT MODIFIED...
#                 name = parts[0]
#                 digest = parts[1]  # e.g., "1f64b4541957"
#                 size_str = parts[2]  # e.g., "5.1"
#                 unit = parts[3]  # e.g., "GB"
#                 modified = " ".join(parts[4:])  # e.g., "3 minutes ago"
                
#                 if name == model_name or name.startswith(model_name):
#                     # Parse size
#                     size_gb = 0
#                     size_bytes = 0
#                     try:
#                         size_value = float(size_str)
#                         if unit == 'GB':
#                             size_gb = size_value
#                             size_bytes = int(size_value * 1024**3)
#                         elif unit == 'MB':
#                             size_gb = size_value / 1024
#                             size_bytes = int(size_value * 1024**2)
#                         elif unit == 'KB':
#                             size_gb = size_value / (1024**2)
#                             size_bytes = int(size_value * 1024)
#                         elif unit == 'B':
#                             size_gb = size_value / (1024**3)
#                             size_bytes = int(size_value)
#                     except ValueError:
#                         pass
                    
#                     return {
#                         'name': name,
#                         'size_gb': size_gb,
#                         'size_bytes': size_bytes,
#                         'modified_at': modified,
#                         'digest': digest,
#                         'details': {}
#                     }
        
#         print(f"⚠️ Model '{model_name}' not found in Ollama")
#         return None
        
#     except subprocess.TimeoutExpired:
#         print(f"⚠️ Timeout getting model info for {model_name}")
#         return None
#     except Exception as e:
#         print(f"⚠️ Error getting model info for {model_name}: {e}")
#         return None

def mark_model_as_tested(agent_type: str, model_name: str, base_path: str = None) -> bool:
    """
    Oznacza model jako przetestowany (tested: true) w config.yaml
    
    Args:
        agent_type (str): Typ agenta
        model_name (str): Nazwa modelu do oznaczenia
        base_path (str): Ścieżka bazowa (opcjonalne)
        
    Returns:
        bool: True jeśli udało się zaktualizować, False w przeciwnym razie
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_file = os.path.abspath(os.path.join(base_path, "..", "..", "examples", "desktop", "input", "agents", agent_type, "evaluation_config", "config.yaml"))
    
    if not os.path.exists(config_file):
        print(f"⚠️ Config file not found: {config_file}")
        return False
    
    try:
        # Wczytaj config
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        models_config = config.get('models_to_evaluate', [])
        model_found = False
        
        # Znajdź model i oznacz jako tested: true
        for model in models_config:
            if isinstance(model, dict) and model.get('name') == model_name:
                model['tested'] = True
                model_found = True
                print(f"✅ Marked {model_name} as tested: true")
                break
        
        if not model_found:
            print(f"⚠️ Model {model_name} not found in config for {agent_type}")
            return False
        
        # Zapisz zaktualizowany config
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"💾 Updated config file: {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating config for {agent_type}: {e}")
        return False

def get_untested_models(agent_type: str, base_path: str = None) -> List[str]:
    """
    Pobiera listę modeli z tested: false
    
    Args:
        agent_type (str): Typ agenta
        base_path (str): Ścieżka bazowa (opcjonalne)
        
    Returns:
        List[str]: Lista nazw modeli do przetestowania
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_file = os.path.abspath(os.path.join(base_path, "..", "..", "examples", "desktop", "input", "agents", agent_type, "evaluation_config", "config.yaml"))
    
    if not os.path.exists(config_file):
        print(f"⚠️ Config file not found: {config_file}")
        return []
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        models_config = config.get('models_to_evaluate', [])
        untested_models = []
        
        for model in models_config:
            if isinstance(model, dict):
                # Sprawdź czy model ma tested: false (lub brak parametru tested)
                if not model.get('tested', False):
                    untested_models.append(model['name'])
        
        print(f"📋 Found {len(untested_models)} untested models for {agent_type}:")
        for model in untested_models:
            print(f"  • {model}")
            
        return untested_models
        
    except Exception as e:
        print(f"❌ Error loading untested models for {agent_type}: {e}")
        return []

def check_model_in_results(agent_type: str, model_name: str, base_path: str = None, evaluation_type: str = "unreferenced") -> bool:
    """
    Sprawdza czy model już ma zapisane wyniki w plikach JSON
    
    Args:
        agent_type (str): Typ agenta
        model_name (str): Nazwa modelu
        base_path (str): Ścieżka bazowa (opcjonalne)
        evaluation_type (str): Typ ewaluacji ("unreferenced" lub "referenced")
        
    Returns:
        bool: True jeśli model ma już wyniki, False w przeciwnym razie

    # >>> check_model_in_results("constant_data_en", "granite3.1-dense:2b-instruct-fp16", evaluation_type="unreferenced")
    # True
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Szukaj w odpowiednim folderze based on evaluation_type
    results_dir = os.path.abspath(os.path.join(base_path, "..", "..", "examples", "desktop", "output", "agents", agent_type, evaluation_type ))
    # print(f"Results dir: {results_dir}")
    
    if not os.path.exists(results_dir):
        return False
    
    # Znajdź pliki JSON z wynikami

    # Nowy centralny plik dla unreferenced
    if evaluation_type == "unreferenced":
        central_file = os.path.join(results_dir, f"partial_results_{agent_type}.json")
    else:
        central_file = os.path.join(results_dir, f"{evaluation_type}_evaluation_results.json")
    # /Users/mariamalycha/Documents/fed-mobile/thesis_generators/source/output/agents/constant_data_en/unreferenced
    # /Users/mariamalycha/Documents/fed-mobile/thesis_generators/source/output/agents/constant_data_en/unreferenced
    json_files = [central_file] if os.path.exists(central_file) else []
    # print(f"JSON files: {json_files}")

    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Sprawdź nową strukturę dla unreferenced (lista evaluations jak referenced)
            if evaluation_type == "referenced" and "evaluations" in results:
                for evaluation in results["evaluations"]:
                    model_info = evaluation.get("model_info", {})
                    if (model_info.get("name") == model_name or 
                        model_info.get("normalized_name") == model_name):
                        print(f"✅ Model {model_name} found in unreferenced evaluations: {os.path.basename(json_file)}")
                        return True
            
            # Sprawdź strukturę dla referenced (lista evaluations)
            if evaluation_type == "unreferenced":
                completed_models = results.get("completed_models", {})
                if model_name in completed_models:
                    print(f"✅ Model {model_name} found in unreferenced completed models: {os.path.basename(json_file)}")
                    return True
            
            # Sprawdź starą strukturę dla unreferenced (backwards compatibility)
            if evaluation_type == "unreferenced" and "models" in results:
                if model_name in results["models"]:
                    print(f"✅ Model {model_name} found in old unreferenced structure: {os.path.basename(json_file)}")
                    return True
            
            # Sprawdź starą strukturę (backwards compatibility)
            models_tested = results.get('models_tested', [])
            if model_name in models_tested:
                print(f"✅ Model {model_name} found in results: {os.path.basename(json_file)}")
                return True
                
            # Sprawdź też w all_results
            all_results = results.get('all_results', {})
            if model_name in all_results:
                print(f"✅ Model {model_name} found in all_results: {os.path.basename(json_file)}")
                return True
                
        except Exception as e:
            print(f"⚠️ Error reading {json_file}: {e}")
            continue
    
    return False

def get_truly_untested_models(agent_type: str, base_path: str = None, evaluation_type: str = "unreferenced") -> List[str]:
    """
    Pobiera listę modeli które mają tested: true ale NIE mają wyników w JSON
    (tzn. zostały oznaczone jako przetestowane ale wyniki się nie zapisały)
    
    Args:
        agent_type (str): Typ agenta
        base_path (str): Ścieżka bazowa (opcjonalne)
        evaluation_type (str): Typ ewaluacji ("unreferenced" lub "referenced")
        
    Returns:
        List[str]: Lista nazw modeli do przetestowania
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_file = os.path.abspath(os.path.join(base_path, "..", "..", "examples", "desktop", "input", "agents", agent_type, "evaluation_config", "config.yaml"))
    
    if not os.path.exists(config_file):
        print(f"⚠️ Config file not found: {config_file}")
        return []
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        models_config = config.get('models_to_evaluate', [])
        models_to_test = []
        
        # Sprawdź tylko modele z tested: true
        for model in models_config:
            if isinstance(model, dict):
                model_name = model['name']
                tested_flag = model.get('tested', False)
                
                if tested_flag:  # Tylko modele z tested: true
                    has_results = check_model_in_results(agent_type, model_name, base_path, evaluation_type)
                    
                    if not has_results:
                        # Model ma tested: true ale brak wyników - trzeba przetestować ponownie
                        models_to_test.append(model_name)
                        print(f"⚠️ Model {model_name} ma tested: true ale BRAK wyników w JSON - będzie przetestowany")
        
        print(f"\n📋 Found {len(models_to_test)} models with tested: true but missing JSON results for {agent_type}:")
        for model in models_to_test:
            print(f"  • {model} (tested: true but no JSON results)")
            
        return models_to_test
        
    except Exception as e:
        print(f"❌ Error loading models for {agent_type}: {e}")
        return []

# if __name__ == "__main__":
#     # Test funkcji
    # print("🧪 Testing model config loader...")
    
    # agents = ['constant_data', 'fluctuating_data', 'periodic_data', 'symptom']
    
    # for agent in agents:
    #     print(f"\n=== {agent.upper()} ===")
    #     models = load_models_for_agent(agent)
    #     config = get_agent_config(agent)
    #     print(f"Agent config: {config}")
        
    #     # Test metadata loading
    #     if models:
    #         print(f"\n🔍 Testing metadata for {agent}:")
    #         models_meta = load_models_with_metadata(agent)
    #         for model_meta in models_meta:
    #             print(f"  {model_meta['name']}: {model_meta['size_gb']:.1f}GB")
