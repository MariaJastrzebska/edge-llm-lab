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
from typing import List, Dict, Any, Optional

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
    
    config_file = os.path.join(base_path, "input", agent_type, "config.yaml")
    
    if not os.path.exists(config_file):
        print(f"⚠️ Config file not found: {config_file}")
        print(f"💡 Create config file with models_to_evaluate section")
        return []
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        models_config = config.get('models_to_evaluate', [])
        
        # Filtruj tylko aktywne modele
        active_models = []
        for model in models_config:
            if isinstance(model, dict):
                if model.get('status') == 'active':
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
    
    config_file = os.path.join(base_path, "input", agent_type, "config.yaml")
    
    if not os.path.exists(config_file):
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('agent_config', {})
        
    except Exception as e:
        print(f"❌ Error loading agent config for {agent_type}: {e}")
        return {}

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
    
    config_file = os.path.join(base_path, "input", agent_type, "config.yaml")
    
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

def get_ollama_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Pobiera informacje o modelu z Ollamy (rozmiar, parametry, etc.)
    
    Args:
        model_name (str): Nazwa modelu w Ollamie
        
    Returns:
        Optional[Dict[str, Any]]: Metadane modelu lub None jeśli błąd
    """
    try:
        # Pobierz listę modeli z `ollama list --json`
        result = subprocess.run(['ollama', 'list', '--json'], 
                              capture_output=True, text=True, timeout=2)
        
        if result.returncode != 0:
            print(f"⚠️ Error running 'ollama list': {result.stderr}")
            return None
            
        models_data = json.loads(result.stdout)
        
        # Znajdź nasz model
        for model in models_data.get('models', []):
            if model.get('name') == model_name or model.get('name', '').startswith(model_name):
                size_bytes = model.get('size', 0)
                size_gb = round(size_bytes / (1024**3), 1) if size_bytes > 0 else 0
                
                return {
                    'name': model.get('name'),
                    'size_gb': size_gb,
                    'size_bytes': size_bytes,
                    'modified_at': model.get('modified_at').isoformat() if model.get('modified_at') else 'Unknown',
                    'digest': model.get('digest'),
                    'details': model.get('details', {})
                }
        
        print(f"⚠️ Model '{model_name}' not found in Ollama")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"⚠️ Timeout getting model info for {model_name}")
        return None
    except Exception as e:
        print(f"⚠️ Error getting model info for {model_name}: {e}")
        return None

def load_models_with_metadata(agent_type: str, base_path: str = None) -> List[Dict[str, Any]]:
    """
    Ładuje modele z metadanymi z Ollamy
    
    Args:
        agent_type (str): Typ agenta
        base_path (str): Ścieżka bazowa (opcjonalne)
        
    Returns:
        List[Dict[str, Any]]: Lista modeli z metadanymi
    """
    models = load_models_for_agent(agent_type, base_path)
    models_with_meta = []
    
    print(f"🔍 Fetching metadata for {len(models)} models...")
    
    for model_name in models:
        ollama_info = get_ollama_model_info(model_name)
        
        model_data = {
            'name': model_name,
            'size_gb': ollama_info.get('size_gb', 0) if ollama_info else 0,
            'available': ollama_info is not None,
            'metadata': ollama_info or {}
        }
        
        models_with_meta.append(model_data)
        
        status = "✅" if ollama_info else "❌"
        size_info = f"({ollama_info.get('size_gb', 0):.1f}GB)" if ollama_info else "(not found)"
        print(f"  {status} {model_name} {size_info}")
    
    return models_with_meta

if __name__ == "__main__":
    # Test funkcji
    print("🧪 Testing model config loader...")
    
    agents = ['constant_data', 'fluctuating_data', 'periodic_data', 'symptom']
    
    for agent in agents:
        print(f"\n=== {agent.upper()} ===")
        models = load_models_for_agent(agent)
        config = get_agent_config(agent)
        print(f"Agent config: {config}")
        
        # Test metadata loading
        if models:
            print(f"\n🔍 Testing metadata for {agent}:")
            models_meta = load_models_with_metadata(agent)
            for model_meta in models_meta:
                print(f"  {model_meta['name']}: {model_meta['size_gb']:.1f}GB")
