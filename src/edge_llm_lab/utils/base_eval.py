import pandas as pd
import time
import csv
import sys
import os

# Add project root and src to path for direct script execution
# This allows 'from edge_llm_lab...' to work regardless of where the script is run from
try:
    _file_path = os.path.abspath(__file__)
    # src/edge_llm_lab/utils/base_eval.py -> ../../.. gets to project root
    _project_root = os.path.abspath(os.path.join(os.path.dirname(_file_path), "../../../"))
    _src_path = os.path.join(_project_root, "src")
    if _src_path not in sys.path:
        sys.path.insert(0, _src_path)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    pass

import yaml
import requests
import atexit
import hashlib
import json
import subprocess
from enum import Enum
from typing import List, Dict, Any, Union, Optional, Literal, Tuple
from pydantic import BaseModel
import ollama
import openai
import instructor
from joblib import Memory
from edge_llm_lab.utils.neptune_utils import NeptuneManager
from datetime import datetime
from typing import Protocol, runtime_checkable

@runtime_checkable
class RunTracker(Protocol):
    def init_run(self, name: str, tags: List[str] = None, params: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> bool: ...
    def log_round_metrics(self, round_num: int, metrics: Dict[str, Any], latency: Dict[str, Any] = None): ...
    def upload_artifact(self, local_path: str, neptune_path: str): ...
    def upload_directory_artifacts(self, local_dir: str, neptune_path_prefix: str, extensions=(".png", ".jpg", ".jpeg", ".json")): ...
    def stop(self): ...



class Agent(Enum):
    CONSTANT_DATA = 'constant_data'
    CONSTANT_DATA_EN = 'constant_data_en'
    FLUCTUATING_DATA = 'fluctuating_data'
    PERIODIC_DATA = 'periodic_data'
    SYMPTOM = 'symptom'
    SYMPTOM_EN = 'symptom_en'
    FLUCTUATING_DATA_EN = 'fluctuating_data_en'
    PERIODIC_DATA_EN = 'periodic_data_en'


class BaseEvaluation:
    """Base class with common evaluation functionality."""

    def __init__(self, model_name, agent, eval_type="default"):
        """Initialize BaseEvaluation with model name, agent, and optional source path.
        Checks if model metadata is cached and sets parameters for referenced and unreferenced evaluations.
        
        # >>> from base_eval import Agent
        # >>> be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN)
        >>> from base_eval import BaseEvaluation
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
        >>> with redirect_stdout(io.StringIO()):
        ...    be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN)
        """
               
        self.model_name = model_name
        self.model_name_norm = self.model_name.replace(':', '_').replace('/', '_')
        self.folders_to_cleanup = []
        self.agent_enum = self.ensure_agent_enum(agent)
        self.agent_type = self.agent_enum.value
        self.agent = self.agent_enum
        self.eval_type = eval_type# Domyślna wartość, do nadpisania w klasach potomnych
        

        self.BASE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.SOURCE_PATH = os.path.abspath(os.path.join(self.BASE_PATH, "../../../examples/desktop"))
        if not os.path.exists(self.SOURCE_PATH):
            os.makedirs(self.SOURCE_PATH, exist_ok=True)

        # Path relative to project root (3 levels up from this file)
        self.APP_PARAMS_PATH = os.path.abspath(os.path.join(self.BASE_PATH, "../../../examples/desktop/config/inference_params.yaml"))

        

        # Add examples/desktop to sys.path to allow importing pydantic_models
        desktop_path = os.path.abspath(os.path.join(self.BASE_PATH, "../../../examples/desktop"))
        if desktop_path not in sys.path:
            sys.path.append(desktop_path)

        if self.agent_type[:-2] == 'en':
            from pydantic_models.pydantic_models_en import (
                ConstantDataAnalysisCOT,
                FluctuatingDataAnalysisCOT,
                PeriodicDataAnalysisCOT,
                SymptomAnalysisCOT,
            )
        else:
            from pydantic_models.pydantic_models import (
                ConstantDataAnalysisCOT,
                FluctuatingDataAnalysisCOT,
                PeriodicDataAnalysisCOT,
                SymptomAnalysisCOT,
            )
        self.TOOLS_DICT = {
            'constant_data': {"pydantic_model": ConstantDataAnalysisCOT, "description": "Send constant medical data", "name": "send_medical_data"},
            'fluctuating_data': {"pydantic_model": FluctuatingDataAnalysisCOT, "description": "Send fluctuating medical data", "name": "send_medical_data"},
            'periodic_data': {"pydantic_model": PeriodicDataAnalysisCOT, "description": "Send periodic medical data", "name": "send_medical_data"},
            'symptom': {"pydantic_model": SymptomAnalysisCOT, "description": "Send symptom data", "name": "send_medical_data"},
            'constant_data_en': {"pydantic_model": ConstantDataAnalysisCOT, "description": "Send constant medical data", "name": "send_medical_data_en"},
            'fluctuating_data_en': {"pydantic_model": FluctuatingDataAnalysisCOT, "description": "Send fluctuating medical data", "name": "send_medical_data_en"},
            'periodic_data_en': {"pydantic_model": PeriodicDataAnalysisCOT, "description": "Send periodic medical data", "name": "send_medical_data_en"},
            'symptom_en': {"pydantic_model": SymptomAnalysisCOT, "description": "Send symptom data", "name": "send_medical_data_en"},
        }

        self.agent_map = {
            "1": Agent.CONSTANT_DATA,
            "2": Agent.CONSTANT_DATA_EN,
            "3": Agent.FLUCTUATING_DATA,
            "4": Agent.PERIODIC_DATA,
            "5": Agent.SYMPTOM,
            "6": Agent.SYMPTOM_EN,
            "7": Agent.FLUCTUATING_DATA_EN,
            "8": Agent.PERIODIC_DATA_EN
        }
        # Ścieżki w nowej strukturze source/

        # self.MODEL_CACHE_FILE = os.path.join(self.SOURCE_PATH,
        #     "output", "model_cache", "model_metadata.json")




        # --- Configuration ---
        self.evaluator_model_name = 'gpt-4o-mini'
        self.evaluator_api_key = os.getenv("OPENAI_API_KEY")
        self.OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.OLLAMA_CLIENT = ollama.Client(host=self.OLLAMA_HOST)
        self.OPENAI_CLIENT = openai.OpenAI(api_key=self.evaluator_api_key)
        self.PATCHED_OPENAI_CLIENT = instructor.patch(self.OPENAI_CLIENT, mode=instructor.Mode.MD_JSON)
        # Initialize llama server process and paths
        self.llama_server_process = None
        self.LLAMA_CPP_PATH = os.path.abspath(os.path.join(self.BASE_PATH, "../../../examples/desktop/llama.cpp"))
        self.LLAMA_SERVER_PATH = os.path.join(
            self.LLAMA_CPP_PATH, "build/bin/llama-server")
        self.LLAMA_SERVER_HOST = "127.0.0.1"
        self.LLAMA_SERVER_PORT = 8080
        self.LLAMA_SERVER_URL = f"http://{self.LLAMA_SERVER_HOST}:{self.LLAMA_SERVER_PORT}"
        
        # # Create instructor-patched client for llama-server (structured output support)
        # self.PATCHED_LLAMA_CLIENT = instructor.from_provider(
        #     "openai/gpt-4",  # Model name (will be overridden in actual calls)
        #     base_url=f"{self.LLAMA_SERVER_URL}/v1",
        #     api_key="dummy"  # llama-server doesn't need API key
        # )
        # Load timeout from config.yaml agent_config, default to 60s
        from edge_llm_lab.core.model_config_loader import get_agent_config
        agent_config = get_agent_config(self.agent_type)
        self.TIMEOUT = agent_config.get('timeout_sec', 60)
        self.MULTI_TURN_GLOBAL_CONFIG = self._load_referenced_params()
        self.MAX_TOKENS = self.MULTI_TURN_GLOBAL_CONFIG.get('max_tokens')
        self.TOP_P = self.MULTI_TURN_GLOBAL_CONFIG.get('top_p')
        self.TEMPERATURE = self.MULTI_TURN_GLOBAL_CONFIG.get('temperature')
        self.CONTEXT_SIZE = self.MULTI_TURN_GLOBAL_CONFIG.get('context_size')
        self.SEED = 42
        self.SEED = 42
        self.current_model_metadata, self.all_model_metadata = self.save_and_get_current_model_metadata()

        # Pluggable Tracking (Default to Neptune if available, else Mock/Console)
        self.neptune: Optional[RunTracker] = NeptuneManager()
        self.inference_engine = "llama-server"
    def _finalize_init(self, model_name, agent, eval_type):
        # check if model metadata cashed
        if self.eval_type in ("referenced", "unreferenced"):
            self.EVALUATION_PROMPT_PATH = os.path.join(self.SOURCE_PATH, "input","agents", self.agent_type, "evaluation_prompt", "gpt_judge_prompt.txt")
            self.PATIENT_SIMULATION_PROMPT_PATH = os.path.join(self.SOURCE_PATH, "input","agents", self.agent_type, "conversation_stimulation","patient_simulation_prompt.txt")
            
            
            
            
            
    def __del__(self):
        """Destruktor - czyści puste foldery na koniec, niezależnie od błędów"""
        self._cleanup_empty_folders()
    
    def _cleanup_empty_folders(self):
        """Czyści puste foldery z listy folders_to_cleanup oraz ich pustych rodziców"""
        for folder_path in self.folders_to_cleanup:
            current_path = os.path.abspath(folder_path)
            # Walk up the tree and remove empty folders
            while current_path and current_path.startswith(self.SOURCE_PATH) and current_path != self.SOURCE_PATH:
                try:
                    if os.path.exists(current_path) and os.path.isdir(current_path) and not os.listdir(current_path):
                        os.rmdir(current_path)
                        print(f"🗑️ Removed empty folder: {current_path}")
                        current_path = os.path.dirname(current_path)
                    else:
                        break
                except Exception as e:
                    print(f"⚠️ Could not remove folder {current_path}: {e}")
                    break
        
        
        
    def create_session(self, model_name=None, fixed_run_number: int = None, fixed_timestamp: str = None, create_all_models: bool = True) -> dict:
        """
        Tworzy sesję ewaluacji.
       return: LOG_FILE, MODEL_RUN, ALL_MODELS_RUN, TIMESTAMP
       >>> import io, sys
       >>> from contextlib import redirect_stdout
       >>> from base_eval import BaseEvaluation
       >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
       >>> with redirect_stdout(io.StringIO()):
       ...     be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN)
       ...     be.create_session() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
       ...     del be # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
       """
        if model_name is None:
            model_name_norm = self.model_name_norm
        else:
            model_name_norm = model_name.replace(":", "_").replace("/", "_")
        main_path_map = self.construct_main_paths_for_eval(model_name_norm)
        print(f"DEBUG: create_session called with create_all_models={create_all_models}")
        run_path_map = self._create_session_folders_and_log_file(main_path_map, self.SOURCE_PATH, fixed_run_number=fixed_run_number, fixed_timestamp=fixed_timestamp, create_all_models=create_all_models)
        print("run_path_map: ", run_path_map)
        
        log_folder = run_path_map["log_folder"]
        model_run_folder = run_path_map["model_run_folder"]
        all_models_run_folder = run_path_map["all_models_run_folder"]
        timestamp = run_path_map["timestamp"]
        model_run = run_path_map["model_run"]
        all_models_run = run_path_map["all_models_run"]
        print("Session folders and log file created")
        print(f"Model run: {model_run_folder}")
        print(f"All models run: {all_models_run_folder}")
        print(f"Timestamp: {timestamp}")
        self.folders_to_cleanup.append(model_run_folder)
        self.folders_to_cleanup.append(all_models_run_folder)
        self.folders_to_cleanup.append(log_folder)
        return {"model_run":model_run, "all_models_run":all_models_run, "log_folder": log_folder, "model_run_folder": model_run_folder, "all_models_run_folder": all_models_run_folder, "timestamp": timestamp, "current_model_metadata": self.current_model_metadata, "all_model_metadata": self.all_model_metadata}
    
    def get_or_create_file_or_folder(self, file_name:str, type_of_file: Literal["reference", "log", "cache", "metadata"], source_path:str=None)->Tuple[str, bool]:
        """
        Tworzy plik referencyjny w katalogu referencyjnym
        zwraca ścieżkę do pliku i boolean czy plik już istnial

        >>> from base_eval import BaseEvaluation
        >>> import tempfile, os
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
        >>> with redirect_stdout(io.StringIO()):
        ...     be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN) # doctest: +ELLIPSIS
        ...     with tempfile.TemporaryDirectory() as d:
        ...         ref_path, _ = be.get_or_create_file_or_folder(file_name="test", type_of_file="reference", source_path=d)
        ...         cache_path, _ = be.get_or_create_file_or_folder(file_name="test", type_of_file="cache", source_path=d)
        ...         log_path, _ = be.get_or_create_file_or_folder(file_name="test", type_of_file="log", source_path=d)
        >>> print(ref_path.endswith("reference/test.json"), cache_path.endswith("gpt-4o"), log_path.endswith("log/test.json"))
        True True True
        """
        if source_path is None:
            source_path = self.SOURCE_PATH
        else:
            source_path = source_path
        if type_of_file == "cache":
            folder = os.path.join(source_path, "output","agents", self.agent_type, self.eval_type,"prompt_cache", self.evaluator_model_name, self.model_name_norm)
        elif type_of_file== "reference":
            folder = os.path.join(source_path, "output","agents", self.agent_type, self.eval_type,"reference")
        elif type_of_file== "log":
            folder = os.path.join(source_path, "output","agents", self.agent_type, self.eval_type,"log")
        elif type_of_file== "metadata":
            folder = os.path.join(source_path, "output", "metadata")

        if type_of_file == "cache":
            folder_existed = os.path.exists(folder) 
            if not folder_existed:
                self.folders_to_cleanup.append(folder)

            os.makedirs(folder, exist_ok=True)
            return folder, folder_existed
        

        file = os.path.join(folder, file_name + ".json")
        if not os.path.exists(file):
            folder_existed = os.path.exists(folder)
            if not folder_existed:
                self.folders_to_cleanup.append(folder)
            os.makedirs(folder, exist_ok=True)
            with open(file, 'w') as f:
                # Initialize with empty evaluations list if it's a log file, otherwise empty dict
                if type_of_file == "log":
                    json.dump({"evaluations": []}, f)
                else:
                    json.dump({}, f)
            return file, False
        else:
            print(f"Plik referencyjny już istnieje: {file}")
            return file, True




        

    @staticmethod
    def load_json_file(file_path):
        """Load JSON file, return None if not exists."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Error loading evaluation log: {file_path}")
        return None

    @staticmethod
    def save_json_file(data, file_path, evaluations=True):
        """Save data to JSON file. New data will be overwritten.
        Validates data before saving to avoid empty structures.
        """
        if evaluations:
            if isinstance(data, str):
                # with open(file_path, 'w', encoding='utf-8') as f:
                #     f.write(data)
                data = BaseEvaluation.pretty_json(data, 'json')

            elif isinstance(data, dict):
                # Walidacja: usuń puste rounds z evaluations
                if "evaluations" in data:
                    cleaned_evaluations = []
                    for evaluation in data["evaluations"]:
                        if evaluation.get("rounds") and len(evaluation["rounds"]) > 0:
                            cleaned_evaluations.append(evaluation)
                        else:
                            print(f"⚠️ Skipping evaluation with empty rounds: {evaluation.get('model_info', {}).get('name', 'unknown')}")
                    data["evaluations"] = cleaned_evaluations
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"Saved data to {file_path}")
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Saved data to {file_path}")

                
    @staticmethod
    def read_txt( path):
        if isinstance(path, str):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
        

    def save_evaluation_log(self, log_data, filename):
        """@todo poprawka:Zapisuje dane do pliku logów w trybie append."""
        # Sprawdź czy plik istnieje i załaduj istniejące dane
        log_file, existed = self.get_or_create_file_or_folder(type_of_file="log", filename=filename)
        if existed:
            existing_data = self.load_json_file(log_file) 
        else:
            existing_data = {"evaluations": []}

        # Dodaj nowe dane do istniejących (append)
        if "evaluations" not in existing_data:
            existing_data["evaluations"] = []

        # Dodaj nowe ewaluacje do listy
        if "evaluations" in log_data:
            existing_data["evaluations"].extend(log_data["evaluations"])

        self.save_json_file(existing_data, log_file, append=False)
        print(
            f"✅ Evaluation data appended to: {log_file} (total evaluations: {len(existing_data['evaluations'])})")


    @staticmethod
    def load_yaml_config(config_path):
        """Load YAML configuration file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
  
    
    def create_tool_for_agent(self):
        return self._create_tool(self.TOOLS_DICT[self.agent_type]['pydantic_model'], self.TOOLS_DICT[self.agent_type]['name'], self.TOOLS_DICT[self.agent_type]['description'])

    @staticmethod
    def _create_tool(pydantic_model, name:str, description:str):
        """
        >>> from base_eval import BaseEvaluation
        >>> from pydantic import BaseModel
        >>> from pydantic import Field
        >>> from typing import List
        >>> class Step(BaseModel):
        ...     step: str
        ...     answer: str
        >>> class CoTReasoning(BaseModel):
        ...     steps: list[Step]
        ...     final_answer: str
        >>> tools_schema = BaseEvaluation._create_tool(CoTReasoning, "CoTReasoning", "reasoining about user guestion")
        >>> print("tools_schema", tools_schema)
        tools_schema [{'type': 'function', 'function': {'name': 'CoTReasoning', 'description': 'reasoining about user guestion', 'parameters': {'$defs': {'Step': {'properties': {'step': {'title': 'Step', 'type': 'string'}, 'answer': {'title': 'Answer', 'type': 'string'}}, 'required': ['step', 'answer'], 'title': 'Step', 'type': 'object'}}, 'properties': {'steps': {'items': {'$ref': '#/$defs/Step'}, 'title': 'Steps', 'type': 'array'}, 'final_answer': {'title': 'Final Answer', 'type': 'string'}}, 'required': ['steps', 'final_answer'], 'title': 'CoTReasoning', 'type': 'object'}}}]
        """
        tool_schema = pydantic_model.model_json_schema()
        # Define a generic tool schema that models can use
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": tool_schema
                }
            }
        ]

    @staticmethod
    def pretty_json(obj, output_type:Literal['json', 'str']='json'):
        """Format JSON object with proper indentation.   
        eg

        >>> from base_eval import BaseEvaluation, Agent
        >>> from base_eval import BaseEvaluation
        >>> response = {"tool_calls": [{"name": "test", "id": "123"}]}
        >>> result = BaseEvaluation.pretty_json(response, 'str') # doctest: +NORMALIZE_WHITESPACE
        >>> print(result)
        {
          "tool_calls": [
            {
              "id": "123",
              "name": "test"
            }
          ]
        }
        """

        if isinstance(obj, str):
            try:
                json_obj = json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, treat as plain text for 'str' output or return as is
                if output_type == 'str':
                    return obj
                return obj
        elif isinstance(obj, dict):
            json_obj = obj
        elif isinstance(obj, BaseModel):
            json_obj = obj.model_dump_json(indent=2)
        else:
            json_obj = str(obj)
        if output_type == 'json':
            return json_obj
        elif output_type == 'str':
            return json.dumps(json_obj, indent=2, ensure_ascii=False, sort_keys=True)
        

    # def init_cache(self, file_name, source_path:str=None):

    #     if source_path is None:
    #         source_path = self.SOURCE_PATH
    #     else:
    #         source_path = source_path
    #     cache_file, _ = self.get_or_create_file_or_folder(file_name=file_name, type_of_file="cache", source_path=source_path)
    #     return Memory(cache_file, verbose=0)

    
    def get_prompt_hash(self, data: Dict) -> str:
        """Generuje unikalny hash na podstawie parametrów zapytania."""
        input_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()

    # def get_cache_dir(self, filename: str = "cache", source_path: str = None) -> str:
    #     """Zwraca ścieżkę do folderu cache."""
    #     source_path = source_path or self.SOURCE_PATH
    #     cache_dir = os.path.join(source_path, self.agent_type, "prompt_cache", self.evaluator_model_name, filename)
    #     os.makedirs(cache_dir, exist_ok=True)
    #     return cache_dir

    def get_openai_response(
        self,
        messages: List[Dict],
        custom_catch_type: str = None,
        tools_schema: List = None,
        lazy_cache: bool = False,
        use_cache: bool = True,
        response_model: Optional[BaseModel] = None,
        custom_catch_ou: Optional[Dict] = None,
        response_format: Literal['str', 'json'] = 'str'
    ) -> Union[str, Dict, tuple[Union[str, Dict], str]]:
        """
        Pobiera odpowiedź z API OpenAI z opcją cache'owania za pomocą joblib.

        Args:
            messages: Lista wiadomości do API.
            custom_catch_type: Nazwa folderu cache.
            tools_schema: Schemat narzędzi dla API.
            lazy_cache: Zwraca krotkę (odpowiedź, klucz cache).
            use_cache: Czy używać cache.
            response_model: Model Pydantic dla odpowiedzi.
            custom_catch_ou: Dodatkowe dane do zapisu w cache.
            response_format: Format odpowiedzi ('str' lub 'dict').

        Returns:
            Odpowiedź z API lub cache, opcjonalnie z kluczem cache.
        >>> from unittest.mock import MagicMock
        >>> import tempfile
        >>> from base_eval import BaseEvaluation, Agent
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
        >>> from pydantic import Field
        >>> from typing import List
        >>> from pydantic import BaseModel
        >>> class Step(BaseModel):
        ...     step: str
        ...     answer: str
        >>> class CoTReasoning(BaseModel):
        ...     steps: list[Step]
        ...     final_answer: str
        >>> tools_schema = BaseEvaluation._create_tool(CoTReasoning, "CoTReasoning", "reasoining about user guestion")
        >>> with redirect_stdout(io.StringIO()):
        ...     be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN) 
        >>> response = be.get_openai_response(
        ...         messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}], 
        ...         tools_schema=tools_schema, 
        ...         custom_catch_type="test",
        ...         response_model=CoTReasoning,
        ...         response_format='str', 
        ...         use_cache=True)
        💰 Używam cache w 'examples/desktop/output/agents/constant_data_en/default/prompt_cache/gpt-4o'
        >>> print(response) # doctest: +NORMALIZE_WHITESPACE
        {
          "arguments": {
            "final_answer": "Paris",
            "steps": [
              {
                "answer": "The country is France.",
                "step": "Identify the country in question."
              },
              {
                "answer": "The capital city of France is Paris.",
                "step": "Determine the capital city of France."
              }
            ]
          },
          "id": "1",
          "name": "GetCapitalOfFrance"
        }  
        """
        # Generowanie klucza cache (tylko dane JSON-serializowalne)
        cache_key_data = {
            "model_name": self.evaluator_model_name,
            "model_tested": self.model_name if custom_catch_type != "reference_conversation" else "reference",
            "tryb": custom_catch_type if custom_catch_type else None,
            "messages": messages if messages else None,
            "tools_schema": tools_schema if tools_schema else None,
            "response_model_name": response_model.__name__ if response_model else None,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "max_tokens": self.MAX_TOKENS,
            "seed": self.SEED,
            "custom_catch_ou": custom_catch_ou if custom_catch_ou else None,
            "response_format": response_format if response_format else None
        }
        cache_key = self.get_prompt_hash(cache_key_data)
        # Dane dla funkcji call_openai (bez obiektów nie-picklowalnych)
        cache_data = cache_key_data.copy()  # Tylko dane serializowalne
        # Inicjalizacja cache z joblib
        
        cache_dir, _ = self.get_or_create_file_or_folder(file_name=custom_catch_type or "cache", type_of_file="cache")
        memory = Memory(cache_dir, verbose=0)

        # Funkcja do wywołania API OpenAI
        def call_openai(cache_data_param):
            # Rozpakowujemy dane z cache_data_param
            messages_param = cache_data_param["messages"]
            tools_param = cache_data_param["tools_schema"]
            response_model_param = response_model  # Używamy oryginalnego obiektu z closure
            response_format_param = cache_data_param.get("response_format", "str")
            class ToolCall(BaseModel):
                name: str
                arguments: response_model_param
                id: str

            if response_model_param or tools_param:
                client =  self.PATCHED_OPENAI_CLIENT
            else:
                client = self.OPENAI_CLIENT

            response = client.chat.completions.create(
                model=cache_data_param["model_name"],
                messages=messages_param,
                tools=tools_param if tools_param else None,
                tool_choice="auto" if tools_param else None,
                response_model=ToolCall if response_model_param else None,
                temperature=cache_data_param["temperature"],
                max_tokens=cache_data_param["max_tokens"],
                top_p=cache_data_param["top_p"],
                seed=cache_data_param["seed"]
            )
            if hasattr(response, 'choices'):  # Check for ChatCompletion first
                raw_content = response.choices[0].message.content
            elif hasattr(response, 'model_dump_json'):  # Fallback for Pydantic model
                raw_content = response.model_dump_json()
            else:
                raw_content = str(response)

            return {"content": raw_content, "format": response_format_param}

        # Cache'owana wersja funkcji
        cached_call = memory.cache(call_openai)
        # Wykonanie zapytania
        try:
            if use_cache:
                # Sprawdzamy czy cache istnieje (tylko jeśli mamy cache'owaną funkcję)
                if hasattr(cached_call, 'check_call_in_cache'):
                    cache_exists = cached_call.check_call_in_cache(cache_data)
                    if cache_exists:
                        print(f"💰 Używam cache w '{cache_dir}'")
                    else:
                        print(f"🔄 Cache miss - wysyłam nowe zapytanie do API i zapisuję w '{cache_dir}'")
                else:
                    print(f"💰 Używam cache w '{cache_dir}'")
                
                result = cached_call(cache_data)
            else:
                print("Pomijam cache, wysyłam nowe zapytanie do API")
                result = call_openai(cache_data)
            if isinstance(result, str) and "content" in result:
                content = result["content"]
            elif isinstance(result, dict) and "content" in result:
                content = result["content"]  # Extract content from dict
                # Only try to parse JSON if format is 'json', not 'str'
                
                content = self.pretty_json(content, response_format)

            else:
                content = result  # Use result directly (string or dict)
            # Dodanie custom_catch_ou do wyniku, jeśli potrzebne
            if custom_catch_ou:
                result = {"response": content, **custom_catch_ou}
            return (content, cache_key) if lazy_cache else content

        except Exception as e:
            print(f"❌ Błąd API OpenAI: {e}")
            print(f"❌ Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            return None  # Return None instead of error string




    
    def _load_referenced_params(self):
        """Load parameters specific to referenced evaluation for multi tunr trategy or agent type.
        >>> from base_eval import Agent
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
        >>> with redirect_stdout(io.StringIO()):
        ...     be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN) # doctest: +ELLIPSIS
        ...     multi_turn_parameters = be._load_referenced_params()
        >>> import pprint
        >>> pprint.pprint(multi_turn_parameters, width=72, sort_dicts=True)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'context_size': 7000,
        'cot_prompt_path': 'examples/desktop/prompts/constant_data_en.txt',
        'cot_schema_path': 'examples/desktop/schemas-outdated/constantdataanalysiscot_en_schema.json',
        'display_prompt': 'examples/desktop/prompts/display_prompt.txt',
        'max_tokens': ...,
        'temperature': ...,
        'top_p': ...,
        'validation_cot_prompt_path': 'examples/desktop/prompts/validation.txt',
        'validation_schema_path': 'examples/desktop/schemas-outdated/constantdata_en_schema.json'}
        """
        path = self.SOURCE_PATH
        params = self.load_yaml_config(self.APP_PARAMS_PATH)
        agents_section = params.get('agents', []) or []

        # Map self.agent_type like "constant_data" or "constant_data_en" to "constant_data_agent"

        mapped_key =  self.agent_type

        # agents_section is a list of dicts; find the matching agent by key
        agent_cfg = next((a for a in agents_section if a.get('key') == mapped_key), {})
        strategies = agent_cfg.get('strategies', {}) or {}
        multi_turn = strategies.get('multi_turn', {}) or {}

        # Exact expected keys (paths + numeric), no extras
        return {
            'cot_prompt_path': os.path.join(path, agent_cfg.get('cot_prompt_path')),
            'validation_cot_prompt_path': os.path.join(path, agent_cfg.get('validation_cot_prompt_path')),
            'display_prompt': os.path.join(path, agent_cfg.get('display_prompt')),
            'validation_schema_path': os.path.join(path, agent_cfg.get('validation_schema_path')),
            'cot_schema_path': os.path.join(path, agent_cfg.get('cot_schema_path')),
            'context_size': multi_turn.get('context_size'),
            'max_tokens': multi_turn.get('max_tokens'),
            'temperature': multi_turn.get('temperature'),
            'top_p': multi_turn.get('top_p')
        }

    @staticmethod
    def get_agent_value(agent):
        """Helper function to get string value from Agent enum or string 
        eg. 
        >>> from base_eval import BaseEvaluation, Agent
        >>> BaseEvaluation.get_agent_value(Agent.CONSTANT_DATA_EN)
        'constant_data_en'

        >>> BaseEvaluation.get_agent_value("constant_data_en")
        'constant_data_en'

        >>> BaseEvaluation.get_agent_value(1)
        '1'
        """
        if isinstance(agent, Agent):
            return agent.value
        elif isinstance(agent, str):
            return agent
        else:
            return str(agent)



    
    @staticmethod
    def select_agent():
        """Wybór agenta z menu 1-5"""
        print("\n📋 Wybierz typ agenta do testowania:")
        print("1. CONSTANT_DATA - zbieranie stałych danych medycznych (PL)")
        print("2. CONSTANT_DATA_EN - zbieranie stałych danych medycznych (EN)")
        print("3. FLUCTUATING_DATA - dane zmienne w czasie")
        print("4. PERIODIC_DATA - dane okresowe")
        print("5. SYMPTOM - zbieranie objawów")

        agent_choice = input("Wybór (1-5): ").strip() or "2"
        
        agent_map = {
            "1": Agent.CONSTANT_DATA,
            "2": Agent.CONSTANT_DATA_EN,
            "3": Agent.FLUCTUATING_DATA,
            "4": Agent.PERIODIC_DATA,
            "5": Agent.SYMPTOM
        }

        if agent_choice not in agent_map:
            print(" ❌ Nieprawidłowy wybór agenta! Używam domyślny CONSTANT_DATA_EN.")
            return Agent.CONSTANT_DATA_EN

        selected_agent = agent_map[agent_choice]
        print(f"✅ Wybrany agent: {selected_agent.value}")
        return selected_agent

    @staticmethod
    def get_random_avilable_model_for_doctest():
        available_models = ollama.list()
        return available_models.models[0].model

    @staticmethod
    def check_model_availability(model_name, install_choice=None):
        """Sprawdź dostępność modelu i zainstaluj jeśli potrzeba"""
        import ollama
        from edge_llm_lab.core.model_config_loader import delete_all_models, pull_model_with_progress
        
        try:
            available_models = ollama.list()
            available_model_names = []
            
            if hasattr(available_models, 'models'):
                for model in available_models.models:
                    name = model.model if hasattr(model, 'model') else str(model)
                    available_model_names.append(name)
            elif isinstance(available_models, dict) and 'models' in available_models:
                for model in available_models['models']:
                    if isinstance(model, dict):
                        name = model.get('model', model.get('name', str(model)))
                    else:
                        name = str(model)
                    available_model_names.append(name)
        except Exception as e:
            print(f" ❌ Błąd sprawdzania modeli: {e}")
            return False
            
        if model_name in available_model_names:
            print(f" ✅ Model {model_name} już jest zainstalowany!")
            return True
            
        # Model nie jest zainstalowany
        print(f" ❌ Model {model_name} nie jest zainstalowany!")
        if install_choice is None:
            install_choice = input(f"Zainstalować {model_name}? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            try:
                print(f" 📥 Instalowanie modelu {model_name}...")
                # DISABLED: Don't delete all models - we want to keep multiple models
                # delete_all_models()
                pull_model_with_progress(model_name)
                
                # Sprawdź czy model rzeczywiście się zainstalował
                try:
                    available_models = ollama.list()
                    available_model_names = []
                    
                    if hasattr(available_models, 'models'):
                        for model in available_models.models:
                            name = model.model if hasattr(model, 'model') else str(model)
                            available_model_names.append(name)
                    elif isinstance(available_models, dict) and 'models' in available_models:
                        for model in available_models['models']:
                            if isinstance(model, dict):
                                name = model.get('model', model.get('name', str(model)))
                            else:
                                name = str(model)
                            available_model_names.append(name)
                    
                    if model_name in available_model_names:
                        print(f" ✅ Model {model_name} zainstalowany!")
                        return True
                    else:
                        print(f" ❌ Model {model_name} nie został zainstalowany poprawnie!")
                        return False
                        
                except Exception as verify_error:
                    print(f" ❌ Błąd weryfikacji instalacji {model_name}: {verify_error}")
                    return False
                    
            except Exception as e:
                print(f" ❌ Błąd instalacji {model_name}: {e}")
                return False
        else:
            print(" ⏭️  Pomijam ten model...")
            return False

    @staticmethod
    def ensure_agent_enum(agent):
        """Helper function to ensure we have an Agent enum
        eg. 
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> from base_eval import BaseEvaluation, Agent
        >>> BaseEvaluation.ensure_agent_enum(Agent.CONSTANT_DATA_EN)
        <Agent.CONSTANT_DATA_EN: 'constant_data_en'>

        >>> BaseEvaluation.ensure_agent_enum("constant_data_en")
        <Agent.CONSTANT_DATA_EN: 'constant_data_en'>

        >>> BaseEvaluation.ensure_agent_enum("constant_data_en").value
        'constant_data_en'
        """
        if isinstance(agent, Agent):
            return agent
        # Accept any Enum (possibly from another module) and map by value
        from enum import Enum as _Enum
        if isinstance(agent, _Enum):
            return Agent(agent.value)
        if isinstance(agent, str):
            return Agent(agent)
        return Agent(str(agent))




    def _parse_json_strings_recursively(self, obj):
        """Rekurencyjnie parsuje wszystkie JSON stringi w strukturze do obiektów."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                result[key] = self._parse_json_strings_recursively(value)
            return result
        elif isinstance(obj, list):
            return [self._parse_json_strings_recursively(item) for item in obj]
        elif isinstance(obj, str):
            # Spróbuj sparsować string jako JSON
            try:
                parsed = json.loads(obj)
                # Jeśli się udało, rekurencyjnie parsuj dalej
                return self._parse_json_strings_recursively(parsed)
            except (json.JSONDecodeError, TypeError):
                # Jeśli nie da się sparsować, zostaw jako string
                return obj
        else:
            return obj

    def get_llama_response(self, tools_schema, context_messages, optimisations=None):
        """Get response using local llama-server. Returns same format as get_ollama_response."""
        
        payload = {
            "messages": context_messages,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "max_tokens": self.MAX_TOKENS,
            "context_size": self.CONTEXT_SIZE,
            "stream": False,
            "seed": self.SEED,
        }
        if tools_schema:
            payload["tools"] = tools_schema
        print("🦙 Using local llama-server for inference")
        try:
            start_resources = self.monitor_resources()
            
            start_time = time.time()
            response = requests.post(
                f"{self.LLAMA_SERVER_URL}/v1/chat/completions",
                json=payload,
                timeout=self.TIMEOUT
            )
            end_time = time.time()
            end_resources = self.monitor_resources()
            result = response.json()
  
        except Exception as e:
            print(f"❌ Error in llama-server request: {e}")
            raise
        
        # Model odpowiada zgodnie z promptem - zwracamy jego naturalną odpowiedź
        # Może to być content (tekst) lub tool_calls (wywołania narzędzi)
        message = result.get('choices', [{}])[0].get('message', {})
        
        # Model odpowiedział - wyciągnij tylko tool_calls jeśli istnieją
        if message.get('tool_calls'):
            # Rekurencyjnie parsuj wszystkie JSON stringi dla czytelnego formatowania
            parsed_tool_calls = self._parse_json_strings_recursively(message['tool_calls'])
            
            response_json = self.pretty_json({
                "tool_calls": parsed_tool_calls
            }, 'str')
        elif message.get('content'):
            response_json = self.pretty_json(message['content'], 'str')
        else:
            # Fallback - zwróć całą wiadomość
            response_json = self.pretty_json(message, 'str')

        # Build latency breakdown from llama-server timings if available
        latency_breakdown = self._parse_llama_timings(result, start_time, end_time, optimisations)
        latency_breakdown['start_resources'] = start_resources
        latency_breakdown['end_resources'] = end_resources
        
        # Calculate resource differences
        resource_differences = self.calculate_resource_differences(start_resources, end_resources)
        if resource_differences:
            latency_breakdown['resource_differences'] = resource_differences

        return response_json, latency_breakdown
        

    def get_ollama_response(self, tools_schema, context):
        
        #print(f"ollama response round_number: {round_number}")
        print(f"generating ollama response")
        start_resources = self.monitor_resources()
        response = self.OLLAMA_CLIENT.chat(
            model=self.model_name,
            messages=context,
            tools=tools_schema if tools_schema else None,
            options={
                'temperature': self.TEMPERATURE,
                'top_p': self.TOP_P,
                'max_tokens': self.MAX_TOKENS,
                'context_size': self.CONTEXT_SIZE,
            }
        )


        response_message = response['message']['content']
        # print(f"!!!!!\nRAW OLLAMA Response: {self.pretty_json(response, 'str')}")
        response_json = self.pretty_json(response_message, 'str')

        # Prawdziwe czasy z Ollama (nanosekund → milisekundy)
        total_time = response.get('total_duration', 0) / 1_000_000
        load_time = response.get('load_duration', 0) / 1_000_000
        prompt_eval_time = response.get('prompt_eval_duration', 0) / 1_000_000
        eval_time = response.get('eval_duration', 0) / 1_000_000

        latency_breakdown = {
            'total_ms': total_time,
            'model_loading_ms': load_time,
            'prompt_evaluation_ms': prompt_eval_time,
            'token_generation_ms': eval_time,
            'optimizations_details': {},
            'resources': start_resources,
            'breakdown_percentage': {
                'loading': (load_time / total_time) * 100 if total_time > 0 else 0,
                'prompt_eval': (prompt_eval_time / total_time) * 100 if total_time > 0 else 0,
                'generation': (eval_time / total_time) * 100 if total_time > 0 else 0,
            },
            'tokens': {
                'prompt_count': response.get('prompt_eval_count', 0),
                'generated_count': response.get('eval_count', 0),
                'throughput_tokens_per_sec': response.get('eval_count', 0) / (eval_time / 1000) if eval_time > 0 else 0,
                'prompt_per_token_ms': None,
                'prompt_per_second':  None,
                'predicted_per_token_ms':  None,
                'predicted_per_second':  None
            }
        }


        return response_json, latency_breakdown

    def _parse_llama_timings(self, result: dict, start_time: float, end_time: float, optimisations: dict = None) -> dict:
        """Parse llama-server timing fields into a consistent latency_breakdown dict.
        Supports both top-level result['timings'] and per-choice timings under result['choices'][0]['timings'].
        When fields are missing, falls back to wall-clock total time.
        """
        total_time_ms = round((end_time - start_time) * 1000, 2)
        timings = result.get('timings') or {}
        choice_timings = None
        try:
            choice_timings = result.get('choices', [{}])[0].get('timings')
        except Exception:
            choice_timings = None

        # Prefer top-level timings; fallback to choice timings
        src = timings if isinstance(timings, dict) and timings else (choice_timings if isinstance(choice_timings, dict) else {})

        prompt_ms = src.get('prompt_ms') if isinstance(src, dict) else None
        predicted_ms = src.get('predicted_ms') if isinstance(src, dict) else None
        prompt_n = src.get('prompt_n') if isinstance(src, dict) else None
        predicted_n = src.get('predicted_n') if isinstance(src, dict) else None


        # Compute total from parts when available
        if isinstance(prompt_ms, (int, float)) and isinstance(predicted_ms, (int, float)):
            computed_total = round(prompt_ms + predicted_ms, 2)
        else:
            computed_total = total_time_ms

        throughput = None
        if isinstance(predicted_ms, (int, float)) and predicted_ms > 0 and isinstance(predicted_n, (int, float)):
            throughput = predicted_n / (predicted_ms / 1000.0)

               # Dodatkowe statystyki z llama.cpp
        prompt_per_token_ms = src.get('prompt_per_token_ms') if isinstance(src, dict) else None
        prompt_per_second = src.get('prompt_per_second') if isinstance(src, dict) else None
        predicted_per_token_ms = src.get('predicted_per_token_ms') if isinstance(src, dict) else None
        predicted_per_second = src.get('predicted_per_second') if isinstance(src, dict) else None
        
        breakdown = {
            'total_ms': computed_total,
            'model_loading_ms': None,  # llama-server API does not expose model load time here
            'prompt_eval_ms': float(prompt_ms) if isinstance(prompt_ms, (int, float)) else None,
            'token_generation_ms': float(predicted_ms) if isinstance(predicted_ms, (int, float)) else None,
            'optimizations_details': optimisations if optimisations else {},  # Track optimization parameters
            'breakdown_percentage': {
                'loading': None,
                'prompt_eval': (float(prompt_ms) / computed_total * 100.0) if isinstance(prompt_ms, (int, float)) and computed_total else None,
                'generation': (float(predicted_ms) / computed_total * 100.0) if isinstance(predicted_ms, (int, float)) and computed_total else None,
            },
            'tokens': {
                'prompt_count': int(prompt_n) if isinstance(prompt_n, (int, float)) else None,
                'generated_count': int(predicted_n) if isinstance(predicted_n, (int, float)) else None,
                'throughput_tokens_per_sec': throughput,
                'prompt_per_token_ms': float(prompt_per_token_ms) if isinstance(prompt_per_token_ms, (int, float)) else None,
                'prompt_per_second': float(prompt_per_second) if isinstance(prompt_per_second, (int, float)) else None,
                'predicted_per_token_ms': float(predicted_per_token_ms) if isinstance(predicted_per_token_ms, (int, float)) else None,
                'predicted_per_second': float(predicted_per_second) if isinstance(predicted_per_second, (int, float)) else None
            }
        }
        # Safely calculate model_loading_ms with None checks
        if computed_total is not None and prompt_ms is not None and predicted_ms is not None:
            model_loading_ms = float(computed_total) - float(prompt_ms) - float(predicted_ms)
            if model_loading_ms < 0:
                model_loading_ms = 0.0
        else:
            model_loading_ms = 0.0 
        breakdown['model_loading_ms'] = model_loading_ms
        breakdown['breakdown_percentage']['loading'] = (model_loading_ms / computed_total * 100.0) if isinstance(computed_total, (int, float)) and computed_total else None
        print(f"breakdown: \n{breakdown}")
        return breakdown

    def run_inference(self, tools_schema, context, fallback = False,optimisations={}):
        """Run inference using the best available method."""
        import requests
        import subprocess
        import time

        # Check if llama-server is running and supports tools
        self.inference_engine = "llama-server" # Primary engine
        try:
            model_path = self.get_model_path_from_ollama(self.model_name)
            response = requests.get(f"{self.LLAMA_SERVER_URL}/v1/models", timeout=self.TIMEOUT)
            if response.status_code == 200:
                server_models = response.json().get("data", [])
                loaded_model_path = server_models[0].get("id") if server_models else None
                
                # Verify loaded model matches self.model_name or its path
                is_correct = False
                if loaded_model_path and model_path:
                    # Normalize paths for absolute comparison
                    norm_loaded = os.path.abspath(os.path.expanduser(loaded_model_path))
                    norm_target = os.path.abspath(os.path.expanduser(model_path))
                    is_correct = (norm_loaded == norm_target)
                
                if not is_correct and loaded_model_path:
                    # Fallback check: maybe it's just the name or basename
                    is_correct = (self.model_name in loaded_model_path or os.path.basename(loaded_model_path) in self.model_name)

                if is_correct:
                    print(f"✅ Reusing existing llama-server with correct model: {loaded_model_path}")
                    return self.get_llama_response(tools_schema, context, optimisations=optimisations)
                else:
                    print(f"⚠️ Existing llama-server has different model loaded: {loaded_model_path}. Target: {model_path}. Restarting...")
                    self._kill_llama_server_processes()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error checking llama-server: {e}")

        # Start a new llama-server
        if model_path and self.start_local_llama_server(model_path, optimisations=optimisations):
            print("✅ Started new llama-server. Waiting 5s for stability...")
            time.sleep(5)  # Add stability buffer
            try:
                return self.get_llama_response(tools_schema, context, optimisations=optimisations)
            except Exception as e:
                print(f"❌ llama-server call failed after start: {e}")
        else:
            print("❌ Could not start llama-server")
        
        # Fallback to Ollama API is commented out per user request
        # print("🔄 Fallback to Ollama API is disabled")
        # if fallback:
        #     self.inference_engine = "ollama"
        #     return self.get_ollama_response(tools_schema, context)
        
        return None, {}

    # def start_local_llama_server(self, model_path, optimisations={}):
    #     """Start local llama-server with the specified model."""
    #     import subprocess
    #     import time
    #     import requests
    #     # Clean up old llama-server processes only (avoid killing our Python client)
    #     self._kill_llama_server_processes()

    #     print(f"🦙 Starting llama-server on port {self.LLAMA_SERVER_PORT} with model: {model_path}")
        
    #     cmd = [
    #         self.LLAMA_SERVER_PATH,
    #         "--model", model_path,
    #         "--port", str(self.LLAMA_SERVER_PORT),
    #         "--host", str(self.LLAMA_SERVER_HOST),
    #         "--ctx-size", str(self.CONTEXT_SIZE),
    #         "--n-predict", str(self.MAX_TOKENS),
    #         "--jinja",
    #         "--verbose",
    #     ]
    #     if optimisations.get("--cache-type-k") or optimisations.get("--cache-type-v"):
    #         kv_cache = self._get_optimal_kv_cache_type()

    #         if kv_cache:
    #             optimisations["--cache-type-k"] = kv_cache
    #             optimisations["--cache-type-v"] = kv_cache
    #         else:
    #             print(f"❌ No optimal KV cache type found for {kv_cache}")
        
    #     for param, value in optimisations.items():
    #         print(f"Optimisation: {param} = {value}")
    #         if value is not None:
    #             cmd.extend([param, str(value)])
    #         else:
    #             cmd.extend([param])

    #     self.llama_server_process = subprocess.Popen(
    #         cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    #     )

    #     print("🦙 Waiting for model to load...")
    #     for i in range(300):
    #         try:
    #             response = requests.get(f"{self.LLAMA_SERVER_URL}/v1/models", timeout=2)
    #             if response.status_code == 200:
    #                 print("🦙 Local llama-server started successfully")
    #                 return True
    #         except Exception:
    #             pass
    #         time.sleep(1)
    #         if i % 30 == 0 and i > 0:
    #             print(f"🦙 Still loading model... ({i}s)")
    #     return False

    def start_local_llama_server(self, model_path, optimisations=None):
        """Start local llama-server with the specified model and optimizations.
        
        Args:
            model_path: Path to the model file
            optimisations: Dictionary of optimization parameters (e.g., {"--flash-attn": None})
            
        Returns:
            bool: True if server started successfully, False otherwise
        """
        import subprocess
        import time
        import requests
        
        # Initialize optimisations if not provided
        if optimisations is None:
            optimisations = {}
        
        # Clean up old llama-server processes
        self._kill_llama_server_processes()

        print(f"🦙 Starting llama-server on port {self.LLAMA_SERVER_PORT}")
        print(f"📦 Model: {model_path}")
        
        # Base command
        cmd = [
            self.LLAMA_SERVER_PATH,
            "--model", model_path,
            "--port", str(self.LLAMA_SERVER_PORT),
            "--host", str(self.LLAMA_SERVER_HOST),
            "--ctx-size", str(self.CONTEXT_SIZE),
            "--n-predict", str(self.MAX_TOKENS),
            "--jinja",
            "--verbose",
        ]
        print(f'optimisations: {optimisations}')
        print(f'optimisation.key("--cache-type-k"): {optimisations.get("--cache-type-k")}')
        # Handle KV cache optimization
        if optimisations is None:
            optimisations = {}

        kv_cache = self._get_optimal_kv_cache_type()
        
        # # If KV cache is enabled but no specific cache types are set
        # if "--kv-cache" in optimisations and not any(k in optimisations for k in ["--cache-type-k", "--cache-type-v"]):
        #     if kv_cache:
        #         optimisations["--cache-type-k"] = kv_cache
        #         optimisations["--cache-type-v"] = kv_cache
        #         print(f"🔧 Using KV cache type: {kv_cache}")
        #     else:
        #         print("⚠️  No optimal KV cache type found, using defaults")
        
        # Add optimization parameters
        if optimisations:
            print("\n🛠️  Applying optimizations:")
            for param, value in optimisations.items():
                print(f"   {param}" + (f"={value}" if value is not None else ""))
                cmd.append(param)
                if value is not None:
                    cmd.append(str(value))
        
        # Start the server
        try:
            print("\n🚀 Starting server with command:")
            print("   " + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
            
            self.llama_server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
                
            )
            
            # Wait for server to be ready
            print("\n⏳ Waiting for model to load...")
   

            start_time = time.time()
            
            for attempt in range(self.TIMEOUT):  # Timeout from config.yaml
                # Check if process crashed
                if self.llama_server_process.poll() is not None:
                    stderr_output = self.llama_server_process.stderr.read()
                    stdout_output = self.llama_server_process.stdout.read()
                    print(f"\n❌ llama-server process crashed!")
                    print(f"Exit code: {self.llama_server_process.returncode}")
                    if stderr_output:
                        print(f"STDERR:\n{stderr_output[:2000]}")
                    if stdout_output:
                        print(f"STDOUT:\n{stdout_output[:2000]}")
                    return False
                
                try:
                    response = requests.get(
                        f"{self.LLAMA_SERVER_URL}/v1/models", 
                        timeout=2
                    )
                    if response.status_code == 200:
                        load_time = time.time() - start_time
                        print(f"✅ Server started successfully in {load_time:.1f}s")
                        return True
                except (requests.RequestException, ConnectionError):
                    pass
                    
                # Print status every 10 seconds
                if attempt > 0 and attempt % 10 == 0:
                    print(f"   Still loading... ({attempt}s)")
                    
                time.sleep(1)
                
        except Exception as e:
            print(f"\n❌ Failed to start server: {str(e)}")
            if hasattr(self, 'llama_server_process'):
                self.llama_server_process.kill()
            return False
        
        print("\n❌ Server failed to start: Timeout")
        # Print stderr/stdout on timeout
        if hasattr(self, 'llama_server_process'):
            try:
                stderr_output = self.llama_server_process.stderr.read()
                stdout_output = self.llama_server_process.stdout.read()
                if stderr_output:
                    print(f"STDERR (last 2000 chars):\n{stderr_output[-2000:]}")
                if stdout_output:
                    print(f"STDOUT (last 2000 chars):\n{stdout_output[-2000:]}")
            except:
                pass
            self.llama_server_process.kill()
        return False

    def _kill_llama_server_processes(self):
        """Safely terminate ONLY llama-server processes bound to the configured port.
        Avoid killing the Python client which may also have an open socket to that port.
        """
        import subprocess
        import os
        import signal
        try:
            # Get PIDs listening on the port (LISTEN only avoids clients)
            result = subprocess.run(
                ["lsof", "-tiTCP:%s" % self.LLAMA_SERVER_PORT, "-sTCP:LISTEN"],
                capture_output=True, text=True
            )
            pids = [pid for pid in result.stdout.strip().split("\n") if pid]
            if not pids:
                return
            for pid in pids:
                # Confirm the process is llama-server
                cmdline = subprocess.run(["ps", "-p", pid, "-o", "command="], capture_output=True, text=True)
                command = cmdline.stdout.strip()
                if not command:
                    continue
                if os.path.basename(self.LLAMA_SERVER_PATH) in command or "llama-server" in command:
                    try:
                        print(f"🦙 Killing llama-server process {pid} ({command})")
                        os.kill(int(pid), signal.SIGTERM)
                    except Exception as e:
                        print(f"⚠️ SIGTERM failed for {pid}: {e}")
                    # If still alive, force kill
                    try:
                        subprocess.run(["kill", "-9", pid], capture_output=True)
                    except Exception as e:
                        print(f"⚠️ SIGKILL failed for {pid}: {e}")
                else:
                    # Skip non llama-server processes (e.g. our Python client)
                    print(f"↩️ Skipping non llama-server process {pid}: {command}")
        except Exception as e:
            print(f"⚠️ Failed to enumerate/kill llama-server processes: {e}")
    @staticmethod
    def get_next_run_number_for_path(base_dir: str) -> int:
        """Get next run number for model and all_models directories - returns (model_run, all_models_run).

        >>> from base_eval import Agent
        >>> import tempfile, os
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
        >>> with tempfile.TemporaryDirectory() as d:
        ...     all_models_dir = os.path.join(d, "unreferenced", "agents", "constant_data_en", "all_models")
        ...     os.makedirs(os.path.join(all_models_dir, "run_014"))
        ...     model_run = BaseEvaluation.get_next_run_number_for_path(base_dir=all_models_dir)
        ...     print(f"{model_run}")
        15
        >>> with tempfile.TemporaryDirectory() as d:
        ...     all_models_dir = os.path.join(d, "unreferenced", "agents", "constant_data_en", "all_models")
        ...     os.makedirs(os.path.join(all_models_dir, "run_010"))
        ...     os.makedirs(os.path.join(all_models_dir, "run_007"))
        ...     model_run = BaseEvaluation.get_next_run_number_for_path(base_dir=all_models_dir)
        ...     print(f"{model_run}")
        11
        """

        if not os.path.exists(base_dir):
            return 1

        existing_runs = [d for d in os.listdir(base_dir) if d.startswith(
            'run_') and os.path.isdir(os.path.join(base_dir, d))]
        if not existing_runs:
            return 1

        run_numbers = []
        for run_dir in existing_runs:
            try:
                # Extract run number from dirname like "run_001"
                run_num = int(run_dir.replace('run_', ''))
                run_numbers.append(run_num)
            except ValueError:
                continue

        return max(run_numbers) + 1 if run_numbers else 1


    
    def construct_main_paths_for_eval(self, model_name_norm=None) -> str:
        """Construct path from metadata. 
        eg. 
        >>> from referenced_clean import EvalModelsReferenced
        >>> from base_eval import Agent
        >>> from pprint import pprint
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
        >>> with redirect_stdout(io.StringIO()):
        ...     be = EvalModelsReferenced(model_name=model_name, agent=Agent.CONSTANT_DATA_EN)
        >>> pprint(be.construct_main_paths_for_eval()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'all_models': 'source/output/agents/constant_data_en/referenced/all_models',
         'log': 'source/output/agents/constant_data_en/referenced/log',
         'model': 'source/output/agents/constant_data_en/referenced/model/...'}
        """
        if model_name_norm is None:
            model_name_norm = self.model_name_norm
        return {"model": os.path.join("output", "agents", self.agent_type, self.eval_type, "model", model_name_norm),
                "all_models": os.path.join("output", "agents", self.agent_type, self.eval_type, "all_models"),
                "log": os.path.join("output", "agents", self.agent_type, self.eval_type, "log")}

    @staticmethod
    def _create_session_folders_and_log_file(path_map: dict, base_path: str, fixed_run_number: int = None, fixed_timestamp: str = None, create_all_models: bool = True) -> dict:
        """Tworzy foldery dla sesji ewaluacji
        eg
        >>> import os, tempfile
        >>> from base_eval import BaseEvaluation, Agent
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path_map = {
        ...         "model": os.path.join(d, "source", "output", "referenced", "agents", "constant_data_en", "model", "granite3.1-dense_2b-instruct-fp16"),
        ...         "all_models": os.path.join(d, "source", "output", "referenced", "agents", "constant_data_en", "all_models"),
        ...         "log": os.path.join(d, "source", "output", "referenced", "agents", "constant_data_en", "log"),
        ...     }
        ...     res = BaseEvaluation._create_session_folders_and_log_file(path_map, d)
        ...     res = BaseEvaluation._create_session_folders_and_log_file(path_map, d)
        ...     print(res["model_run_folder"][-3:], res["all_models_run_folder"][-3:])
        ...     print(res["model_run"], res["all_models_run"]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        002 002
        run_002 run_002
        """

        # create folders paths
        log_folder = os.path.join(base_path, path_map["log"])
        model_folder = os.path.join(base_path, path_map["model"])
        all_models_folder = os.path.join(base_path,path_map["all_models"])

        # Calculate max run number from both folders to keep them in sync
        if fixed_run_number is not None:
             model_run = f"run_{fixed_run_number:03d}"
             all_models_run = f"run_{fixed_run_number:03d}"
        else:
             model_run_num = BaseEvaluation.get_next_run_number_for_path(model_folder)
             all_models_run_num = BaseEvaluation.get_next_run_number_for_path(all_models_folder)
             
             model_run = f"run_{model_run_num:03d}"
             all_models_run = f"run_{all_models_run_num:03d}"
        

        model_folder = os.path.join( model_folder, model_run)
        all_models_folder = os.path.join(all_models_folder, all_models_run)

        # create folders if not exist
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(model_folder, exist_ok=True)
        if create_all_models:
            os.makedirs(all_models_folder, exist_ok=True)

        return {"model_run":model_run, "all_models_run":all_models_run, "model_run_folder": model_folder, "all_models_run_folder": all_models_folder, "log_folder": log_folder, "timestamp": fixed_timestamp if fixed_timestamp else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}

    def get_model_path_from_ollama(self, model_name):
        """Get the local file path for a model from Ollama storage using Ollama API."""
        try:
            import ollama
            
            # Use Ollama API to get model info
            model_info = ollama.show(model_name)
            
            # Extract the modelfile to find the FROM line with blob reference
            modelfile = model_info.get('modelfile', '')
            
            # Parse the FROM line to get the blob hash
            # Format: FROM /path/to/blob or FROM @sha256:hash
            for line in modelfile.split('\n'):
                line = line.strip()
                if line.startswith('FROM '):
                    from_path = line[5:].strip()
                    
                    # If it's already a full path, use it
                    if from_path.startswith('/') or from_path.startswith('~'):
                        blob_path = os.path.expanduser(from_path)
                        if os.path.exists(blob_path):
                            return blob_path
                    
                    # If it's a blob reference (@sha256:hash), construct path
                    if from_path.startswith('@sha256:'):
                        blob_hash = from_path[8:]  # Remove '@sha256:' prefix
                        blob_path = os.path.expanduser(f"~/.ollama/models/blobs/sha256-{blob_hash}")
                        if os.path.exists(blob_path):
                            return blob_path
            
            # Fallback: try to extract from model details
            details = model_info.get('details', {})
            if 'parent_model' in details:
                # Recursively try parent model
                return self.get_model_path_from_ollama(details['parent_model'])
                
            print(f"⚠️ Could not find blob path for {model_name}")
            return None
            
        except Exception as e:
            print(f"❌ Error getting model path for {model_name}: {e}")
            return None

    
    def _get_optimal_kv_cache_type(self, quantisation=None):
        """Determine optimal KV cache type based on cached model metadata.
        
        >>> from base_eval import BaseEvaluation, Agent
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
        >>> with redirect_stdout(io.StringIO()):
        ...     be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN) # doctest: +ELLIPSIS
        ...     be.current_nodel_metadata = {"quantization_level": "fp8"}
        >>> print(be._get_optimal_kv_cache_type("fp8"))
        q8_0
        """
        import re
        allowed = ["f16", "q8_0", "q4_0", "q2_0"]

        if quantisation is None:
            metadata = self.current_model_metadata
            quantization = metadata.get('quantization_level')
        else:
            quantization = quantisation

        if quantization is None:
            return "f16"  # Fallback

        normalized_quantization = str(quantization).lower()
        
        if normalized_quantization in allowed:
            return normalized_quantization
        
        # Wyciągnij liczbę z inputu (np. "16" z "fp16")
        number_match = re.search(r'\d+', normalized_quantization)
        if not number_match:
            return "f16"  # Fallback jeśli brak liczby
        input_number = int(number_match.group())  # np. 16 jako int
        
        # Porównaj numerycznie z liczbami w allowed
        for option in allowed:
            option_number_match = re.search(r'\d+', option)
            if option_number_match:
                option_number = int(option_number_match.group())  # np. 16 z "f16"
                if input_number == option_number:
                    return option
        
        return "f16"  # Fallback jeśli brak dopasowania

    #@todo sprawdzic i poprawic
    
    def save_and_get_current_model_metadata(self, source_path:str=None) -> tuple[dict, dict]:
        """Ensure model metadata is cached.
        eg:
        >>> from base_eval import BaseEvaluation, Agent
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> from pprint import pprint 
        >>> import tempfile
        >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
        >>> with redirect_stdout(io.StringIO()):
        ...     be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN) # doctest: +ELLIPSIS
        ...     with tempfile.TemporaryDirectory() as d:
        ...         metadata, _ = be.save_and_get_current_model_metadata(source_path=d)
        >>> pprint(metadata.keys()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        dict_keys(['architecture', 'parameter_size', 'parameter_size_orginal', 'context_length', 'embedding_length', 'quantization_level', 'model_format', 'model_size_bytes', 'model_size_gb', 'cached_at', 'context_size', 'max_tokens', 'parameter_size_display'])
        """

        metadata_file, existed = self.get_or_create_file_or_folder(file_name="models_metadata", type_of_file="metadata", source_path=source_path)
        if existed:
            metadata_data = self.load_json_file(metadata_file)
        else:
            metadata_data = {"model": {}}

        # CRITICAL: Metadata is SACRED - never overwrite if it exists
        if metadata_data.get("model", {}).get(self.model_name):  
            cached_metadata = metadata_data["model"][self.model_name]
            print(f"✅ Model metadata already cached for {self.model_name} - NEVER overwriting")
            
            # Warn if metadata has invalid values (but don't fix them)
            if cached_metadata.get('model_size_gb', 0) == 0.0:
                print(f"⚠️ WARNING: Cached metadata for {self.model_name} has model_size_gb=0.0")
                print(f"⚠️ This likely means metadata was saved when model was not in Ollama")
                print(f"⚠️ To fix: restore old models_metadata.json or delete this entry and re-run with model in Ollama")
            
            return cached_metadata, metadata_data
        
        # Only fetch and save metadata if it doesn't exist yet    
        print(f"📥 Fetching NEW metadata for {self.model_name} (first time)")
        
        # CRITICAL: Check if model is in Ollama before fetching metadata
        try:
            self.OLLAMA_CLIENT.show(self.model_name)
        except Exception as e:
            print(f"⚠️ WARNING: Model {self.model_name} not in Ollama - skipping metadata fetch")
            print(f"⚠️ Metadata will be fetched when model is available")
            # Return dummy metadata to prevent crashes
            dummy_metadata = {
                'architecture': 'Unknown',
                'parameter_size': 0,
                'parameter_size_orginal': '0B',
                'context_length': 0,
                'embedding_length': 0,
                'quantization_level': 'Unknown',
                'model_format': 'gguf',
                'model_size_bytes': 0,
                'model_size_gb': 0.0,
                'cached_at': datetime.now().isoformat(),
                'context_size': self.CONTEXT_SIZE,
                'max_tokens': self.MAX_TOKENS,
                'parameter_size_display': '0B'
            }
            return dummy_metadata, metadata_data
        
        current_model_metadata = self.extract_and_norma_model_param_from_ollama()
        metadata_data["model"][self.model_name] = current_model_metadata
        print(f"💾 Saving metadata for {self.model_name} (this will NEVER be overwritten)")
        print(f"All metadata after update:")
        print(metadata_data)
        self.save_json_file(metadata_data, metadata_file)
        print(f"Current model metadata:")
        print(current_model_metadata)
        return current_model_metadata, metadata_data

    def extract_and_norma_model_param_from_ollama(self):
        """
        Pobiera informacje o modelu z Ollamy (rozmiar, parametry, etc.)
        
        Args:
            model_name (str): Nazwa modelu w Ollamie
            
        Returns:
            Optional[Dict[str, Any]]: Metadane modelu lub None jeśli błąd

        Przyklad pelnego wyniku dla modelu granite3.1-dense:2b-instruct-fp16:
        > pprint(extracted_model_details) 
        {
            'architecture': 'granite',
            'cached_at': '...',
            'context_length': 131072,
            'context_size': 7000,
            'embedding_length': 2048,
            'max_tokens': 3500,
            'model_format': 'gguf',
            'model_size_bytes': 5476083302,
            'model_size_gb': 5.1,
            'parameter_size': 2500000000,
            'parameter_size_display': '2.5B',
            'parameter_size_orginal': '2.5B',
            'quantization_level': 'F16'
        } 

        >>> from base_eval import BaseEvaluation, Agent
        >>> from pprint import pprint
        >>> import io, sys
        >>> from contextlib import redirect_stdout
        >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
        >>> with redirect_stdout(io.StringIO()):
        ...    be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN) # doctest: +ELLIPSIS
        ...    extracted_model_details = be.extract_and_norma_model_param_from_ollama()
        >>> pprint(extracted_model_details.keys())
        dict_keys(['architecture', 'parameter_size', 'parameter_size_orginal', 'context_length', 'embedding_length', 'quantization_level', 'model_format', 'model_size_bytes', 'model_size_gb', 'cached_at', 'context_size', 'max_tokens', 'parameter_size_display'])
        """
        print(f"📥 Fetching model metadata for {self.model_name}...")
        
        # Check cached metadata first to avoid unnecessary ollama calls
        if hasattr(self, 'models_metadata') and self.models_metadata and self.model_name in self.models_metadata:
             print(f"✅ Found cached metadata for {self.model_name}")
             return self.models_metadata[self.model_name]
             
        model_details = {}
        show_output = ""
        
        try:
            result = subprocess.run(['ollama', 'show', self.model_name], capture_output=True, text=True, check=True)
            show_output = result.stdout
            print(f"Step 1: ollama show output: {show_output}")
        except Exception as e:
            print(f"⚠️ Could not run 'ollama show {self.model_name}': {e}. Using placeholders.")
            show_output = ""

        import re
        for key, pattern in [
            ('family', r'architecture\s+(\w+)'),
            ('parameter_size', r'parameters\s+([\d.]+[BMK])'),
            ('quantization_level', r'quantization\s+(\w+)'),
            ('context_length', r'context length\s+(\d+)'),
            ('embedding_length', r'embedding length\s+(\d+)'),
            ('modelfile', r'modelfile\s+(\w+)')
        ]:
            default_val = 'Unknown'
            if key in ['context_length', 'embedding_length']:
                default_val = '0'
            elif key == 'parameter_size':
                default_val = '0B'
            
            match = re.search(pattern, show_output)
            model_details[key] = match.group(1) if match else default_val
        print(f"Step 1: extracted from ollama show: {model_details}")

        model_details['format'] = 'gguf'
        model_size_bytes = 0
        
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            print(f"Step 2: ollama list output: {result}")
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[1:]:
                    parts = line.split()
                    if parts and parts[0] == self.model_name:
                        number_str = parts[2]
                        unit_str = parts[3].upper()
                        print(f"Step 2: extract model size: {number_str} unit: {unit_str}")

                        if 'GB' in unit_str:
                            model_size_bytes = int(float(number_str) * 1024**3)
                            model_size_gb = number_str
                            print(f"Step 2: model_size_bytes if GB: {model_size_bytes}")
                        elif 'MB' in unit_str:
                            model_size_bytes = int(float(number_str) * 1024**2)
                            print(f"Step 2: model_size_bytes if MB: {model_size_bytes}")
                        elif 'KB' in unit_str:
                            model_size_bytes = int(float(number_str) * 1024)
                            print(f"Step 2: model_size_bytes if KB: {model_size_bytes}")
                        elif 'B' in unit_str:
                            model_size_bytes = int(float(number_str))
                            print(f"Step 2: model_size_bytes if B: {model_size_bytes}")
                        break
        except Exception as e:
            print(f"⚠️ Could not run 'ollama list': {e}")
        
        print(f"Step 3: normalized model_size_bytes: {model_size_bytes}")

        
        param_size = model_details['parameter_size']
        if isinstance(param_size, str):
            # Convert "2.5B" to 2.5, "600M" to 0.6, etc.
            param_size_upper = param_size.upper().strip()
            print(f"step 4: param_size_upper: {param_size_upper}")
            try:
                if 'B' in param_size_upper:
                    # Billions
                    param_size_extracted = float(param_size_upper.replace('B', '').strip())
                    param_size_dispaly = param_size_extracted
                    parameter_size_nominal = param_size_extracted * 10**9

                elif 'M' in param_size_upper:
                    # Millions - convert to billions
                    param_size_extracted = float(param_size_upper.replace('M', '').strip()) 
                    param_size_dispaly = param_size_extracted / 1000.0
                    parameter_size_nominal = param_size_extracted * 10**6
                elif 'K' in param_size_upper:
                    # Thousands - convert to billions
                    param_size_extracted = float(param_size_upper.replace('K', '').strip())
                    param_size_dispaly = param_size_extracted / 1000000.0
                    parameter_size_nominal = param_size_extracted * 10**3
                else:
                    # Try to parse as number
                    param_size_extracted = float(param_size_upper)
                    param_size_dispaly = param_size_extracted
                    parameter_size_nominal = param_size_extracted
                
        
            except ValueError as e:
                print(f"problem with param_size_extracted: {e} no conversion possible from {param_size_upper} to number")

                param_size_extracted = param_size_upper
                param_size_dispaly = param_size_upper
                parameter_size_nominal = param_size_upper
            model_details['parameter_size_extracted'] = param_size_extracted
            model_details['parameter_size_nominal'] = int(parameter_size_nominal)


            print(f"step 4: param_size_normalized: {param_size_dispaly}")
            if param_size_dispaly > 0:
                if param_size_dispaly >= 1.0:
                    model_details['parameter_size_display'] = f"{param_size_dispaly:.1f}B"
                else:
                    # For models < 1B, show with more precision
                    model_details['parameter_size_display'] = f"{param_size_dispaly:.2f}B"
            else:
                model_details['parameter_size_display'] = model_details['parameter_size'] 
            print(f"step 4: display parameter size: {model_details['parameter_size_display']}")

        # Dodatkowe informacje z modelfile jeśli dostępne
        if 'modelfile' in model_details:
            modelfile_content = model_details['modelfile']
            print(f"step 5: modelfile content: {modelfile_content}")
            # Parsuj modelfile dla dodatkowych parametrów
            if 'PARAMETER' in modelfile_content:
                lines = modelfile_content.split('\n')
                parameters = {}
                for line in lines:
                    if line.strip().startswith('PARAMETER'):
                        parts = line.strip().split(' ', 2)
                        if len(parts) >= 3:
                            param_name = parts[1]
                            param_value = parts[2]
                            parameters[param_name] = param_value
                model_details['model_parameters'] = parameters
                # Inicjalizacja domyślnych wartości (PRZED użyciem)
        if 'parameter_size_display' not in model_details:
            model_details['parameter_size_display'] = model_details['parameter_size']
        if 'model_parameters' not in model_details:
            model_details['model_parameters'] = {}


        current_model_metadata = {
                    'architecture': model_details['family'],
                    'parameter_size': model_details['parameter_size_nominal'],
                    'parameter_size_orginal': model_details['parameter_size_extracted'],
                    'parameter_size_orginal': param_size_upper,
                    'context_length': int(model_details['context_length']) if model_details['context_length'].isdigit() else None,
                    'embedding_length': int(model_details['embedding_length']) if model_details['embedding_length'].isdigit() else None,
                    'quantization_level': model_details['quantization_level'],
                    'model_format': model_details['format'],
                    'model_size_bytes': model_size_bytes,
                    'model_size_gb': round(model_size_bytes / 1024**3, 2),
                    'cached_at': datetime.now().isoformat(),
                    'context_size': self.CONTEXT_SIZE,
                    'max_tokens': self.MAX_TOKENS,
                    'parameter_size_display': model_details['parameter_size_display'],
                    #'model_parameters': model_details['model_parameters']
        }
        return current_model_metadata


    @staticmethod
    def get_truly_untested_models(agent_type, evaluation_type, only_tested_true=True):
        """Pobiera modele z config.yaml które NIE mają wyników w evaluation_resultus.json
        
        Args:
            agent_type: Typ agenta (np. 'constant_data_en')
            evaluation_type: Typ ewaluacji ('referenced' lub 'unreferenced')
            only_tested_true: Czy filtrować tylko modele z tested: true
            optimization_configs: Lista konfiguracji optymalizacji do przetestowania
                                 np. [{'param1': 'value1'}, {'param2': 'value2'}]
        
        Returns:
            Lista tupli (model_name, optimization_config) lub lista model_name jeśli brak optymalizacji
        """
        import os
        import yaml
        
        # Załaduj modele z config (z filtrowaniem tested: true)
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.abspath(os.path.join(base_path, "..", "..", "..", "examples", "desktop", "input", "agents", agent_type, "evaluation_config", "config.yaml"))
        
        if not os.path.exists(config_file):
            print(f"⚠️ Config file not found: {config_file}")
            return []
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            models_config = config.get('models_to_evaluate', [])
            all_models = []
            
            for model in models_config:
                if isinstance(model, dict):
                    model_name = model['name']
                    tested_flag = model.get('tested', False)
                    
                    if only_tested_true:
                        # Tylko modele z tested: true
                        if tested_flag:
                            all_models.append(model_name)
                    else:
                        # Wszystkie modele
                        all_models.append(model_name)
                elif isinstance(model, str):
                    if not only_tested_true:  # Tylko gdy nie filtrujemy tested: true
                        all_models.append(model)
            

        
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return []
        
        if not all_models:
            return []
        

        # Sprawdź które modele mają wyniki w evaluation_results.json
        log_file = os.path.abspath(os.path.join(base_path, "..", "..", "..", "examples", "desktop", "output", "agents", agent_type, evaluation_type, "log", "evaluation_results.json"))
        
        tested_models = set()
        if os.path.exists(log_file):
            try:
                log_data = BaseEvaluation.load_json_file(log_file)
                if log_data and "evaluations" in log_data:
                    for evaluation in log_data["evaluations"]:
                        model_name = evaluation.get("model_info", {}).get("name")
                        if model_name:
                            tested_models.add(model_name)
                print(f"📊 Tested models: {tested_models}")
                print(f"📊 Znaleziono {len(tested_models)} modeli z wynikami w {evaluation_type} log")
            except Exception as e:
                print(f"⚠️ Błąd odczytu loga: {e}")
        
        # Zwróć modele które nie mają wyników
        untested_models = [model for model in all_models if model not in tested_models]
        
        print(f"📋 Modeli do testowania ({evaluation_type}): {len(untested_models)}/{len(all_models)}")
        for model in untested_models:
            print(f"  • {model}")
        # Jeśli podano konfiguracje optymalizacji, stwórz kombinacje model + optymalizacja
        # if optimization_configs:
        #     model_optimization_pairs = []
        #     for model_name in all_models:
        #         for opt_config in optimization_configs:
        #             model_optimization_pairs.append((model_name, opt_config))
        #     return model_optimization_pairs
        # else:
        #     return all_models
        return untested_models
        
# if __name__ == "__main__":
#     from base_eval import BaseEvaluation, Agent
#     import dotenv
#     dotenv.load_dotenv()
#     be = BaseEvaluation(model_name=model_name, agent=Agent.CONSTANT_DATA_EN)

#     #messages = [{"role": "system", "content": "You are a patient in an English-speaking clinic. Answer the doctor's questions naturally.\n\nYOUR PATIENT DATA:\nI'm a woman, height 5'5\", born 03/15/1990. I won't provide any more information.\n\nINSTRUCTIONS:\n- Respond in English\n- Be natural and realistic\n- Keep answers brief and relevant\n- If you don't know something, say you don't know\n- Behave like a real patient would\n\nDoctor asks: \"What is your name?\"\n\nRespond briefly and naturally:"}]
#     messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}]
#     response = be.get_openai_response(
#         messages=messages,
#         response_format='str',
#         use_cache=False,
#         write_cache=False
#     )
#     print(response)
    #    x = { 
    #     "choices": [
    #         {
    #         "finish_reason": "stop",
    #         "index": 0,
    #         "logprobs": null,
    #         "message": {
    #             "annotations": [],
    #             "audio": null,
    #             "content": "The capital of France is Paris.",
    #             "function_call": null,
    #             "refusal": null,
    #             "role": "assistant",
    #             "tool_calls": null
    #         }
    #         }
    #     ],
    #     "created": 1757065343,
    #     "id": "chatcmpl-CCNFXfpCeY9s79btoce71rvxvWQI4",
    #     "model": "GPT-4o-2024-07-18",
    #     "object": "chat.completion",
    #     "service_tier": "default",
    #     "system_fingerprint": "fp_8bda4d3a2c",
    #     "usage": {
    #         "completion_tokens": 7,
    #         "completion_tokens_details": {
    #         "accepted_prediction_tokens": null,
    #         "audio_tokens": 0,
    #         "reasoning_tokens": 0,
    #         "rejected_prediction_tokens": null
    #         },
    #         "prompt_tokens": 24,
    #         "prompt_tokens_details": {
    #         "audio_tokens": 0,
    #         "cached_tokens": 0
    #         },
    #         "total_tokens": 31
    #     }
    #     }
    #     }


    def get_last_sessions(self, key_dict, log_file, group_by_keys) -> dict:
        """
        Returns newest non-empty session for each optimisation type from log file, grouped by model_name and optimisation.
        
        Args:
            key_dic: Dictionary of keys to filter sessions (e.g., model_name, agent_type)
            log_file: Path to log file with evaluation results
            group_by: List of fields to group sessions by (e.g., ['model_name', 'optimisation'])
            optimisation_types: List of optimization configs to filter by, None = all
            
        Returns:
            Dictionary with optimization as key and session data as value
            
        Example:
            >>> import json
            >>> from unittest.mock import mock_open, MagicMock
            >>> from base_eval import BaseEvaluation
            >>> import tempfile
            >>> import io
            >>> from contextlib import redirect_stdout
            >>> from referenced_clean import EvalModelsReferenced
            >>> model_name = BaseEvaluation.get_random_avilable_model_for_doctest()
            >>> log_data = {
            ...     "evaluations": [
            ...         {
            ...             "model_name": "A",
            ...             "agent_type": "1",
            ...             "topP": 0.9,
            ...             "session_timestamp": "2025-06-04T22:55:00Z",
            ...             "optimisation": {"type": "opt1"},
            ...             "rounds": [
            ...                 {
            ...                     "round": 0,
            ...                     "context": "...",
            ...                     "llm_response": "...",
            ...                     "reference_response": "...",
            ...                     "metrics": "...",
            ...                     "latency_breakdown": "..."
            ...                 }
            ...             ]
            ...         },
            ...         {
            ...             "model_name": "A",
            ...             "agent_type": "1",
            ...             "topP": 0.9,
            ...             "session_timestamp": "2025-06-04T23:55:00Z",
            ...             "optimisation": {"type": "opt1"},
            ...             "rounds": [
            ...                 {
            ...                     "round": 0,
            ...                     "context": "...",
            ...                     "llm_response": "...",
            ...                     "reference_response": "...",
            ...                     "metrics": "...",
            ...                     "latency_breakdown": "..."
            ...                 }
            ...             ]
            ...         },
            ...         {
            ...             "model_name": "B",
            ...             "agent_type": "1",
            ...             "topP": 0.9,
            ...             "session_timestamp": "2025-06-04T22:55:00Z",
            ...             "optimisation": {"type": "opt2"},
            ...             "rounds": [
            ...                 {
            ...                     "round": 0,
            ...                     "context": "...",
            ...                     "llm_response": "...",
            ...                     "reference_response": "...",
            ...                     "metrics": "...",
            ...                     "latency_breakdown": "..."
            ...                 }
            ...             ]
            ...         }
            ...     ]
            ... }
            >>> key_dict = {
            ...     "agent_type": "1",
            ...     "topP": 0.9
            ... }
            >>> group_by = ["model_name", "optimisation"]
            >>> with redirect_stdout(io.StringIO()):
            ...    evaluator = EvalModelsReferenced(model_name, Agent.CONSTANT_DATA_EN)
            ...    with tempfile.TemporaryDirectory() as temp_dir:
            ...        log_file = os.path.join(temp_dir, "evaluation_resultus.json")
            ...        BaseEvaluation.save_json_file(log_data, log_file, evaluations=False)
            ...        result = evaluator.get_last_sessions(key_dict, log_file, group_by)
            >>> print(result)
            {('A', (('type', 'opt1'),)): {'model_name': 'A', 'agent_type': '1', 'topP': 0.9, 'session_timestamp': '2025-06-04T23:55:00Z', 'optimisation': {'type': 'opt1'}, 'rounds': [{'round': 0, 'context': '...', 'llm_response': '...', 'reference_response': '...', 'metrics': '...', 'latency_breakdown': '...'}]}, ('B', (('type', 'opt2'),)): {'model_name': 'B', 'agent_type': '1', 'topP': 0.9, 'session_timestamp': '2025-06-04T22:55:00Z', 'optimisation': {'type': 'opt2'}, 'rounds': [{'round': 0, 'context': '...', 'llm_response': '...', 'reference_response': '...', 'metrics': '...', 'latency_breakdown': '...'}]}}

        """
        session_data = self.load_json_file(log_file)
        if not session_data or 'evaluations' not in session_data:
            return {}

        # Filter evaluations based on key_dic and non-empty rounds
        matching_evaluations = self._filter_by_key_dic(session_data['evaluations'], key_dict)

        last_sessions_by_group = self._group_sessions_by_and_filter_last(matching_evaluations, group_by_keys)


        return last_sessions_by_group
    
    @staticmethod
    def _group_sessions_by_and_filter_last(session_data: List, group_by_keys: Optional[List[str]] = None) -> Dict:
        """
        Groups sessions by optimization type.
        
        Args:
            session_data: List of session data
            group_by: List of fields to group sessions by (e.g., ['model_name', 'optimisation'])
            
        Returns:
            Dictionary with optimization as key and session data as value
            
        Example:
        >>> session_data = [
        ...     {
        ...         "model_name": "model1",
        ...         "optimisation": "opt1",
        ...         "session_timestamp": "2025-06-04T22:55:00Z",
        ...         "topP": 0.9,
        ...         "rounds": [
        ...             {
        ...                 "round": 0,
        ...                     "context": "...",
        ...                 }
        ...             ]
        ...     },
        ...     {
        ...         "model_name": "model1",
        ...         "optimisation": "opt1",
        ...         "session_timestamp": "2025-06-04T22:55:00Z",
        ...         "topP": 0.9,
        ...         "rounds": [
        ...             {
        ...                 "round": 0,
        ...                     "context": "...",
        ...                 }
        ...             ]
        ...     },
        ...     {
        ...         "model_name": "model2",
        ...         "optimisation": "opt2",
        ...         "session_timestamp": "2025-06-04T22:55:00Z",
        ...         "topP": 0.9,
        ...         "rounds": [
        ...             {
        ...                 "round": 0,
        ...                     "context": "...",
        ...                 }
        ...             ]
        ...     }
        ... ]
        >>> result = BaseEvaluation._group_sessions_by_and_filter_last(session_data, ['model_name', 'optimisation'])
        >>> result == {('model1', 'opt1'): {'model_name': 'model1', 'optimisation': 'opt1', 'session_timestamp': '2025-06-04T22:55:00Z', 'topP': 0.9, 'rounds': [{'round': 0, 'context': '...'}]}, ('model2', 'opt2'): {'model_name': 'model2', 'optimisation': 'opt2', 'session_timestamp': '2025-06-04T22:55:00Z', 'topP': 0.9, 'rounds': [{'round': 0, 'context': '...'}]}}
        True
        """
        # if group_by_keys not in session_data:
        #     raise ValueError(f"Group by keys {group_by_keys} not found in session data")
        #     return {}
        
        from collections import defaultdict
        try:
            if group_by_keys is not None:
                groups = defaultdict(list)
                for session in session_data:
                    # Handle nested dictionaries by converting them to tuples of (key, value) pairs
                    key_parts = []
                    for k in group_by_keys:
                        value = session[k]
                        if isinstance(value, dict):
                            # Convert dict to tuple of sorted items for hashability
                            value = tuple(sorted(value.items()))
                        key_parts.append(value)
                    key = tuple(key_parts)
                    groups[key].append(session)  # return list of sessions for each key
                
                latest = {}
                for key, sessions in groups.items():
                    latest[key] = max(sessions, key=lambda s: s['session_timestamp'])
                return latest
        except Exception as e:
            print(f"❌ Failed to group sessions: {e}")
            return {}

        


        

    @staticmethod
    def _filter_by_key_dic(session_data: List, key_dic: Dict, min_run_number: int = 1) -> List:
        """
        Returns newest non-empty session for each optimisation type from log file, grouped by model_name and optimisation.
        
        Args:
            key_dic: Dictionary of keys to filter sessions (e.g., model_name, agent_type)
            session_data: Session data with evaluations
            group_by: List of fields to group sessions by (e.g., ['model_name', 'optimisation'])
            optimisation_types: List of optimization configs to filter by, None = all
            
        Returns:
            Dictionary with optimization as key and session data as value
            
        Example:
        >>> key_dic = {
        ...     "model_name": "A",
        ...     "topP": 0.9
        ... }
        >>> session_data = [
        ...     {
        ...         "model_name": "A",
        ...         "agent_type": "1",
        ...         "topP": 0.9,
        ...         "rounds": [
        ...             {
        ...                 "round": 0,
        ...                     "context": "...",
        ...                 }
        ...             ]
        ...     },
        ...     {
        ...         "model_name": "A",
        ...         "agent_type": "1",
        ...         "topP": 0.9,
        ...         "rounds": [
        ...             {
        ...                 "round": 0,
        ...                     "context": "...",
        ...                 }
        ...             ]
        ...     },
        ...     {
        ...         "model_name": "B",
        ...         "agent_type": "1",
        ...         "topP": 0.9,
        ...         "rounds": [
        ...             {
        ...                 "round": 0,
        ...                     "context": "...",
        ...                 }
        ...             ]
        ...     }
        ... ]
        >>> result = BaseEvaluation._filter_by_key_dic(session_data,key_dic)
        >>> len(result)==2
        True
        """

        try:
            # Filter evaluations based on key_dic and non-empty rounds
            matching_evaluations = [
                sess for sess in session_data
                if all(sess.get(key) == value for key, value in key_dic.items())
                and sess.get('rounds') and len(sess['rounds']) >= min_run_number
            ]


            if not matching_evaluations:
                print(f"No matching sessions with non-empty rounds found for keys: {key_dic}")
                return []
        except Exception as e:
            print(f"❌ Failed to filter sessions: {e}")
            return []
        return matching_evaluations

    def monitor_resources(self):
        # Pamięć
        import psutil
        import platform
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Dodatkowe informacje o urządzeniu
        device_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "machine": platform.machine(),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        }
        
        memory_info = {
            "ram_total_gb": memory.total / (1024 ** 3),
            "ram_used_gb": memory.used / (1024 ** 3),
            "swap_used_gb": swap.used / (1024 ** 3)
        }

        # Energia (na macOS: użycie powermetrics dla CPU/GPU power w mW)
        try:
            import subprocess
            import os
            # Uruchom powermetrics na 1 sekundę i pobierz średnie
            # Użyj -A (non-interactive) żeby nie pytać o hasło jeśli sudo jest skonfigurowane
            cmd = "sudo -n powermetrics -n 1 -i 1000"
            output = subprocess.check_output(cmd, shell=True, text=True, timeout=10, stderr=subprocess.DEVNULL)
            # Parsuj proste metryki (np. CPU power, GPU power)
            cpu_power = None
            gpu_power = None
            
            # Look for CPU Power and GPU Power (case insensitive)
            for line in output.split('\n'):
                if 'CPU Power:' in line and 'mW' in line:
                    try:
                        cpu_power = float(line.split('CPU Power:')[1].split('mW')[0].strip())
                    except (IndexError, ValueError):
                        pass
                        
                if 'GPU Power:' in line and 'mW' in line:
                    try:
                        gpu_power = float(line.split('GPU Power:')[1].split('mW')[0].strip())
                    except (IndexError, ValueError):
                        pass
            energy_info = {
                "cpu_power_mw": cpu_power,
                "gpu_power_mw": gpu_power,
                "total_power_mw": (cpu_power + gpu_power) if cpu_power is not None and gpu_power is not None else (cpu_power if cpu_power is not None else None)
            }
        except subprocess.TimeoutExpired:
            print("⚠️ powermetrics timeout - skipping energy monitoring")
            energy_info = {"cpu_power_mw": None, "gpu_power_mw": None, "total_power_mw": None}
        except subprocess.CalledProcessError as e:
            print(f"⚠️ powermetrics failed (may need sudo password): {e}")
            energy_info = {"cpu_power_mw": None, "gpu_power_mw": None, "total_power_mw": None}
        except Exception as e:
            print(f"⚠️ Energy monitoring error: {e}")
            energy_info = {"cpu_power_mw": None, "gpu_power_mw": None, "total_power_mw": None}

        return {"memory": memory_info, "energy": energy_info, "device": device_info}

    def calculate_resource_differences(self, start_resources, end_resources):
        """Calculate differences between start and end resources."""
        if not start_resources or not end_resources:
            return None
            
        start_memory = start_resources.get('memory', {})
        end_memory = end_resources.get('memory', {})
        start_energy = start_resources.get('energy', {})
        end_energy = end_resources.get('energy', {})
        
        # Calculate memory differences
        memory_diff = {
            "ram_delta_gb": end_memory.get('ram_used_gb', 0) - start_memory.get('ram_used_gb', 0),
            "swap_delta_gb": end_memory.get('swap_used_gb', 0) - start_memory.get('swap_used_gb', 0),
            "ram_start_gb": start_memory.get('ram_used_gb', 0),
            "ram_end_gb": end_memory.get('ram_used_gb', 0),
            "swap_start_gb": start_memory.get('swap_used_gb', 0),
            "swap_end_gb": end_memory.get('swap_used_gb', 0)
        }
        
        # Calculate energy differences (if available)
        start_cpu = start_energy.get('cpu_power_mw')
        end_cpu = end_energy.get('cpu_power_mw')
        start_gpu = start_energy.get('gpu_power_mw')
        end_gpu = end_energy.get('gpu_power_mw')
        
        energy_diff = {
            "cpu_power_start_mw": start_cpu,
            "cpu_power_end_mw": end_cpu,
            "gpu_power_start_mw": start_gpu,
            "gpu_power_end_mw": end_gpu,
            "cpu_power_delta_mw": (end_cpu - start_cpu) if start_cpu is not None and end_cpu is not None else None,
            "gpu_power_delta_mw": (end_gpu - start_gpu) if start_gpu is not None and end_gpu is not None else None
        }
        
        return {"memory": memory_diff, "energy": energy_diff}



