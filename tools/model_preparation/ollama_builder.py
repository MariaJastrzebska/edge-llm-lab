import json
import asyncio
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Type, Dict, Any, List, Optional
import logging
import os
from pydantic import BaseModel, Field
import llama_cpp
import ollama
from datetime import datetime, date
import re
import shutil
import gc
import atexit
from pprint import pprint, pformat
import timeit
import traceback
import coremltools as ct
import numpy as np

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model_preparation.config.config import get_settings
from fed_mobile_chat_kivy.agent.agent_data_collector import MedScreeningSetup, MedScreeningAnalysis, ConstantData, Symptom, FluctuatingData
import yaml
from llm_evaluation_suite.models.ollama_llm import OllamaLLM
from llm_evaluation_suite.tools.medical_tools import SubmitMedicalData

# Niestandardowy encoder dla JSON
def date_handler(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

knowledge_map_data_collector = """
        ELEMENTY OBOWIĄZKOWE:
        1. Dane stale pacjenta (płeć, wzrost, grupa krwi, data urodzenia, imię)
        2. Symptomy pacjenta (objawy, intensywność, czas trwania, częstotliwość, data)
        3. Niektóre dane fluktuujące (glukoza, ciśnienie, stres, aktywność, waga)
        
        Elementy opcjonalne:
        - dane okresowe (np. wyniki badań)
        - dane dietetyczne (np. kalorie, makroskładniki)

        Przeanalizuj wywiad medyczny i:
        1. Jeśli brakuje informacji, określ status jako incomplete i przygotuj pytania o brakujące dane
        2. Jeśli wszystkie informacje są kompletne, określ status jako COMPLETE i przygotuj podsumowanie
        3. Jeśli dane z wywiadu są potwierdzone przez pacjenta, oznacz jako CONFIRMED
        Jeśli to pierwsza rozmowa z pacjentem:
        1. Przywitaj się imieniem użytkownika i poinformuj rozmówcę, że jesteś asystentem wywiadu medycznego oraz że aplikacja będzie zbierać dane o jego stanie zdrowia i proponować wyzwania medyczne w celu poprawy stanu zdrowia.

        TWOJA ODPOWIEDŹ MUSI BYĆ TYLKO POPRAWNYM JSON-EM W FORMIE ZGODNEJ Z OCZEKIWANĄ STRUKTURĄ!
        Nie generuj dialogu, nie powtarzaj pytań, zwracaj tylko strukturę JSON.
        Przykład: {\"status\": \"incomplete\", \"missing_info\": [{\"field\": \"gender\", \"question\": \"Jaka jest Twoja płeć?\"}], \"current_info\": {...}}.

        Przeprowadź rozmowę z pacjentem w formie czatu, aby zebrać pełne dane o jego stanie zdrowia i zwróć odpowiedź w formacie JSON.
        Zwracaj follow-up questions w polu `missing_info`, aby iteracyjnie zbierać dane, dopóki status nie będzie 'complete'.
        Nie ustawiaj statusu 'confirmed' — ten status ustawi pacjent po potwierdzeniu danych.
        Jeśli dane (stałe, objawy, fluktuujące, okresowe) są już podane w `current_info`, nie pytaj o nie ponownie.
        Jeśli proponujesz nowe dane, upewnij się, że są kompletne i logiczne.
        Po każdej odpowiedzi zwracaj pełne dane w `current_info\", aby pacjent mógł je potwierdzić przed zapisem.
"""

class OllamaBuilder:
    def __init__(self, model_config_name: str, verbose: bool = False):
        self.settings = get_settings()
        try:
            self.config = self.settings.get_model_config(model_config_name)
            # Zwiększenie limitu pamięci do 5 GB
            self.config["inference"]["memory_management"]["max_memory_usage"] = "5GB"
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise
        self.verbose = verbose
        self.model = None
        self.platform = self._detect_platform()
        self.use_api = self.config["model"]["use_api"]
        self.model_name = self.config["model"]["name"]
        self.max_input_size = self.config["platforms"][self.platform]["metadata"]["input_size"][1]
        self.max_memory_usage = self._parse_memory_limit(self.config["inference"]["memory_management"]["max_memory_usage"])
        self.gc_interval = self.config["inference"]["memory_management"]["gc_interval"] / 1000
        self.cleanup_on_exit = self.config["inference"]["memory_management"]["cleanup_on_exit"]
        atexit.register(self.cleanup)
        self.model_name_short = self.config["model"]["name_short"]

    def _parse_memory_limit(self, limit_str: str) -> int:
        value = float(limit_str.replace('GB', '').replace('MB', ''))
        if self.verbose:
            print(f"Memory limit: {value} GB")
        return int(value * 1024 * 1024 * 1024)

    def _check_memory_usage(self):
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss
        if self.verbose:
            print(f"Memory usage: {memory_usage / 1024 / 1024:.2f} MB / {self.max_memory_usage / 1024 / 1024:.2f} MB")
        if memory_usage > self.max_memory_usage:
            logger.warning(f"Użycie pamięci ({memory_usage / 1024 / 1024:.2f} MB) przekroczyło limit ({self.max_memory_usage / 1024 / 1024:.2f} MB). Wywołanie GC.")
            gc.collect()

    def initialize_model(self):
        if self.verbose:
            print("Initializing model...")
        try:
            platform_config = self.config["platforms"][self.platform]
            model_config = {
                "model_path": platform_config["paths"]["llama_model"],
                "n_ctx": self.config["model_specific"]["context"]["max_length"],
                "n_batch": 512,
                "quantization": self.config["model_specific"]["quantization"]["type"],
                "n_threads": self.config["inference"]["n_threads"],
                "use_metal": True,
                "n_gpu_layers": 32
            }
            self._load_llama_model(**model_config)
            if self.verbose:
                print("Model initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            return False

    def _detect_platform(self) -> str:
        if sys.platform == "darwin":
            return "darwin"
        elif sys.platform == "ios":
            return "ios"
        elif sys.platform == "android":
            return "android"
        else:
            raise ValueError("Unsupported platform")

    def _load_llama_model(self, model_path, n_ctx, n_batch, quantization, n_threads, use_metal, n_gpu_layers):
        if self.verbose:
            print(f"Loading model from {model_path} with n_ctx={n_ctx}, quantization={quantization}")
        if self.model is None and not self.use_api:
            try:
                self.model = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_batch=n_batch,
                    n_threads=n_threads,
                    use_metal=use_metal,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                logger.info(f"Loaded llama.cpp model from {model_path} with n_ctx={n_ctx}, quantization={quantization}")
            except Exception as e:
                logger.error(f"Failed to load llama.cpp model: {str(e)}")
                raise
        return self.model

    def chunk_text(self, text: str, use_chunking: bool = True, chunk_size: int = 500) -> List[str]:
        if self.verbose:
            print("Chunking text...")
        if not use_chunking:
            return [text.encode('utf-8').decode('utf-8')]

        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        word_count = 0

        for paragraph in paragraphs:
            words = paragraph.split()
            if word_count + len(words) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk).encode('utf-8').decode('utf-8'))
                current_chunk = []
                word_count = 0
            current_chunk.extend(words)
            word_count += len(words)

        if current_chunk:
            chunks.append(" ".join(current_chunk).encode('utf-8').decode('utf-8'))

        if self.verbose:
            print(f"Generated chunks: {chunks}")
        return chunks

    def process_input_with_chunks(self, user_input: str, accumulated_response: Dict, system_prompt: str, user_prefix: str, chunk_size: int = 500) -> Dict:
        if self.verbose:
            print("Processing input with chunks...")
        chunks = self.chunk_text(user_input, use_chunking=True, chunk_size=chunk_size)
        context = f"{system_prompt}"

        for i, chunk in enumerate(chunks):
            current_state = json.dumps(accumulated_response, ensure_ascii=False, default=date_handler)
            formatted_user = f"{user_prefix}{chunk}\nAktualny stan: {current_state}"
            full_prompt = context + formatted_user

            try:
                self._check_memory_usage()
                if self.use_api:
                    if self.verbose:
                        print(f"Sending API request for chunk {i + 1}")
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": formatted_user}
                        ],
                        options={
                            "temperature": self.config["inference"]["temperature"],
                            "top_p": self.config["inference"]["top_p"],
                            "num_predict": self.config["inference"]["max_new_tokens"],
                            "stop": self.config["inference"]["stop_sequences"]
                        }
                    )
                    response_text = response["message"]["content"].encode('utf-8').decode('utf-8')
                else:
                    if self.model is None:
                        self.initialize_model()
                    if self.verbose:
                        print(f"Processing chunk {i + 1} with model")
                    output = self.model(
                        full_prompt[:self.max_input_size],
                        max_tokens=self.config["inference"]["max_new_tokens"],
                        temperature=self.config["inference"]["temperature"],
                        top_p=self.config["inference"]["top_p"],
                        stop=self.config["inference"]["stop_sequences"]
                    )
                    response_text = output['choices'][0]['text'].encode('utf-8').decode('utf-8')
                    self.model.__del__()

                response_text = response_text.strip()
                if self.verbose:
                    print(f"Response for chunk {i + 1}: {response_text}")
                if response_text.startswith("{"):
                    try:
                        chunk_response = json.loads(response_text)
                        for key, value in chunk_response.items():
                            if key in accumulated_response:
                                if isinstance(accumulated_response[key], list) and isinstance(value, list):
                                    accumulated_response[key].extend(value)
                                else:
                                    accumulated_response[key] = value
                            else:
                                accumulated_response[key] = value
                        context += formatted_user + f"\n<|start_header_id|>assistant<|end_header_id|>\n\n{response_text}"
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in chunk {i + 1}: {response_text}")
            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                continue

        return accumulated_response

    async def model_call(self, user_input: str, verbose: bool = False, system_prompt: str = None, response_model: Optional[Type[BaseModel]] = None, use_chunking: bool = True, chunk_size: int = 500) -> Any:
        if self.verbose:
            print("Model call initiated...")
        if system_prompt is None:
            system_prompt = knowledge_map_data_collector

        system_prompt = system_prompt.encode('utf-8').decode('utf-8')
        user_input = user_input.encode('utf-8').decode('utf-8')

        # Pobierz chat_format i stop_sequences z konfiguracji
        chat_format = self.config["inference"]["chat_format"]
        stop_sequences = self.config["inference"]["stop_sequences"]

        # Definiowanie prefixów na podstawie chat_format
        if chat_format == "llama-3":
            system_prefix = "<|start_header_id|>system<|end_header_id|>\n\n"
            user_prefix = "<|start_header_id|>user<|end_header_id|>\n\n"
            stop_sequences = [seq.strip(":") for seq in stop_sequences]  # ["Human", "Assistant"]
        else:
            system_prefix = "System: "
            user_prefix = "User: "
            stop_sequences = ["\n"]
        
        # Inicjalizacja accumulated_response jako pusty dict
        accumulated_response = {}

        # Przetwarzanie z użyciem chunkingu lub bez
        start_time = timeit.default_timer()
        if use_chunking:
            if self.verbose:
                print("Processing with chunking...")
            accumulated_response = self.process_input_with_chunks(user_input, accumulated_response, system_prompt, user_prefix, chunk_size)
        else:
            if self.verbose:
                print("Processing without chunking...")
            current_state = json.dumps(accumulated_response, ensure_ascii=False, default=date_handler)
            formatted_user = f"{user_prefix}{user_input}\nAktualny stan: {current_state}"
            full_prompt = f"{system_prefix}{system_prompt}{formatted_user}"

            try:
                self._check_memory_usage()
                if self.use_api:
                    if self.verbose:
                        print("Sending API request...")
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": formatted_user}
                        ],
                        options={
                            "temperature": self.config["inference"]["temperature"],
                            "top_p": self.config["inference"]["top_p"],
                            "num_predict": self.config["inference"]["max_new_tokens"],
                            "stop": stop_sequences
                        }
                    )
                    response_text = response["message"]["content"].encode('utf-8').decode('utf-8')
                else:
                    if self.model is None:
                        self.initialize_model()
                    if self.verbose:
                        print("Processing with model...")
                    output = self.model(
                        full_prompt[:self.max_input_size],
                        max_tokens=self.config["inference"]["max_new_tokens"],
                        temperature=self.config["inference"]["temperature"],
                        top_p=self.config["inference"]["top_p"],
                        stop=stop_sequences
                    )
                    response_text = output['choices'][0]['text'].encode('utf-8').decode('utf-8')
                    self.model.__del__()

                response_text = response_text.strip()
                if self.verbose:
                    print(f"Raw response text: {response_text}")
                if response_text.startswith("{"):
                    try:
                        chunk_response = json.loads(response_text)
                        for key, value in chunk_response.items():
                            if key in accumulated_response:
                                if isinstance(accumulated_response[key], list) and isinstance(value, list):
                                    accumulated_response[key].extend(value)
                                else:
                                    accumulated_response[key] = value
                            else:
                                accumulated_response[key] = value
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {response_text}")
            except Exception as e:
                logger.error(f"Error processing input: {str(e)}")

        final_response_text = json.dumps(accumulated_response, ensure_ascii=False)
        elapsed_time = timeit.default_timer() - start_time

        print("\n=== Raw Response ===")
        try:
            parsed_json = json.loads(final_response_text)
            print(pformat(parsed_json, width=80, sort_dicts=False))
        except json.JSONDecodeError:
            print(final_response_text)

        if verbose:
            input_length = len(user_input.split())
            output_length = len(final_response_text.split())
            print(f"\nDługość wejścia: {input_length} słów")
            print(f"Długość wyjścia: {output_length} słów")
            print(f"Czas wykonania: {elapsed_time:.4f} sekund")

        try:
            if response_model:
                parsed_response = response_model.model_validate_json(final_response_text)
                if verbose:
                    print("\n=== Processed Response ===")
                    pprint(parsed_response.model_dump(), width=80, sort_dicts=False)
                return parsed_response
        except ValueError as e:
            logger.error(f"Failed to parse response with {response_model.__name__}: {str(e)}")
            return {"error": str(e), "raw_response": final_response_text}

        except Exception as e:
            logger.error(f"Error in model call: {str(e)}")
            return final_response_text

    def cleanup(self):
        logger.info("Cleaning up OllamaBuilder...")
        if self.model:
            self.model.__del__()
            self.model = None
        gc.collect()


    def generate_agent_prompt(self, user_input: str, knowledge_map_data_collector: str, current_info: Optional[MedScreeningSetup] = None) -> str:
        current_info_json = json.dumps(current_info.model_dump() if current_info else {}, default=date_handler, indent=2, ensure_ascii=False) if current_info else ""
        return f"""
{knowledge_map_data_collector}

Data rozmowy: {DateTime.now().toIso8601String().strftime('%Y-%m-%d %H:%M:%S')}
Aktualne dane: {current_info_json}
Użytkownik: {user_input}
"""
    

      
    def evaluate_tool_call(self, user_query: str, tools_to_evaluate: Dict[str, Type[BaseModel]]) -> Dict:
        """Ocenia zdolność modelu do wywołania narzędzia na podstawie zapytania użytkownika."""
        print("--- Starting Tool Call Evaluation ---")
        print(f"Test Scenario: User query is \"{user_query}\"")

        # Używamy dedykowanej klasy do budowy promptu i wywołania
        # Możesz dostosować model używany w config.py w llm_evaluation_suite
        llm = OllamaLLM(model=self.model_name)

        # Wywołaj model z zapytaniem i dostępnymi narzędziami
        response = llm.invoke(user_query, tools_to_evaluate)

        print("\n--- Evaluation Result ---")
        print(f"Parsed Response: {json.dumps(response, indent=2)}")

        # Walidacja odpowiedzi
        if 'error' in response:
            result = {"status": "FAILED", "reason": "Error during LLM call", "details": response.get('error')}
        elif 'tool_calls' in response:
            try:
                tool_call_data = response['tool_calls'][0]['function']
                tool_name = tool_call_data['name']
                tool_args = tool_call_data['arguments']

                if tool_name in tools_to_evaluate:
                    # Walidacja argumentów za pomocą Pydantic
                    tools_to_evaluate[tool_name](**tool_args)
                    result = {"status": "SUCCESS", "tool_name": tool_name, "arguments": tool_args}
                else:
                    result = {"status": "FAILED", "reason": f"Model called an unknown tool '{tool_name}'."}
            except Exception as e:
                result = {"status": "FAILED", "reason": "Tool arguments validation failed or malformed structure", "details": str(e)}
        elif 'response' in response:
            result = {"status": "INFO", "reason": "Model chose not to use a tool.", "response": response.get('response')}
        else:
            result = {"status": "FAILED", "reason": "Response is not in the expected format."}
        
        print(f"\n[RESULT] {result['status']}: {result.get('reason', '')}")
        return result

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Inicjalizacja OllamaBuilder...")
    try:
        builder = OllamaBuilder("medexpert", verbose=True)
        if builder.initialize_model():
            print("Model initialized successfully")
        else:
            print("Failed to initialize model")
            sys.exit(1)
    except Exception as e:
        print(f"Błąd inicjalizacji: {e}")
        traceback.print_exc()
        sys.exit(1)

    class PatientData(BaseModel):
        symptoms: List[str] = Field(..., description="Lista objawów zgłaszanych przez pacjenta")
        medical_history: List[str] = Field(..., description="Historia choroby pacjenta")
        test_results: Optional[Dict[str, str]] = Field(None, description="Wyniki badań pacjenta")
        medications: Optional[List[str]] = Field(None, description="Lista przyjmowanych leków")

    formatted_system_prompt = """Otrzymasz opis objawów pacjenta, jego historii choroby i wyników badań.

    WAŻNE: Nie wymyślaj żadnych informacji! Użyj WYŁACZNIE danych podanych w opisie pacjenta.
    Jeśli informacja nie jest wprost zawarta w opisie, NIE DODAWAJ jej.
    
    Twoim zadaniem jest stworzenie poprawnego JSON-a zgodnego z modelem Pydantic:
    - symptoms: tylko wymienione objawy, dosłownie jak w tekście
    - medical_history: tylko faktyczne choroby wymienione w tekście
    - test_results: tylko wyniki badań podane w tekście (jako pary klucz-wartość)
    - medications: tylko leki wymienione w tekście
    
    Jeśli jakiejś informacji nie ma w tekście, użyj pustej listy [] lub obiektu {} dla tego pola.
    Trzymaj się dosłownie tekstu - nie interpretuj, nie zgaduj, nie dodawaj.
    WYNIK MUSI BYĆ POPRAWNYM JSON-EM, np. {\"symptoms\": [\"osłabienie\"], \"medical_history\": [\"cukrzyca typu 2\"], ...}"""

    user_prompt = """Od jakiegos czasu czuję się osłabiony, zwłaszcza rano.   
    Mam cukrzycę typu 2 od kilku lat, biorę metforminę, ale ostatnio gorzej się czuję. 
    W wynikach badań ostatnio wyszedł mi podwyższony poziom hemoglobiny glikowanej, a lekarz wspominał coś o możliwości zmiany leczenia.    
    Mam mniej energii i czuję, że coś jest nie tak."""

    # Testy
    async def test_no_response_model_chunking():
        print("\n=== Test bez response_model z chunkingiem ===")
        input_length = len(user_prompt.split())
        response = await builder.model_call(
            user_input=user_prompt,
            verbose=True,
            system_prompt=formatted_system_prompt,
            use_chunking=True,
            chunk_size=10
        )
        if isinstance(response, dict) and "error" in response:
            print(f"Surowa odpowiedź: {response['raw_response']}")
        else:
            output_length = len(response.split()) if isinstance(response, str) else len(json.dumps(response, ensure_ascii=False).split())
            print(f"Długość wejścia: {input_length} słów")
            print(f"Długość wyjścia: {output_length} słów")
            print(f"Surowa odpowiedź: {response}")

    async def test_no_response_model_no_chunking():
        print("\n=== Test bez response_model bez chunkingiem ===")
        input_length = len(user_prompt.split())
        response = await builder.model_call(
            user_input=user_prompt,
            verbose=True,
            system_prompt=formatted_system_prompt,
            use_chunking=False
        )
        if isinstance(response, dict) and "error" in response:
            print(f"Surowa odpowiedź: {response['raw_response']}")
        else:
            output_length = len(response.split()) if isinstance(response, str) else len(json.dumps(response, ensure_ascii=False).split())
            print(f"Długość wejścia: {input_length} słów")
            print(f"Długość wyjścia: {output_length} słów")
            print(f"Surowa odpowiedź: {response}")

    async def test_with_patient_data_chunking():
        print("\n=== Test z PatientData z chunkingiem ===")
        input_length = len(user_prompt.split())
        response = await builder.model_call(
            user_input=user_prompt,
            verbose=True,
            system_prompt=formatted_system_prompt,
            response_model=PatientData,
            use_chunking=True,
            chunk_size=10
        )
        if isinstance(response, dict) and "error" in response:
            print(f"Błąd: {response['error']}")
            print(f"Surowa odpowiedź: {response['raw_response']}")
        else:
            output_length = len(json.dumps(response.model_dump(), ensure_ascii=False).split())
            print(f"Długość wejścia: {input_length} słów")
            print(f"Długość wyjścia: {output_length} słów")
            print(f"Ostateczny JSON: {json.dumps(response.model_dump(), indent=2, ensure_ascii=False)}")

    async def test_with_patient_data_no_chunking():
        print("\n=== Test z PatientData bez chunkingiem ===")
        input_length = len(user_prompt.split())
        response = await builder.model_call(
            user_input=user_prompt,
            verbose=True,
            system_prompt=formatted_system_prompt,
            response_model=PatientData,
            use_chunking=False
        )
        if isinstance(response, dict) and "error" in response:
            print(f"Błąd: {response['error']}")
            print(f"Surowa odpowiedź: {response['raw_response']}")
        else:
            output_length = len(json.dumps(response.model_dump(), ensure_ascii=False).split())
            print(f"Długość wejścia: {input_length} słów")
            print(f"Długość wyjścia: {output_length} słów")
            print(f"Ostateczny JSON: {json.dumps(response.model_dump(), indent=2, ensure_ascii=False)}")

    async def test_with_agent_prompt_chunking():
        print("\n=== Test z agentem z chunkingiem ===")
        current_info = MedScreeningSetup(
            constant_data=ConstantData(reported_date=date.today(), name="Jan"),
            symptoms=[Symptom(name="osłabienie")],
            daily_data=FluctuatingData(stress_level=5)
        )
        input_length = len(user_prompt.split())
        response = await builder.model_call(
            user_input=user_prompt,
            verbose=True,
            system_prompt=builder.generate_agent_prompt(user_input=user_prompt, current_info=current_info, knowledge_map_data_collector=knowledge_map_data_collector),
            response_model=MedScreeningAnalysis,
            use_chunking=True,
            chunk_size=10
        )
        if isinstance(response, dict) and "error" in response:
            print(f"Błąd: {response['error']}")
            print(f"Surowa odpowiedź: {response['raw_response']}")
        else:
            output_length = len(json.dumps(response.model_dump(), ensure_ascii=False).split())
            print(f"Długość wejścia: {input_length} słów")
            print(f"Długość wyjścia: {output_length} słów")
            print(f"Ostateczny JSON: {json.dumps(response.model_dump(), indent=2, ensure_ascii=False)}")

    async def test_with_agent_prompt_no_chunking():
        print("\n=== Test z agentem bez chunkingiem ===")
        current_info = MedScreeningSetup(
            constant_data=ConstantData(reported_date=date.today(), name="Jan"),
            symptoms=[Symptom(name="osłabienie")],
            daily_data=FluctuatingData(stress_level=5)
        )
        input_length = len(user_prompt.split())
        response = await builder.model_call(
            user_input=user_prompt,
            verbose=True,
            system_prompt=builder.generate_agent_prompt(user_input=user_prompt, current_info=current_info, knowledge_map_data_collector=knowledge_map_data_collector),
            response_model=MedScreeningAnalysis,
            use_chunking=False
        )
        if isinstance(response, dict) and "error" in response:
            print(f"Błąd: {response['error']}")
            print(f"Surowa odpowiedź: {response['raw_response']}")
        else:
            output_length = len(json.dumps(response.model_dump(), ensure_ascii=False).split())
            print(f"Długość wejścia: {input_length} słów")
            print(f"Długość wyjścia: {output_length} słów")
            print(f"Ostateczny JSON: {json.dumps(response.model_dump(), indent=2, ensure_ascii=False)}")

    try:
        # Dictionary to store timing results
        timing_results = {}

        print("\n=== Performance Tests ===")
        
        # Test bez response_model z chunkingiem
        start_time = timeit.default_timer()
        asyncio.run(test_no_response_model_chunking())
        timing_results['no_response_chunking'] = timeit.default_timer() - start_time
        print(f"Completed test: no_response_chunking in {timing_results['no_response_chunking']:.4f}s")
        
        # Test bez response_model bez chunkingiem
        start_time = timeit.default_timer()
        asyncio.run(test_no_response_model_no_chunking())
        timing_results['no_response_no_chunking'] = timeit.default_timer() - start_time
        print(f"Completed test: no_response_no_chunking in {timing_results['no_response_no_chunking']:.4f}s")
        
        # Test z PatientData z chunkingiem
        start_time = timeit.default_timer()
        asyncio.run(test_with_patient_data_chunking())
        timing_results['patient_data_chunking'] = timeit.default_timer() - start_time
        print(f"Completed test: patient_data_chunking in {timing_results['patient_data_chunking']:.4f}s")
        
        # Test z PatientData bez chunkingiem
        start_time = timeit.default_timer()
        asyncio.run(test_with_patient_data_no_chunking())
        timing_results['patient_data_no_chunking'] = timeit.default_timer() - start_time
        print(f"Completed test: patient_data_no_chunking in {timing_results['patient_data_no_chunking']:.4f}s")
        
        # Test z agentem z chunkingiem
        start_time = timeit.default_timer()
        asyncio.run(test_with_agent_prompt_chunking())
        timing_results['agent_prompt_chunking'] = timeit.default_timer() - start_time
        print(f"Completed test: agent_prompt_chunking in {timing_results['agent_prompt_chunking']:.4f}s")

        # Test ewaluacji wywołania narzędzia
        async def test_tool_calling_evaluation():
            print("\n--- Uruchamianie testu ewaluacji wywołania narzędzi ---")
            # Użyj swojej konfiguracji modelu, np. 'granite-3.1-dense-16k-gguf'
            builder = OllamaBuilder(model_config_name="granite-3.1-dense-16k-gguf", verbose=True)
            
            test_query = "The patient is John Doe, he is 45 years old. He complains about a headache and fever. This is an urgent case."
            
            # Tutaj możesz podmienić `SubmitMedicalData` na swoje modele Pydantic
            # np. z fed_mobile_chat_kivy.agent.agent_data_collector
            tools_for_test = {"send_medical_data": SubmitMedicalData}
            
            evaluation_result = builder.evaluate_tool_call(user_query=test_query, tools_to_evaluate=tools_for_test)
            
            print("\n--- Końcowy wynik ewaluacji ---")
            pprint(evaluation_result)

        # Uruchomienie nowego testu
        asyncio.run(test_tool_calling_evaluation())
        
        # Test z agentem bez chunkingiem
        start_time = timeit.default_timer()
        asyncio.run(test_with_agent_prompt_no_chunking())
        timing_results['agent_prompt_no_chunking'] = timeit.default_timer() - start_time
        print(f"Completed test: agent_prompt_no_chunking in {timing_results['agent_prompt_no_chunking']:.4f}s")

        # Print performance summary
        print("\n=== Performance Summary ===")
        print("Timing results (in seconds):")
        for test_name, duration in timing_results.items():
            print(f"{test_name:20}: {duration:.4f}s")
        
        # Calculate and print comparisons
        base_time = min(timing_results.values())  # Używamy najszybszego testu jako bazy
        print("\nRelative performance (compared to fastest test):")
        for test_name, duration in timing_results.items():
            relative_perf = (duration / base_time - 1) * 100
            print(f"{test_name:20}: {'+' if relative_perf > 0 else ''}{relative_perf:.1f}%")

        # Find fastest and slowest
        fastest = min(timing_results.items(), key=lambda x: x[1])
        slowest = max(timing_results.items(), key=lambda x: x[1])
        print(f"\nFastest test: {fastest[0]} ({fastest[1]:.4f}s)")
        print(f"Slowest test: {slowest[0]} ({slowest[1]:.4f}s)")
        print(f"Performance difference: {((slowest[1]/fastest[1])-1)*100:.1f}%")

    except Exception as e:
        print(f"Błąd podczas testów: {e}")
        traceback.print_exc()
        sys.exit(1)