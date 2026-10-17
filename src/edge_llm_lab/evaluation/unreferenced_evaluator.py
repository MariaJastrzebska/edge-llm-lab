from edge_llm_lab.evaluation.dataset import DatasetEvaluation
from edge_llm_lab.visualization.unreferenced_plots import UnreferencedPlots
import ollama
import openai
import pandas as pd
import numpy as np
import time
import json
import csv
from typing import List, Dict, Any, Union, Optional, Literal, Tuple
import os
from datetime import datetime
import matplotlib.pyplot as plt
from enum import Enum
import sys
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath("src"))

from edge_llm_lab.utils.base_eval import BaseEvaluation, Agent
import numpy as np
import glob
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv()
    print(" Environment variables loaded from .env")
except ImportError:
    print(" python-dotenv not installed, using system environment variables")




class EvalModelsUnreferenced(DatasetEvaluation, UnreferencedPlots, BaseEvaluation):
    class RoundNumber(Enum):
        ZERO = 0
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5

    def __init__(self, model_name, agent, eval_type="unreferenced"):
        """Initialize EvalModelsUnreferenced with model name, agent, and optional source path.
        """
        super().__init__(model_name, agent, eval_type)
        self.eval_type = eval_type
        self._finalize_init(model_name=model_name, agent=agent, eval_type=self.eval_type)

        # # Initialize attributes that may be accessed before pipeline_eval_model
        self.optimisation = [{
                "--flash-attn": None,
                "--cont-batching": None
        }]
        self.tools = self.create_tool_for_agent()
        self.cot_prompt = self.read_txt(self.MULTI_TURN_GLOBAL_CONFIG.get('cot_prompt_path'))

        self.patient_dir = os.path.join(
            self.SOURCE_PATH, "input", "agents", self.agent_type,
            "conversation_stimulation"
        )

    # ========================================
    # HELPER METHODS
    # ========================================

    def evaluate_case(self, case, optimisation, use_cache=False):
        return self.run_functional_test_with_patient(
            case['patient_file'], case['expected_file'], optimisation=optimisation,
            use_cache=use_cache, max_rounds=case.get('max_rounds', 10))

    def _find_in_nested_dict(self, obj, keys):
        """Rekurencyjnie szuka wartości dla podanych kluczy w zagnieżdżonej strukturze"""
        if not isinstance(obj, (dict, list)):
            return None

        if isinstance(obj, dict):
            for key in keys:
                if key in obj:
                    return obj[key]
            # Szukaj rekurencyjnie
            for value in obj.values():
                result = self._find_in_nested_dict(value, keys)
                if result is not None:
                    return result

        elif isinstance(obj, list):
            for item in obj:
                result = self._find_in_nested_dict(item, keys)
                if result is not None:
                    return result

        return None

    def _load_patient_simulation_prompt(self, patient_context, doctor_question):
        """Load patient simulation prompt from file based on agent type"""
        # Determine agent folder name
        if hasattr(self, 'agent'):
            agent_folder = self.agent.value
        else:
            agent_folder = 'constant_data'  # fallback

        # Build path to patient simulation prompt
        base_path = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(self.SOURCE_PATH, 'input', "agents", agent_folder, "conversation_stimulation", "patient_simulation_prompt.txt")

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()

            # Format with variables
            return prompt_template.format(
                patient_context=patient_context,
                doctor_question=doctor_question
            )
        except FileNotFoundError:
            print(f"  Patient simulation prompt not found: {prompt_file}")
            # Fallback to Polish prompt
            return ""

    def extract_status_and_question(self, llm_response: str) -> Tuple[str, str, Dict]:
        """Ekstraktuje status, pytanie i dane z odpowiedzi modelu - elastycznie"""
        try:
            response_data = json.loads(llm_response)

            # Szukaj statusu - może być w różnych miejscach
            status_raw = self._find_in_nested_dict(response_data,
                ['status', 'MedScreeningInputStatus', 'value', 'completion_status'])

            # Normalizuj status
            status = 'INCOMPLETE'
            if status_raw:
                status_str = str(status_raw).lower()
                if status_str.strip() == 'complete':
                    status = 'COMPLETE'
                else:
                    status = 'INCOMPLETE'

            # Szukaj pytania - może być w różnych miejscach
            doctor_question = self._find_in_nested_dict(response_data,
                ['question', 'doctor_question', 'next_question'])

            # Jeśli nie znaleziono, szukaj w missing_info
            if not doctor_question:
                missing_info = self._find_in_nested_dict(response_data, ['missing_info'])
                if missing_info and isinstance(missing_info, dict):
                    doctor_question = missing_info.get('question', '')

            doctor_question = str(doctor_question) if doctor_question else ''

            # Szukaj zebranych danych
            current_info = self._find_in_nested_dict(response_data,
                ['current_info', 'collected_data', 'current_data', 'data'])

            if not isinstance(current_info, dict):
                current_info = {}

            print(f"   Parsed - Status: {status}, Question: '{doctor_question}'")

            return status, doctor_question, current_info

        except Exception as e:
            print(f"    Error parsing response: {e} from {llm_response}")

        return "INCOMPLETE", "", {}

    # ========================================
    # PLOTTING FUNCTIONS (analogiczne do referenced)
    # ========================================




    # ========================================
    # PATIENT SIMULATION FUNCTIONS
    # ========================================

    def simulate_patient_response(self, patient_context, doctor_question, use_cache=False):
        """
        Symuluje odpowiedź pacjenta na pytanie lekarza używając GPT.

        Args:
            patient_context (str): Kontekst/informacje o pacjencie
            doctor_question (str): Pytanie zadane przez lekarza (model)
            round_number (int): Numer rundy (do context)

        Returns:
            str: Odpowiedź pacjenta lub None w przypadku błędu
        """
        if not self.OPENAI_CLIENT:
            print("OpenAI client not available for patient simulation")
            raise RuntimeError('Patient simulation requires an API client')

        # Load patient simulation prompt from file based on agent type
        patient_prompt = self._load_patient_simulation_prompt(patient_context, doctor_question)
        # tools_schema = self.create_tool_for_agent()
        # print(f"Patient stimulation prompt: {patient_prompt}")

        try:
            messages = [{"role": "user", "content": patient_prompt}]
            # print(f"Messages for chat gpt : {messages}")
            patient_response = self.get_openai_response(
                    custom_catch_type="patient_response_stimulation",
                    messages=messages,
                    use_cache=use_cache,
                )

            # print(f"Patient response: {patient_response}")
            if isinstance(patient_response, dict):
                patient_response = patient_response["content"]
            elif isinstance(patient_response, str):
                patient_response = patient_response
            else:
                patient_response = str(patient_response)
            patient_response = patient_response.strip()
            # print(f"Patient response: {patient_response}")

            return patient_response
        except Exception as e:
            print(f"Error in patient simulation: {e}")
            raise RuntimeError('Patient simulator failed') from e

    def evaluate_single_patient_round(
        self,
        round_num: int,
        patient_context: str,
        conversation_history: List[Dict],
        collected_data: Dict,
        tools_schema: Dict,
        optimisation: Dict = {},
        use_cache: bool = False
    ) -> Dict:
        """Wykonuje pojedynczą rundę rozmowy z pacjentem"""

        print(f"\n  📝 Round {round_num}")
        start_time = time.time()

        # Użyj conversation_history jako messages bezpośrednio
        # conversation_history już zawiera role: system/user/assistant

        # Wywołaj model z messages z optymalizacjami
        llm_response, latency_breakdown = self.run_inference(
            tools_schema, conversation_history, optimisations=optimisation  # Przekazujemy optimisations
        )
        duration = time.time() - start_time
        # Parsuj odpowiedź
        status, doctor_question, new_collected_data = self.extract_status_and_question(
            llm_response
        )

        # Jeśli jest pytanie, symuluj odpowiedź pacjenta
        patient_response = ""
        if doctor_question and status != "COMPLETE":
            patient_response = self.simulate_patient_response(
                patient_context,
                doctor_question,
                use_cache=use_cache
            )
            print(f"     🤖 Doctor: {doctor_question}")
            print(f"     👤 Patient: {patient_response}")



        return {
            "round": round_num,
            "model_response": llm_response,
            "doctor_question": doctor_question,
            "patient_response": patient_response,
            "status": status,
            "collected_data": new_collected_data,
            "latency_breakdown": latency_breakdown,
            "duration_sec": duration
        }

    def run_functional_test_with_patient(
        self,
        patient_file: str,
        expected_file: str,
        optimisation: Dict = {},
        max_rounds: int = 10,
        use_cache: bool = False
    ) -> Dict:
        """Przeprowadza pełny test funkcjonalny z jednym pacjentem"""

        patient_name = os.path.basename(patient_file).replace('.txt', '')
        print(f"\n🏥 Testing with patient: {patient_name}")

        # Wczytaj kontekst pacjenta
        with open(patient_file, 'r', encoding='utf-8') as f:
            patient_context = f.read()

        # Wczytaj oczekiwane wyniki
        expected_data = {}
        if os.path.exists(expected_file):
            with open(expected_file, 'r', encoding='utf-8') as f:
                expected_data = json.load(f)

        # Rozpocznij z podstawowym system promptem
        conversation_history = [
            {"role": "system", "content": self.read_txt(self.MULTI_TURN_GLOBAL_CONFIG.get('cot_prompt_path'))},
            {"role": "user", "content": "Rozpocznij Zbieranie Danych"}
        ]
        collected_data = {}
        rounds_data = []
        completed = False

        # Przygotuj tools schema dla agenta
        tools_schema = self.create_tool_for_agent()

        # Wykonuj rundy
        for round_num in range(1, max_rounds + 1):
            round_result = self.evaluate_single_patient_round(
                round_num,
                patient_context,
                conversation_history,
                collected_data,
                tools_schema,
                optimisation=optimisation,
                use_cache=use_cache
            )

            rounds_data.append(round_result)

            # Aktualizuj dane
            if round_result["collected_data"]:
                collected_data = round_result["collected_data"]

            # Aktualizuj historię konwersacji
            conversation_history.append({
                "role": "assistant",
                "content": round_result["model_response"]
            })

            # Dodaj odpowiedź pacjenta jeśli była
            if round_result["patient_response"]:
                conversation_history.append({
                    "role": "user",
                    "content": round_result["patient_response"]
                })

            # Sprawdź czy zakończono
            if round_result["status"] == "COMPLETE":
                completed = True
                print(f"   Completed in {round_num} rounds")
                break

        # Określ typ pacjenta i wynik testu
        patient_type = expected_data.get("patient_type", "successful")
        test_result = self.evaluate_test_result(
            patient_type,
            completed,
            collected_data,
            expected_data
        )

        return {
            "patient_name": patient_name,
            "patient_type": patient_type,
            "expected_data": expected_data,
            "collected_data": collected_data,
            "status": "COMPLETE" if completed else "INCOMPLETE",
            "test_result": test_result,
            "rounds_count": len(rounds_data),
            "rounds_data": rounds_data,
            "completed_successfully": completed,
            "total_duration_sec": sum(r["duration_sec"] for r in rounds_data)
        }

    def evaluate_test_result(
        self,
        patient_type: str,
        completed: bool,
        collected_data: Dict,
        expected_data: Dict
    ) -> str:
        """Ocenia wynik testu na podstawie typu pacjenta"""

        if patient_type == "successful":
            if not completed:
                return "timeout"

            # Sprawdź zgodność danych
            expected_fields = expected_data.get("expected_data", {})
            if not expected_fields:
                return "false_complete"

            matched_fields = 0
            total_fields = len(expected_fields)

            for field, expected_value in expected_fields.items():
                if field in collected_data:
                    # Proste porównanie dla pierwszego agenta
                    if str(collected_data[field]).lower() == str(expected_value).lower():
                        matched_fields += 1

            match_ratio = matched_fields / total_fields if total_fields > 0 else 0

            if match_ratio >= 0.9:
                return "perfect_success"
            elif match_ratio >= 0.5:
                return "partial_success"
            else:
                return "false_complete"

        else:  # unsuccessful patient
            if completed:
                return "false_success"
            else:
                return "correct_rejection"

    # ========================================
    # EVALUATION FUNCTIONS (jak w referenced)
    # ========================================

    def evaluate_all_patients(
        self,
        session_data: Dict,
        log_file: str,
        model_run_folder: str,
        timestamp: str,
        run_number: str,
        optimisation: Dict = {},
        use_cache: bool = False
    ) -> Dict:
        """Ewaluuje wszystkich pacjentów dla modelu (jak evaluate_all_rounds w referenced)"""

        print(f"\n  Starting unreferenced evaluation for {self.model_name}")

        # Pobierz pliki pacjentów
        successful_dir = os.path.join(self.patient_dir, "successful")
        unsuccessful_dir = os.path.join(self.patient_dir, "unsuccessful")

        patient_files = []

        # Dodaj successful patients
        if os.path.exists(successful_dir):
            files = sorted(glob.glob(os.path.join(successful_dir, "*.txt")))  # Sortuj alfabetycznie
            for file in files:
                if "expected" not in file:
                    expected_file = file.replace(".txt", "_expected.json")
                    patient_files.append((file, expected_file, "successful"))

        # Dodaj unsuccessful patients
        if os.path.exists(unsuccessful_dir):
            files = sorted(glob.glob(os.path.join(unsuccessful_dir, "*.txt")))  # Sortuj alfabetycznie
            for file in files:
                if "expected" not in file:
                    expected_file = file.replace(".txt", "_expected.json")
                    patient_files.append((file, expected_file, "unsuccessful"))

        if not patient_files:
            print("No patient files found!")
            return {}

        print(f"📁 Found {len(patient_files)} patient files")

        # Wyświetl listę pacjentów
        print("\n📝 Patient files to test:")
        for i, (pf, _, pt) in enumerate(patient_files, 1):
            patient_name = os.path.basename(pf).replace('.txt', '')
            print(f"   {i}. {patient_name} ({pt})")
        print()

        # Testuj każdego pacjenta
        all_results = []
        for i, (patient_file, expected_file, patient_type) in enumerate(patient_files, 1):
            patient_name = os.path.basename(patient_file).replace('.txt', '')
            print(f"\n{'='*60}")
            print(f"🏥 Testing patient {i}/{len(patient_files)}: {patient_name}")
            print(f"{'='*60}")

            try:
                result = self.run_functional_test_with_patient(
                    patient_file,
                    expected_file,
                    optimisation=optimisation,
                    use_cache=use_cache
                )
                all_results.append(result)

                # Zapisz wykres dla pacjenta (jak plot_metrics w referenced)
                if getattr(self, '_generate_plots', True):
                    self.plot_per_patient_summary(result, model_run_folder, timestamp)

                # APPEND DO LOGU PO KAŻDYM PACJENCIE
                print(f"  Saving patient {i}/{len(patient_files)} to log...")
                try:
                    # Wczytaj istniejący log lub stwórz nowy
                    log_data = self.load_json_file(log_file) if os.path.exists(log_file) else {"evaluations": []}

                    # Znajdź lub stwórz wpis dla tej sesji
                    session_found = False
                    for eval_entry in log_data["evaluations"]:
                        if (eval_entry.get("session_timestamp") == timestamp and
                            eval_entry.get("model_name") == self.model_name and
                            eval_entry.get("optimisation") == optimisation):
                            # Aktualizuj istniejący wpis
                            eval_entry["patients"].append(result)
                            session_found = True
                            break

                    if not session_found:
                        # Stwórz nowy wpis dla tej sesji
                        session_entry = {
                            "model_name": self.model_name,
                            "agent_type": self.agent.value,
                            "evaluation_type": "unreferenced",
                            "session_timestamp": timestamp,
                            "run_number": run_number,
                            "output_directory": model_run_folder,
                            "metadata": self.current_model_metadata,
                            "optimisation": optimisation,
                            "patients": [result]
                        }
                        log_data["evaluations"].append(session_entry)

                    # Zapisz atomowo
                    self.save_json_file(log_data, log_file)
                    print(f" Patient {i}/{len(patient_files)} saved to log")

                except Exception as save_error:
                    print(f"  Could not save patient {i} to log: {save_error}")

            except Exception as e:
                print(f"Error testing patient {patient_file}: {e}")
                traceback.print_exc()
                all_results.append({'patient_name': patient_name, 'patient_type': patient_type,
                                    'test_result': 'error', 'completed_successfully': False,
                                    'rounds_count': 0, 'rounds_data': [], 'total_duration_sec': 0,
                                    'error_type': type(e).__name__})

        # Oblicz metryki agregowane
        aggregated_metrics = self.calculate_aggregated_metrics(all_results)

        # Przygotuj dane sesji
        session_data["aggregated_metrics"] = aggregated_metrics
        session_data["patients"] = all_results

        # session_data = {
        #     "session_timestamp": timestamp,
        #     "model_name": self.model_name,
        #     "agent_type": self.agent.value,
        #     "evaluation_type": "unreferenced",

        #     "run_number": run_number,
        #     "output_directory": model_run_folder,
        #     # "metadata": self.current_model_metadata,
        #     "aggregated_metrics": aggregated_metrics,
        #     "patients": all_results  # zamiast "rounds" jak w referenced
        # }

        # Zapisz do logu
        try:
            # Przygotuj dane do logu
            log_data = self.load_json_file(log_file) if os.path.exists(log_file) else {"evaluations": []}

            # Dodaj nową ewaluację
            log_data['evaluations'] = [entry for entry in log_data['evaluations']
                                       if not (entry.get('session_timestamp') == timestamp
                                               and entry.get('model_name') == self.model_name
                                               and entry.get('optimisation') == optimisation)]
            log_data['evaluations'].append(session_data)

            # Zapisz atomowo
            self.save_json_file(log_data, log_file)
            print(f" All data safely logged to: {log_file}")

        except Exception as e:
            print(f"Failed to save log data: {e}")

        # Stwórz wykres podsumowujący dla modelu
        if getattr(self, '_generate_plots', True):
            self.plot_model_summary(session_data, model_run_folder, timestamp)

        return session_data

    def calculate_aggregated_metrics(self, all_results: List[Dict]) -> Dict:
        """Oblicza metryki agregowane dla wszystkich pacjentów"""

        if not all_results:
            return {}

        # Podziel na typy pacjentów
        successful_patients = [r for r in all_results if r["patient_type"] == "successful"]
        unsuccessful_patients = [r for r in all_results if r["patient_type"] == "unsuccessful"]

        # Metryki dla successful patients
        successful_metrics = {}
        if successful_patients:
            perfect = len([r for r in successful_patients if r["test_result"] == "perfect_success"])
            partial = len([r for r in successful_patients if r["test_result"] == "partial_success"])
            false_complete = len([r for r in successful_patients if r["test_result"] == "false_complete"])
            timeout = len([r for r in successful_patients if r["test_result"] == "timeout"])

            completed = [r for r in successful_patients if r["completed_successfully"]]

            successful_metrics = {
                "total": len(successful_patients),
                "perfect_success": perfect,
                "partial_success": partial,
                "false_complete": false_complete,
                "timeout": timeout,
                "success_rate": (perfect + partial) / len(successful_patients),
                "avg_rounds_to_complete": np.mean([r["rounds_count"] for r in completed]) if completed else 0,
                "avg_completion_time_sec": np.mean([r["total_duration_sec"] for r in completed]) if completed else 0
            }

            # Field collection rates
            field_rates = {}
            fields = ["name", "gender", "height", "blood_type", "date_of_birth"]
            for field in fields:
                collected = sum(1 for r in successful_patients
                              if field in r.get("collected_data", {}))
                field_rates[field] = collected / len(successful_patients)
            successful_metrics["field_collection_rates"] = field_rates

        # Metryki dla unsuccessful patients
        unsuccessful_metrics = {}
        if unsuccessful_patients:
            correct = len([r for r in unsuccessful_patients if r["test_result"] == "correct_rejection"])
            false_success = len([r for r in unsuccessful_patients if r["test_result"] == "false_success"])

            unsuccessful_metrics = {
                "total": len(unsuccessful_patients),
                "correct_rejection": correct,
                "false_success": false_success,
                "success_rate": correct / len(unsuccessful_patients),
                "avg_rounds_to_reject": np.mean([r["rounds_count"] for r in unsuccessful_patients])
            }

        # Overall metrics
        total_tests = len(all_results)
        test_passes = (
            successful_metrics.get("perfect_success", 0) +
            successful_metrics.get("partial_success", 0) +
            unsuccessful_metrics.get("correct_rejection", 0)
        )

        # Latency metrics
        all_latencies = []
        for result in all_results:
            for round_data in result.get("rounds_data", []):
                if "latency_breakdown" in round_data:
                    breakdown = round_data["latency_breakdown"]
                    if "total_ms" in breakdown:
                        all_latencies.append(breakdown["total_ms"])

        return {
            "successful_patients": successful_metrics,
            "unsuccessful_patients": unsuccessful_metrics,
            "overall": {
                "total_tests": total_tests,
                "test_pass_rate": test_passes / total_tests if total_tests > 0 else 0,
                "avg_latency_per_round_ms": np.mean(all_latencies) if all_latencies else 0,
                "min_latency_ms": np.min(all_latencies) if all_latencies else 0,
                "max_latency_ms": np.max(all_latencies) if all_latencies else 0
            }
        }

    # ========================================
    # PLOTTING FUNCTIONS
    # ========================================



    def get_models_from_logs(self, log_file):
        """Pobiera listę wszystkich modeli z pliku logów"""
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)

            models_set = set()
            for eval_session in data.get('evaluations', []):
                model_name = eval_session.get('model_name')
                if model_name and eval_session.get('patients') and len(eval_session['patients']) > 0:
                    models_set.add(model_name)

            models_list = sorted(list(models_set))
            print(f" Found models in logs: {models_list}")
            return models_list

        except Exception as e:
            print(f"  Error reading models from logs: {e}")
            return []

    def get_optimisations(self):
        """Optymalizacja dla unreferenced - najlepsza metoda na podstawie analizy referenced"""
        # Na podstawie analizy Round-by-Round Performance Comparison
        # najlepsza kombinacja to Flash Attention + Continuous Batching
        individual_optimizations = [
            {},  # Baseline
            {"--flash-attn": None, "--cont-batching": None},  # Najlepsza kombinacja z analizy
        ]

        combination_optimizations = []  # Pusta lista dla unreferenced

        print(f" Optimization plan for unreferenced:")
        print(f"    Individual optimizations: {len(individual_optimizations)}")
        print(f"    Najlepsza metoda: Flash Attention + Continuous Batching")
        print(f"    Uzasadnienie: Najniższa latencja (50-60ms), stabilna wydajność, najwyższa jakość generacji")
        print(f"    Combination optimizations: {len(combination_optimizations)}")

        return individual_optimizations, combination_optimizations

    def get_the_most_accurate_model(self):
        """
        Zwraca najlepszy model na podstawie test pass rate.
        """
        return getattr(self, '_best_model', self.model_name)

    # ========================================
    # STATIC PLOTTING METHODS dla all_models (porównanie wielu modeli)
    # ========================================




    # ========================================
    # MAIN PIPELINE (jak w referenced)
    # ========================================



    def evaluate_all_patients_with_optimisation(self, session_data, session_locations, optimisation):
        """
        Wrapper dla evaluate_all_patients z obsługą optymalizacji.
        Analogiczny do evaluate_all_rounds w referenced.
        """
        print(f"\n Starting batch evaluation for {self.model_name} with optimisation: {optimisation}")

        # Pobierz metadane modelu
        print(f"  Architecture: {self.current_model_metadata.get('architecture', 'Unknown')}")
        print(f"  Parameters: {self.current_model_metadata.get('parameter_size_display', 'Unknown')}")
        print(f"  Size: {self.current_model_metadata.get('model_size_gb', 0)} GB")

        # Ewaluuj wszystkich pacjentów (używa istniejącej logiki)
        session_results = self.evaluate_all_patients(
            session_data,
            session_locations["log_file"],
            session_locations["model_output_directory"],
            session_data["session_timestamp"],
            session_locations["model_run_number"],
            optimisation=optimisation,
            use_cache=session_data["use_cache"]
        )

        if session_results:
            print(f"\n Evaluation completed successfully!")
            print(f"   Test Pass Rate: {session_results['aggregated_metrics']['overall']['test_pass_rate']:.1%}")
            print(f"   Output: {session_locations['model_output_directory']}")
        else:
            print(f"\nEvaluation failed!")

        return session_results

    def pipeline_eval_model(self, mode: Literal["logs_only", "logs_and_viz", "viz_only"] = "logs_and_viz", use_cache: bool = True, optimisations: List[Dict] = [{}]):
        """
        Pipeline evaluation with 3 modes (analogiczny do referenced):

        Args:
            mode: Evaluation mode
                - "logs_only": Only perform evaluation and logging, no visualizations
                - "logs_and_viz": Perform evaluation, logging, and generate visualizations (default)
                - "viz_only": Only generate visualizations from existing logs, no evaluation
            use_cache: Whether to use caching for evaluations
            optimisations: List of optimization configurations to test (dla unreferenced tylko jedna: [{}])
        """

        if mode not in {'logs_only', 'logs_and_viz', 'viz_only'}:
            raise ValueError('Unknown evaluation mode')
        self._generate_plots = mode != 'logs_only'
        successful_evaluations = 0
        session_metadata = self.create_session()
        timestamp = session_metadata["timestamp"]
        log_folder = session_metadata["log_folder"]
        model_run_folder = session_metadata["model_run_folder"]
        run_number = session_metadata["model_run"]
        all_models_run = session_metadata["all_models_run"]
        all_models_run_folder = session_metadata["all_models_run_folder"]
        print(f" Session created: {timestamp},\n log folder: {log_folder},\n model run folder: {model_run_folder},\n all models run folder: {all_models_run_folder}")
        tool = self.create_tool_for_agent()
        log_file, _ = self.get_or_create_file_or_folder(f"{self.agent_type}_evaluation_results", type_of_file="log")
        all_models_log_file, _ = self.get_or_create_file_or_folder(f"all_models_{self.agent_type}_evaluation_results", type_of_file="log")
        print(f" All models run number: {run_number}")

        self.tools = tool
        self.cot_prompt = self.read_txt(self.MULTI_TURN_GLOBAL_CONFIG.get('cot_prompt_path'))
        session_data = {
            "session_timestamp": timestamp,
            "model_name": self.model_name,
            "evaluator_model_name":self.evaluator_model_name,
            "evaluation_type": self.eval_type,
            "optimisation": None,
            "parameters": self.MULTI_TURN_GLOBAL_CONFIG,
            "model_norm_name": self.model_name_norm,
            "agent_type": self.agent_type,
            "use_cache": use_cache,
            "tools": self.tools,
            "cot_prompt": self.cot_prompt,
            "aggregated_metrics": None,
            "patients": None,

        }
        session_locations = {
            "model_run_number": str(run_number),
            "all_models_run_number": str(all_models_run),
            "model_output_directory": model_run_folder,
            "all_models_output_directory": all_models_run_folder,
            "log_file": log_file,
            "all_models_log_file": all_models_log_file
        }

        # Handle 3 modes: "logs_only", "logs_and_viz", "viz_only"
        if mode == "viz_only":
            # Only visualizations, no logging - plot for all models in logs
            # Get list of all models from logs
            list_of_models = self.get_models_from_logs(session_locations["log_file"])

            if not list_of_models:
                print("No models found in logs for visualization")
                list_of_models = [self.model_name]  # Fallback to current model

            self.per_patient_plots(session_locations=session_locations, timestamp=timestamp, list_of_models=list_of_models)
            self.per_model_plots(session_locations=session_locations, timestamp=timestamp, list_of_models=list_of_models)
            self.all_models_plots(session_locations=session_locations, timestamp=timestamp)

        elif mode == "logs_only":
            # Only logging, no visualizations
            # Use provided optimisations or get default ones
            if not optimisations or optimisations == [{}]:
                optimisations_to_use = self.optimisation  # Użyj najlepszej kombinacji z __init__
            else:
                optimisations_to_use = optimisations

            # Run evaluation for each optimization (dla unreferenced powinna być tylko jedna)
            for optimization in optimisations_to_use:
                print(f"  Testing optimization: {optimization}")
                session_data_copy = session_data.copy()
                session_data_copy["optimisation"] = optimization
                self.evaluate_all_patients_with_optimisation(session_data_copy, session_locations, optimization)
                successful_evaluations += 1
            print(f" Ewaluacja modelu zakończona!")

        else:  # mode == "logs_and_viz" (default)
            # Both logging and visualizations
            # Use provided optimisations or get default ones
            if not optimisations or optimisations == [{}]:
                optimisations_to_use = self.optimisation  # Użyj najlepszej kombinacji z __init__
            else:
                optimisations_to_use = optimisations

            # Run evaluation for each optimization
            for optimization in optimisations_to_use:
                print(f"  Testing optimization: {optimization}")
                session_data_copy = session_data.copy()
                session_data_copy["optimisation"] = optimization
                self.evaluate_all_patients_with_optimisation(session_data_copy, session_locations, optimization)
                successful_evaluations += 1

            # Generate visualizations after logging
            if self._generate_plots:
                self.per_patient_plots(session_locations=session_locations, timestamp=timestamp)
                self.per_model_plots(session_locations=session_locations, timestamp=timestamp)
                self.all_models_plots(session_locations=session_locations, timestamp=timestamp)

            print(f" Ewaluacja modelu zakończona!")
# ========================================
# MAIN EXECUTION (identyczne jak w referenced)
# ========================================

if __name__ == "__main__":
    # Użyj nowych static methods z BaseEvaluation
    from edge_llm_lab.utils.base_eval import Agent
    agent_type_enum = Agent.CONSTANT_DATA_EN  # Auto-select CONSTANT_DATA_EN

    # === WSZYSTKIE PYTANIA NA POCZĄTKU ===
    print("\n KONFIGURACJA GLOBALNYCH USTAWIEŃ")
    print("="*50)

    # 1. Wybór modeli do testowania
    print("\n  WYBÓR MODELI DO TESTOWANIA:")
    print("1. Wszystkie modele z config.yaml")
    print("2. Tylko modele z tested: true bez wyników w logach")
    mode_choice = input("Wybierz tryb: 1 - Wszystkie modele z config, 2 - Tylko tested: true bez wyników: ")

    # 2. Czy automatycznie pobierać modele
    print("\n POBIERANIE MODELI:")
    auto_install = input("Czy automatycznie pobierać brakujące modele? (y/n): ").lower().strip()
    install_choice = "y" if auto_install in ["y", "yes", "tak"] else "n"

    # 3. Tryb działania
    print("\n  TRYB DZIAŁANIA:")
    print("1. Same logi (bez wizualizacji)")
    print("2. Logi + wizualizacje")
    print("3. Same wizualizacje (bez logowania)")
    execution_mode = input("Wybierz tryb: 1 - Same logi, 2 - Logi+viz, 3 - Same viz: ")

    print("\n" + "="*50)
    print(" KONFIGURACJA ZAKOŃCZONA")
    print("="*50)

    # Określ tryb na podstawie execution_mode
    if execution_mode == "1":
        mode = "logs_only"  # Same logi
        print("  Tryb: Same logi (bez wizualizacji)")
    elif execution_mode == "2":
        mode = "logs_and_viz"  # Logi + viz (domyślne zachowanie)
        print("  Tryb: Logi + wizualizacje")
    else:
        mode = "viz_only"   # Same viz
        print("  Tryb: Same wizualizacje")

    # Pobierz modele do testowania
    if mode_choice == "1":
        models_to_evaluate = EvalModelsUnreferenced.get_truly_untested_models(agent_type_enum.value, "unreferenced", only_tested_true=False)
        print("  Tryb: Wszystkie modele z config")
    else:
        models_to_evaluate = EvalModelsUnreferenced.get_truly_untested_models(agent_type_enum.value, "unreferenced", only_tested_true=True)
        print("  Tryb: Tylko tested: true bez wyników")

    if not models_to_evaluate:
        print(f"Brak modeli do testowania dla agenta {agent_type_enum.value}")
        exit(1)

    total_models = len(models_to_evaluate)
    print(f" Znaleziono {total_models} modeli do testowania")

    # === PĘTLA PRZEZ MODELE ===
    best_model = None
    for i, model_name in enumerate(models_to_evaluate, 1):
        print(f"\n{'='*60}")
        print(f" MODEL {i}/{total_models}: {model_name}")
        print(f"{'='*60}")

        # Sprawdź dostępność modelu (używaj globalnego ustawienia)
        if not BaseEvaluation.check_model_availability(model_name, install_choice=install_choice):
            print(f"  Pomijam model {model_name}...")
            continue

        print(f"  Using model: {model_name}")
        print(f"  Using agent: {agent_type_enum}")

        # Inicjalizuj evaluator dla tego modelu
        unreferenced_model_evaluator = EvalModelsUnreferenced(
                model_name=model_name,
                agent=agent_type_enum
            )

        # Uruchom ewaluację
        print(f" Running unreferenced evaluation for {model_name} and {agent_type_enum}")
        unreferenced_model_evaluator.pipeline_eval_model(mode=mode)  # Użyj wybranego trybu!
        current_best = unreferenced_model_evaluator.get_the_most_accurate_model()
        if not best_model:
            best_model = current_best
        del unreferenced_model_evaluator

        # Oznacz model jako tested
        try:
            from model_config_loader import mark_model_as_tested
            mark_model_as_tested(agent_type_enum.value, model_name)
            print(f"   Model {model_name} oznaczony jako tested: true")
        except Exception as e:
            print(f"   Nie udało się oznaczyć modelu jako tested: {e}")

    # === NAJLEPSZY MODEL ===
    if not best_model:
        print("Nie znaleziono żadnego działającego modelu!")
        exit(1)

    print(f"\n NAJLEPSZY MODEL: {best_model}")
    print("="*60)

    print("\n EWALUACJA ZAKOŃCZONA!")
