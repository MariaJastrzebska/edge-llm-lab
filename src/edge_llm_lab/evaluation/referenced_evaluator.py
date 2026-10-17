from edge_llm_lab.evaluation.dataset import DatasetEvaluation
from edge_llm_lab.visualization.referenced_plots import ReferencedPlots
from edge_llm_lab.metrics.referenced import ReferencedMetrics
import numpy as np
import json
from typing import List, Dict, Any, Union, Optional, Literal
from pydantic import BaseModel, Field
import os
import sys

# Add project root and src to path for direct script execution
try:
    _file_path = os.path.abspath(__file__)
    # src/edge_llm_lab/evaluation/referenced_evaluator.py -> ../../.. gets to project root
    _project_root = os.path.abspath(os.path.join(os.path.dirname(_file_path), "../../../"))
    _src_path = os.path.join(_project_root, "src")
    if _src_path not in sys.path:
        sys.path.insert(0, _src_path)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    pass

from datetime import datetime
import matplotlib.pyplot as plt
from enum import Enum

from edge_llm_lab.utils.base_eval import BaseEvaluation, Agent

try:
    from dotenv import load_dotenv
    load_dotenv()
    print(" Environment variables loaded from .env")
except ImportError:
    print(" python-dotenv not installed, using system environment variables")


class EvalModelsReferenced(DatasetEvaluation, ReferencedPlots, ReferencedMetrics, BaseEvaluation):
    class RoundNumber(Enum):
        ZERO = 0
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5

    # Pydantic models for GPT Judge structured output (matches log format exactly)
    class CriterionScore(BaseModel):
        score: int
        explanation: str

    class GPTJudgeResponse(BaseModel):
        criteria_scores: Dict[str, 'EvalModelsReferenced.CriterionScore']
        overall_strengths: List[str] = Field(default_factory=list)
        overall_weaknesses: List[str] = Field(default_factory=list)

    _cached_references = {}  # agent_type -> reference_file_path

    def __init__(self, model_name, agent, eval_type="referenced"):
        """Initialize EvalModelsReferenced with model name, agent, and optional source path.
        """
        super().__init__(model_name, agent, eval_type)
        self.eval_type = eval_type
        self._finalize_init(model_name=model_name, agent=agent, eval_type=self.eval_type)

        # Initialize attributes that may be accessed before pipeline_eval_model
        self.optimisation = {}
        self.tools = self.create_tool_for_agent()
        self.cot_prompt = self.read_txt(self.MULTI_TURN_GLOBAL_CONFIG.get('cot_prompt_path'))































    def create_reference_response(self,tools_schema, max_rounds=10, interactive=True):

        """
        Interaktywne tworzenie referencyjnej konwersacji z użyciem ChatGPT.
        Zapisuje pełną konwersację do pliku JSON w katalogu reference_conversations/.
        """
        reference_file, existed = self.get_or_create_file_or_folder(f"reference_{self.agent_type}", type_of_file="reference")
        if existed:

            print(f" Reference file already exists: {reference_file}")
            reference_data = self.load_json_file(reference_file)
            if reference_data and reference_data != {}:
                if not interactive:
                    return reference_file
                overwrite = input(f" ❓ Reference already exists. Overwrite with new interactive session? (y/n): ").lower().strip()
                if overwrite not in ['y', 'yes', 'tak']:
                    return reference_file
            else:
                print("   Reference file is empty. Starting new interactive session...")



        print(f" Creating reference conversation for agent: {self.agent_type}")
        system_content = self.read_txt(self.MULTI_TURN_GLOBAL_CONFIG.get('cot_prompt_path'))


        # Get tools schema for this agent



        # Add tools schema and formatting instructions to system prompt
        if tools_schema:
            tools_json = json.dumps(tools_schema, indent=2, ensure_ascii=False)
            system_content += f"\n\nRespond in JSON format, either with `tool_call` (a request to call tools) or with `response` reply to the user's request.\n\nYou can call any of the following tools to satisfy the user's requests:\n{tools_json}\n\nExample tool call syntax:\n\nassistant: {{\n  \"tool_calls\": [\n    {{\n      \"name\": \"tool_name\",\n      \"arguments\": {{\n        \"arg1\": \"some_value\"\n      }},\n      \"id\": \"call_1___\"\n    }}\n  ]\n}}"

        conversation = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Rozpocznij Zbieranie Danych"}
        ]
        conversation_en = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Start Data Collection"}
        ]
        if self.agent_type.endswith("_en"):
            conversation = conversation_en
        else:
            conversation = conversation

        round_number = 0

        while round_number < max_rounds:
            print(f"\n ROUND {round_number}")
            try:
                assistant_content = self.get_openai_response(
                    custom_catch_type="reference_conversation",
                    messages=conversation,
                    tools_schema=tools_schema,
                )
                conversation.append(
                    {"role": "assistant", "content": assistant_content})

                print(" ChatGPT Response:")
                try:
                    parsed = json.loads(assistant_content)
                    print(json.dumps(parsed, indent=2, ensure_ascii=False))
                except Exception:
                    print(assistant_content)

                user_input = input(
                    ">>> Wprowadź odpowiedź użytkownika (lub 'quit'/'save'): ").strip()
                if user_input.lower() in ["quit", "q", "exit"]:
                    print(" Stopping conversation creation...")
                    break
                if user_input.lower() in ["save", "s"]:
                    print(" Saving conversation...")
                    break
                if not user_input:
                    print(" Empty input, skipping...")
                    continue

                conversation.append({"role": "user", "content": user_input})
                round_number += 1
            except Exception as e:
                print(f" Error in round {round_number}: {e}")
                break

        # Save reference - TYLKO jeśli conversation ma sensowne dane
        try:
            # Walidacja przed zapisem
            if not conversation or len(conversation) < 2:
                # print(f"Cannot save reference - conversation too short: {len(conversation)} messages")
                return None

            # Sprawdź czy są odpowiedzi od assistanta
            assistant_responses = [m for m in conversation if m.get('role') == 'assistant']
            if not assistant_responses:
                print(f"Cannot save reference - no assistant responses found")
                return None

            reference_data = {
                "agent": self.agent_type,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_rounds": len([m for m in conversation if m.get('role') == 'user']) - 1,
                "conversation": conversation,
            }
            reference_data = self.pretty_json(reference_data, 'json')
            self.save_json_file(reference_data, reference_file)
            print(f" Reference saved: {reference_file}")
            return reference_file
        except Exception as e:
            print(f"Error saving reference: {e}")
            return None



    def load_reference_conversation(self, reference_file):
        """Ładuje referencyjną konwersację z pliku JSON."""
        print("loading reference conversation from file: ", reference_file)
        if not os.path.exists(reference_file):
            print(f" Reference file not found: {reference_file}")
            return None
        try:
            with open(reference_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f" Loaded reference for {self.agent_type}")
            return data.get('conversation', [])
        except Exception as e:
            print(f" Error loading reference: {e}")
            return None

    def get_reference_response_for_round(self, reference_file, round_number=RoundNumber.ZERO):
        """Zwraca treść odpowiedzi asystenta dla danej rundy z referencji."""
        print(f" Loading reference conversation for agent: {self.agent_type} and round: {round_number}")
        conversation = self.load_reference_conversation(reference_file)
        if not conversation:
            return None
        assistant_responses = [
            m for m in conversation if m.get('role') == 'assistant']
        idx = round_number.value if isinstance(
            round_number, self.RoundNumber) else int(round_number)
        if 0 <= idx < len(assistant_responses):
            return assistant_responses[idx].get('content', '')
        print(f" Round {idx} not found in reference conversation")
        return None

    def get_reference_context_for_round(self, reference_file, round_number=0):
        """Return context strictly before the selected assistant target."""
        conversation = self.load_reference_conversation(reference_file)
        if not conversation:
            raise ValueError("Missing or empty reference conversation")
        index = round_number.value if hasattr(round_number, 'value') else int(round_number)
        targets = [i for i, message in enumerate(conversation) if message.get('role') == 'assistant']
        if index < 0 or index >= len(targets):
            raise IndexError("Reference round is out of range")
        return conversation[:targets[index]]

    def evaluate_case(self, case, optimisation, use_cache=False):
        """Capture every assistant target, optionally deferring all grading."""
        conversation = case['conversation']
        rounds = []
        from edge_llm_lab.evaluation.dataset import reference_turns
        for index, context, target in reference_turns(conversation):
            self.current_round_context = context
            response, latency = self.run_inference(self.tools, context, optimisations=optimisation)
            metrics = ({} if getattr(self, 'defer_scoring', False) else
                       self.calculate_fast_metrics(response, target, use_cache))
            rounds.append({'round': len(rounds), 'context': context,
                           'reference_response': target, 'llm_response': response,
                           'latency_breakdown': latency, 'metrics': metrics})
        return {'rounds': rounds, 'protocol': 'frozen_reference_replay'}


    def create_dummy_round_context(self, round_number=RoundNumber.ZERO):
        print(f" Creating dummy round context for agent: {self.agent_type} and round: {round_number}")

        if round_number == self.RoundNumber.ZERO:
            return self.context_messages[:2]
        if round_number == self.RoundNumber.ONE:
            return self.context_messages[:4]
        if round_number == self.RoundNumber.TWO:
            return self.context_messages[:6]
        if round_number == self.RoundNumber.THREE:
            return self.context_messages[:8]
        if round_number == self.RoundNumber.FOUR:
            return self.context_messages[:10]
        return self.context_messages

    def create_dummy_reference_response(self, round_number=RoundNumber.ZERO):
        print(f" Creating dummy reference response for agent: {self.agent_type} and round: {round_number}")
        if round_number == self.RoundNumber.ZERO:
            content_str = self.context_messages[2]["content"]
        if round_number == self.RoundNumber.ONE:
            content_str = self.context_messages[4]["content"]
        if round_number == self.RoundNumber.TWO:
            content_str = self.context_messages[6]["content"]
        if round_number == self.RoundNumber.THREE:
            content_str = self.context_messages[8]["content"]
        if round_number == self.RoundNumber.FOUR:
            content_str = self.context_messages[10]["content"]

        # Zwracamy sformatowany string JSON, a nie obiekt dict
        return self.pretty_json(content_str, 'str')







    def evaluate_single_round(self, session_data,round_num, optimisation):
        """Ewaluuje pojedynczą rundę i zapisuje wyniki."""
        print(f" round_num: {round_num} for {self.model_name}, {self.agent_type}")
        context = self.get_reference_context_for_round(session_data['reference_file'], round_num)
        # print(f"======context======: {context}")


        llm_response, latency_breakdown = self.run_inference(
            session_data['tools'], context, optimisations=optimisation
        )
        # print(f"======llm_response======: {llm_response}")

        # Get reference response
        reference_response = self.get_reference_response_for_round(
             session_data['reference_file'], round_num)
        # print(f"======reference_response======: {reference_response}")
        if not reference_response:
            print(f" No reference found for round {round_num}")
            return None

        self.current_round_context = context
        # Calculate metrics
        comprehensive_metrics = self.calculate_fast_metrics(
            llm_response, reference_response, session_data['use_cache'])
        # print(f"======comprehensive_metrics======: {comprehensive_metrics}")
        # Create plot and save
        # print(f" Plotting per round with reference...")
        round_data = {
            "round": round_num.value if hasattr(round_num, 'value') else round_num,
            "context": context,
            "reference_response": reference_response,
            "llm_response": llm_response,
            "latency_breakdown": latency_breakdown,
            "metrics": comprehensive_metrics,

        }
        # if plot:
        #     plot_path = self.plot_per_round_with_reference(
        #         round_data=round_data,
        #         metadata=self.current_model_metadata,
        #         output_dir=session_data['model_output_directory'],
        #         session_timestamp=session_data['session_timestamp']
        #     )
        #     print(f"======plot_path======: {plot_path}")
        #     print(f" Plot saved: {plot_path}")
        #     # Prepare round data for logging


        print(f" Round {round_num} evaluation completed!")


        return round_data

    def evaluate_all_rounds(self, session_data, session_locations, optimisation, inference_params=None, log_file=None):
        """
        Ewaluuje wszystkie rundy w jednej sesji.

        Args:
            session_data: Dict z danymi sesji
            session_locations: Dict z lokalizacjami plików
            optimisation: Dict z optymalizacjami
            inference_params: Dict z parametrami inferencji (temperature, top_p, etc.)
            log_file: Opcjonalna ścieżka do pliku logów

        Example:
            # >>> # session_data contains optimisation parameter
            # >>> session_data = {
            # ...     "optimisation": {"--kv-cache": None},
            # ...     "model_name": "granite3.1-dense:2b",
            # ...     "rounds": None,
            # ...     # ... other fields
            # ... }
            # >>>
            # >>> evaluator.evaluate_all_rounds(session_data, session_locations)
            # >>> # After evaluation, session_data['rounds'] contains:
            # >>> # [
            # >>> #   {'round': 0, 'metrics': {...}, 'latency_breakdown': {...}},
            # >>> #   {'round': 1, 'metrics': {...}, 'latency_breakdown': {...}},
            # >>> #   ...
            # >>> # ]
            # >>> # Saves to log file with session_data including all rounds
        """
        # Jeśli są inference_params, zaktualizuj session_data
        if inference_params:
            session_data["parameters"]["context_size"] = inference_params.get("context_size", session_data["parameters"]["context_size"])
            session_data["parameters"]["max_tokens"] = inference_params.get("max_tokens", session_data["parameters"]["max_tokens"])
            session_data["parameters"]["temperature"] = inference_params.get("temperature", session_data["parameters"]["temperature"])
            session_data["parameters"]["top_p"] = inference_params.get("top_p", session_data["parameters"]["top_p"])

        if not log_file:
            log_file = session_locations["log_file"]

        print(f"\n Starting batch evaluation for {self.model_name}")


        # Load existing log data
        log_data = self.load_json_file(log_file) or {"evaluations": []}  #  Zawsze dict!

        rounds = list(range(sum(m.get('role') == 'assistant' for m in
                                self.load_reference_conversation(session_data['reference_file']) or [])))
        if not rounds:
            raise ValueError('Reference has no assistant targets')
        round_results = []

        # Get optimisation from session_data
        optimisation = session_data.get('optimisation', {})

        for i, round_num in enumerate(rounds):
            print
            try:
                print(f" Evaluating round {round_num}")
                round_data = self.evaluate_single_round(session_data,
                    round_num, optimisation)
                # print(f"======round_data======: {round_data}")
                if round_data:
                    round_results.append(round_data)
                    # print(f"======round_results======: {round_results}")
                    print(
                        f" Round {round_num} completed ({i+1}/{len(rounds)})")

                    # Log to Neptune in real-time using NeptuneManager
                    self.neptune.log_round_metrics(
                        round_num=round_num,
                        metrics=round_data['metrics'],
                        latency=round_data.get('latency_breakdown')
                    )
                else:
                    print(f" Round {round_num} failed")
            except Exception as e:
                print(f" Error in round reference_file {round_num}: {e}")
                round_results.append({'round': round_num, 'status': 'error',
                                      'error_type': type(e).__name__, 'metrics': {},
                                      'latency_breakdown': {}})
                continue

        # Get model metadata (będzie dostępne po pierwszym wywołaniu run_inference)

        # print(f" model_metadata: {model_metadata}")

        # Print model metadata as requested

        print(f"  Architecture: {self.current_model_metadata.get('architecture', 'Unknown')}")
        print(f"  Parameters: {self.current_model_metadata.get('parameter_size_display', 'Unknown')}")
        print(f"  Size: {self.current_model_metadata.get('model_size_gb', 0)} GB")

        # Prepare session data for logging with model metadata
        session_data["rounds"] = round_results

        try:
            # Walidacja: nie zapisuj sesji z pustymi rounds
            if session_data.get("rounds") and len(session_data["rounds"]) > 0:
                # Add session to log data
                log_data["evaluations"].append(session_data)
                print(f" Session added with {len(session_data['rounds'])} rounds")
            else:
                print(f"  Skipping session save - no rounds completed")
                return None

            # Atomic save - only write if everything succeeded
            self.save_json_file(log_data, log_file)
            print(f" All data safely logged to: {log_file}")

        except Exception as e:
            print(f"Failed to save log data: {e}")
            print("   Evaluation completed but log not saved - data preserved in memory")
        # return {
        #     "session_timestamp": session_data["session_timestamp"],
        #     "run_number": str(session_data["model_run_number"]),
        #     "output_dir": session_locations["model_output_directory"],
        #     "log_file": session_locations["log_file"],
        #     "results": round_results,
        #     "success_count": len(round_results),
        #     "total_rounds": len(rounds),
        #     # "summary_plot": summary_plot_path
        # }


































    def get_models_from_logs(self, log_file):
        """Pobiera listę wszystkich modeli z pliku logów"""
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)

            models_set = set()
            for eval_session in data.get('evaluations', []):
                model_name = eval_session.get('model_name')
                if model_name and eval_session.get('rounds') and len(eval_session['rounds']) > 0:
                    models_set.add(model_name)

            models_list = sorted(list(models_set))
            print(f" Found models in logs: {models_list}")
            return models_list

        except Exception as e:
            print(f"  Error reading models from logs: {e}")
            return []

    def get_existing_optimizations_from_logs(self, log_file):
        """Sprawdź jakie optymalizacje już istnieją w logach dla tego modelu i agenta"""
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)

            existing_optimizations = set()
            for eval_session in data.get('evaluations', []):
                if (eval_session.get('model_name') == self.model_name and
                    eval_session.get('agent_type') == self.agent_type and
                    eval_session.get('rounds') and len(eval_session['rounds']) > 0):
                    # Konwertuj optymalizację do string dla porównania
                    opt_str = str(eval_session.get('optimisation', {}))
                    existing_optimizations.add(opt_str)

            print(f" Found {len(existing_optimizations)} existing optimizations in logs:")
            for opt in sorted(existing_optimizations):
                print(f"   {opt}")

            return existing_optimizations
        except Exception as e:
            print(f"  Error reading logs: {e}")
            return set()

    def pipeline_eval_model(self, mode: Literal["logs_only", "logs_and_viz", "viz_only"] = "logs_and_viz", use_cache: bool = True, optimisations_choice: Literal["selected", "test"] = "selected", inference_params=False, use_polish: bool = True, stage_name: str = "evaluation", generate_comparison: bool = True, generate_per_round: bool = True, generate_per_model: bool = True, generate_aggr_over_rounds: bool = True, neptune_tags_list: Optional[list] = None, pipeline_run_number: int = None, pipeline_timestamp: str = None):
        """
        Pipeline evaluation with 3 modes:

        Args:
            mode: Evaluation mode
                - "logs_only": Only perform evaluation and logging, no visualizations
                - "logs_and_viz": Perform evaluation, logging, and generate visualizations (default)
                - "viz_only": Only generate visualizations from existing logs, no evaluation
            use_cache: Whether to use caching for evaluations
            optimisations: List of optimization configurations to test
            inference_params: List of inference parameter combinations to test (optional)
                - If provided: Test these specific parameter combinations
                - If None: Use standard parameters from config
            stage_name: Name of the current evaluation stage (e.g. "stage_1_selection")
        """
        # optimization config loading
        selected = [{"name": "default"}] # Placeholder default
        test = [{"name": "default"}] # Placeholder default

        # Try to load from custom_optimizations.yaml if available.
        # BUGFIX: was a CWD-relative path hardcoded to constant_data_en —
        # resolve relative to this repo and to the CURRENT agent, with the
        # constant_data_en file as fallback.
        try:
             _agents_dir = os.path.abspath(os.path.join(
                 os.path.dirname(__file__), "../../../examples/desktop/input/agents"))
             optim_config_path = os.path.join(
                 _agents_dir, self.agent_type, "evaluation_config/custom_optimizations.yaml")
             if not os.path.exists(optim_config_path):
                 optim_config_path = os.path.join(
                     _agents_dir, "constant_data_en/evaluation_config/custom_optimizations.yaml")
             if os.path.exists(optim_config_path):
                 import yaml
                 with open(optim_config_path, 'r') as f:
                     optim_config = yaml.safe_load(f)
                     selected = optim_config.get('selected', selected)
                     test = optim_config.get('test', test)
        except Exception as e:
            print(f"⚠️ Could not load custom optimizations, using defaults: {e}")

        if optimisations_choice == "selected":
            optimisations = selected
        if optimisations_choice == "test":
            optimisations = test



        successful_evaluations = 0
        session_metadata = self.create_session(fixed_run_number=pipeline_run_number, fixed_timestamp=pipeline_timestamp, create_all_models=generate_comparison)
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

        # Log Recovery Logic for Viz Mode
        if mode == "viz_only" and not os.path.exists(log_file):
            print(f"⚠️ Local log file not found: {log_file}")
            # Only ask if we are NOT in an automated pipeline or if explicitly requested
            # For now, let's skip it in automated viz_only to avoid hanging
            # recover_choice = input("🔎 Would you like to recover this log from Neptune? [y/n]: ").lower().strip()
            # if recover_choice in ["y", "yes", "tak"]:
            #     ...
            print("💡 Recovery from Neptune skipped in automated mode. Check if log_file exists.")

        print(f" All models run number: {run_number}")
        if self.agent_type in EvalModelsReferenced._cached_references:
            reference_file = EvalModelsReferenced._cached_references[self.agent_type]
            print(f" Using cached reference for {self.agent_type}: {reference_file}")
        else:
            # Skip interactive reference creation in non-eval modes
            interactive_ref = (mode not in ["viz_only", "logs_only"])
            reference_file = self.create_reference_response(tools_schema=tool, max_rounds=10, interactive=interactive_ref)
            EvalModelsReferenced._cached_references[self.agent_type] = reference_file
        self.tools = tool
        self.cot_prompt = self.read_txt(self.MULTI_TURN_GLOBAL_CONFIG.get('cot_prompt_path'))


        session_data = {
            "session_timestamp": timestamp,
            "model_name": self.model_name,
            "evaluator_model_name":self.evaluator_model_name,
            "evaluation_type": self.eval_type,
            "optimisations": optimisations,
            "engine":None,
            "parameters": self.MULTI_TURN_GLOBAL_CONFIG,
            "model_norm_name": self.model_name_norm,
            "agent_type": self.agent_type,
            "use_cache": use_cache,
            "tools": self.tools,
            "cot_prompt": self.cot_prompt,
            "reference_file":reference_file,
            "rounds": None,
        }

        session_locations = {
            "model_run_number": str(run_number),
            "all_models_run_number": str(all_models_run),
            "model_output_directory": model_run_folder,
            "all_models_output_directory": all_models_run_folder,
            "log_file": log_file,
            "all_models_log_file": all_models_log_file
        }

        # Przygotuj log_file_params_test jeśli są inference_params
        log_file_params_test = None
        inference_param_combinations = None
        if inference_params:
            log_file_params_test = session_locations["log_file"].replace(".json", "_inference_params_test.jsonl")
            # Pobierz kombinacje parametrów do testowania
            inference_param_combinations = self.get_inference_parameter_combinations()

        def log_results():
            successful_evaluations = 0
            for optimization in optimisations:
                session_data["optimisation"] = optimization
                print(f"  Testing optimization: {optimization}")
                try:
                    if inference_params and inference_param_combinations:
                        for inference_param in inference_param_combinations:
                            print(f"  Testing inference parameter: {inference_param}")
                            self.evaluate_all_rounds(session_data, session_locations, optimization, inference_param, log_file_params_test)
                    else:
                        print(f"  Testing optimization without inference parameters")
                        self.evaluate_all_rounds(session_data, session_locations, optimization)

                except Exception as e:
                    print(f"Error evaluating all rounds: {e}")

                successful_evaluations += 1
            print(f" Ewaluacja modelu zakończona!")

        # Plotting flags
        # generate_per_round passed as arg
        generate_aggr_over_rounds = generate_per_model
        # generate_per_model passed as arg
        generate_all_models = generate_comparison

        def plot_results():
            # Only visualizations, no logging - plot for all models in logs
            # Get list of all models from logs
            list_of_models = self.get_models_from_logs(session_locations["log_file"])

            # For single model runs (not summary), restrict scope to current model only
            if not generate_comparison:
                list_of_models = [self.model_name]
                print(f"📉 Single run mode: Restricting visualization to current model: {self.model_name}")

            if not list_of_models:
                print("No models found in logs for visualization")
                list_of_models = [self.model_name]  # Fallback to current model



            if generate_per_round:
                try:
                    print("\n Generating per-round plots...")
                    self.per_round_plots(session_locations=session_locations, timestamp=timestamp, list_of_models=list_of_models, use_polish=use_polish)
                except Exception as e:
                    print(f"⚠️ Error generating per-round plots: {e}")
            else:
                print("Skipping per-round plots")

            if generate_aggr_over_rounds:
                try:
                    print("\n Generating aggr_over_rounds plots...")
                    self.per_model_plots(session_locations=session_locations, timestamp=timestamp, list_of_models=list_of_models, use_polish=use_polish)
                except Exception as e:
                    print(f"⚠️ Error in aggr_over_rounds plots: {e}")
            else:
                print("Skipping aggr_over_rounds plots")

            if generate_per_model:
                try:
                    print("\n Generating per-model plots...")
                    self.per_model_plots(session_locations=session_locations, timestamp=timestamp, list_of_models=list_of_models, use_polish=use_polish)
                except Exception as e:
                    print(f"⚠️ Error generating per-model plots: {e}")
            else:
                print("Skipping per-model plots")

            if generate_all_models:
                try:
                    print("\n Generating all-models plots...")
                    self.all_models_plots(session_locations=session_locations, timestamp=timestamp, use_polish=use_polish, interactive=False)
                except Exception as e:
                    print(f"⚠️ Error generating all-models plots: {e}")
            else:
                print("Skipping all-models plots")

            # Plot inference parameters comparison if test data exists
            # Użyj stałej ścieżki do pliku w katalogu log/ zamiast konkretnego runu
            log_file_params_test = session_locations["log_file"].replace("model/", "log/").replace(".json", "_inference_params_test.jsonl")
            if os.path.exists(log_file_params_test):
                print("\n Generating inference parameters comparison plots...")
                params_data = self.load_json_file(log_file_params_test)
                if params_data and params_data.get('evaluations'):
                    # Przygotuj dane dla plot_inference_parameters_comparison
                    parameter_results = {}
                    for eval_session in params_data['evaluations']:
                        params = eval_session.get('parameters', {})
                        param_key = f"ctx:{params.get('context_size', 0)}_max:{params.get('max_tokens', 0)}_temp:{params.get('temperature', 0)}_p:{params.get('top_p', 0)}"
                        parameter_results[param_key] = {
                            'parameters': params,
                            'session_data': eval_session
                        }

                    if parameter_results:
                        plot_path = self.plot_inference_parameters_comparison(
                            parameter_results=parameter_results,
                            model_name=self.model_name,
                            agent_type=self.agent_type,
                            plotting_session_timestamp=timestamp,
                            output_dir=session_locations["model_output_directory"],
                            output_file_name=f"inference_params_comparison_{timestamp}"
                        )
                        if plot_path:
                            print(f"  ✓ Inference parameters comparison plot saved to: {plot_path}")
                else:
                    print("  ℹ️ No inference parameters test data found")
            else:
                print("  ℹ️ No inference parameters test file found")

        # Handle 3 modes: "logs_only", "logs_and_viz", "viz_only"
        # Initialize Neptune Run using NeptuneManager
        neptune_initialized = False
        if mode in ["logs_only", "logs_and_viz", "viz_only"]:
             # Prepare tags based on scenario
             neptune_tags = [self.agent_type, self.eval_type, stage_name]
             if neptune_tags_list:
                 neptune_tags.extend(neptune_tags_list)
             if inference_params:
                 neptune_tags.append("inference_test")
             if optimisations_choice == "test":
                 neptune_tags.append("quantization_test")

             # Prepare core parameters
             params = {
                 "model_name": self.model_name,
                 "agent_type": self.agent_type,
                 "eval_type": self.eval_type,
                 "temperature": self.TEMPERATURE,
                 "context_size": self.CONTEXT_SIZE,
                 "max_tokens": self.MAX_TOKENS,
                 "seed": self.SEED
             }

             # Prepare metadata
             metadata = None
             if self.current_model_metadata:
                 metadata = {
                     "architecture": self.current_model_metadata.get('architecture'),
                     "parameter_size": self.current_model_metadata.get('parameter_size_display'),
                     "quantization": self.current_model_metadata.get('quantization_level'),
                     "format": self.current_model_metadata.get('model_format'),
                     "model_size_gb": self.current_model_metadata.get('model_size_gb')
                 }

             neptune_initialized = self.neptune.init_run(
                 name=f"EVAL_{self.model_name_norm}",
                 tags=neptune_tags,
                 params=params,
                 metadata=metadata
             )

             if neptune_initialized:
                 if os.path.exists(reference_file):
                     self.neptune.upload_artifact(reference_file, "reference_file")
                 if self.cot_prompt:
                     self.neptune.run["cot_prompt"] = self.cot_prompt
                 # Log prompt file if exists
                 prompt_path = self.MULTI_TURN_GLOBAL_CONFIG.get('cot_prompt_path')
                 if prompt_path and os.path.exists(prompt_path):
                     self.neptune.upload_artifact(prompt_path, "system_prompt")

                 # Upload Full Source Code
                 current_file_path = os.path.abspath(__file__)
                 src_dir = os.path.abspath(os.path.join(os.path.dirname(current_file_path), "../.."))
                 print(f"DEBUG: Uploading source code from {src_dir}")
                 self.neptune.upload_directory_artifacts(src_dir, "source_code", extensions=(".py", ".yaml", ".md"))

                 # Upload Config Files
                 # Try to locate config files relative to the script execution or known paths
                 possible_config_paths = [
                     "examples/desktop/input/agents/constant_data_en/evaluation_config/config.yaml",
                     "examples/desktop/input/agents/constant_data_en/evaluation_config/config_quantized.yaml",
                     "examples/desktop/input/agents/constant_data_en/evaluation_config/custom_optimizations.yaml",
                     "examples/desktop/output/metadata/models_metadata.json"
                 ]
                 for config_path in possible_config_paths:
                     if os.path.exists(config_path):
                         self.neptune.upload_artifact(config_path, f"config/{os.path.basename(config_path)}")

                 # Dump and upload active configuration as JSON
                 active_config_dump_path = os.path.join(log_folder, "active_config_dump.json")
                 self.save_json_file(self.MULTI_TURN_GLOBAL_CONFIG, active_config_dump_path, evaluations=False)
                 self.neptune.upload_artifact(active_config_dump_path, "config/active_config_dump.json")

        # Handle 3 modes: "logs_only", "logs_and_viz", "viz_only"
        if mode == "viz_only":
            plot_results()

        elif mode == "logs_only":
            log_results()

        elif mode == "logs_and_viz":
            log_results()
            plot_results()

        # Upload final logs and artifacts to Neptune
        if neptune_initialized:
            print("📤 Uploading results and visualizations to Neptune...")
            # Upload main log file
            if os.path.exists(log_file):
                self.neptune.upload_artifact(log_file, "final_logs/main_log")

            # Upload all_models log file
            if os.path.exists(all_models_log_file):
                self.neptune.upload_artifact(all_models_log_file, "final_logs/all_models_log")

            # Upload results from BOTH folders (use session_locations for correct paths)
            model_upload_folder = session_locations["model_output_directory"]
            all_models_upload_folder = session_locations["all_models_output_directory"]

            if os.path.exists(model_upload_folder):
                 num_files = len([f for f in os.listdir(model_upload_folder) if not f.startswith('.')])
                 print(f"DEBUG: Folder {model_upload_folder} contains {num_files} files before upload.")
            else:
                 print(f"DEBUG: Folder {model_upload_folder} DOES NOT EXIST before upload.")

            self.neptune.upload_directory_artifacts(model_upload_folder, "model_artifacts", gallery_path="visualizations/mosaic")
            self.neptune.upload_directory_artifacts(all_models_upload_folder, "comparison_artifacts", gallery_path="visualizations/mosaic")

            # Log successful count
            # Generate Best Models Summary if we have multiple models
            if generate_all_models:
                 self.generate_best_models_summary(session_locations, timestamp, use_polish)

            # Log scalar metrics for comparison in Neptune Dashboard
            self.log_metrics_to_neptune(session_locations)

            self.neptune.stop()
            print(" Neptune upload completed")

        # Cleanup unused all_models run folder if not used for comparison
        if not generate_comparison and os.path.exists(all_models_run_folder):
            try:
                # Check if empty (ignoring hidden files)
                if not any(f for f in os.listdir(all_models_run_folder) if not f.startswith('.')):
                     print(f"  Removing unused all_models run folder: {all_models_run_folder}")
                     os.rmdir(all_models_run_folder)
            except Exception as e:
                print(f"⚠️ Failed to remove unused all_models folder: {e}")

        return session_metadata

    def log_metrics_to_neptune(self, session_locations):
        """Parses local logs and uploads scalar metrics to Neptune for comparison."""
        print(" Logging scalar metrics to Neptune...")
        log_file = session_locations["log_file"]

        # Load logs
        if not os.path.exists(log_file):
            print("⚠️ Log file not found, skipping metric logging.")
            return

        try:
             data = self.load_json_file(log_file)
             if not data or 'evaluations' not in data:
                 return

             # Get the latest evaluation session
             # In viz_only or logs_and_viz, we might be interested in the most recent run
             # For simplicity, we log the metrics of the LAST session matching our config

             # Filter sessions for current model/agent
             print(f"DEBUG: Filtering logs for model='{self.model_name}', agent='{self.agent_type}'")
             available_entries = [(s.get('model_name'), s.get('agent_type')) for s in data.get('evaluations', [])]
             print(f"DEBUG: Found {len(available_entries)} entries in log. Unique models: {set(m[0] for m in available_entries)}")

             matching_sessions = [
                 s for s in data.get('evaluations', [])
                 if s.get('model_name') == self.model_name and s.get('agent_type') == self.agent_type
             ]

             if not matching_sessions:
                 print("⚠️ No matching sessions found in log.")
                 return

             latest_session = matching_sessions[-1]
             rounds = latest_session.get('rounds', [])

             if not rounds:
                 return

             # Aggregate metrics
             gpt_scores = []
             latencies = []
             cpu_energies = []
             gpu_energies = []

             for r in rounds:
                 metrics = r.get('metrics', {})
                 breakdown = r.get('latency_breakdown', {})

                 # Score
                 score = metrics.get('gpt_judge', {}).get('score')
                 if score is not None:
                     gpt_scores.append(score)

                 # Latency
                 lat = breakdown.get('total_ms')
                 if lat is not None:
                     latencies.append(lat)

                 # Energy
                 res_diff = breakdown.get('resource_differences', {}).get('energy', {})
                 cpu = res_diff.get('cpu_power_delta_mw')
                 gpu = res_diff.get('gpu_power_delta_mw')

                 if cpu: cpu_energies.append(abs(cpu))
                 if gpu: gpu_energies.append(abs(gpu))

             # Log averages to Neptune
             if gpt_scores:
                 avg_score = sum(gpt_scores) / len(gpt_scores)
                 self.neptune.run["metrics/avg_gpt_score"] = avg_score
                 self.neptune.run["metrics/max_gpt_score"] = max(gpt_scores)

             if latencies:
                 avg_lat = sum(latencies) / len(latencies)
                 self.neptune.run["metrics/avg_latency_ms"] = avg_lat
                 self.neptune.run["metrics/min_latency_ms"] = min(latencies)
                 self.neptune.run["metrics/throughput_tok_sec"] = 1000.0 / avg_lat if avg_lat > 0 else 0

             if cpu_energies:
                 self.neptune.run["metrics/avg_cpu_power_mw"] = sum(cpu_energies) / len(cpu_energies)

             if gpu_energies:
                 self.neptune.run["metrics/avg_gpu_power_mw"] = sum(gpu_energies) / len(gpu_energies)

             print(" Scalar metrics logged to Neptune.")

        except Exception as e:
            print(f"⚠️ Failed to log metrics to Neptune: {e}")







    def get_optimisations(self):
        kv_cache = self._get_optimal_kv_cache_type()


        # FAZA 1: Wszystkie optymalizacje ODDZIELNIE dla llama-server
        individual_optimizations = [

            # {},  # Baseline

            # ========== NOWE OPTYMALIZACJE DO PRZETESTOWANIA ==========
            {"--n-gpu-layers": 99},  # Wszystko na GPU (safe: jeśli model ma <99 warstw)
            {"--n-gpu-layers": -1},  # Wszystko na GPU (auto-detect: może być szybsze)
            # {"--split-mode": "none"},  # Brak podziału (domyślne dla Metal)
            # {"--split-mode": "layer"},  # Podział warstwowy (rzadko używane na Metal)
            # {"--split-mode": "row"},  # Podział wierszowy (rzadko używane na Metal)
            # # KV Cache Quantization (kwantyzacja dla oszczędności pamięci)
            # {"--cache-type-k": "q8_0"},  # 8-bit quantization dla K cache
            # {"--cache-type-v": "q8_0"},  # 8-bit quantization dla V cache
            # {"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"},  # Oba 8-bit
            # {"--cache-type-k": "q4_0"},  # 4-bit quantization dla K cache
            # {"--cache-type-v": "q4_0"},  # 4-bit quantization dla V cache
            # {"--cache-type-k": "q4_0", "--cache-type-v": "q4_0"},  # Oba 4-bit
            # {"--cache-type-k": "q8_0", "--cache-type-v": "q4_0"},  # Mieszane: K=8bit, V=4bit
            # {"--cache-type-k": "q4_0", "--cache-type-v": "q8_0"},  # Mieszane: K=4bit, V=8bit

            # # Context & Attention Management (jeszcze nietestowane)

            # # {"--keep": 512},  # Zachowaj 512 tokenów w kontekście
            # # {"--keep": 1024},  # Zachowaj 1024 tokeny
            # # {"--no-context-shift": None},  # Wyłącz przesuwanie kontekstu

            # # RoPE (Rotary Position Embedding) optimizations
            # {"--rope-scaling": "linear"},  # Liniowe skalowanie RoPE
            # {"--rope-scaling": "yarn"},  # YaRN skalowanie
            # {"--rope-scale": 0.5},  # Zmniejsz skalę RoPE
            # {"--rope-scale": 2.0},  # Zwiększ skalę RoPE
            # {"--rope-freq-base": 10000},  # Bazowa częstotliwość RoPE
            # {"--rope-freq-scale": 0.5},  # Skalowanie częstotliwości

            # # Group Attention (GQA optimization)
            # {"--grp-attn-n": 1},  # Group attention N=1
            # {"--grp-attn-n": 2},  # Group attention N=2
            # {"--grp-attn-w": 256},  # Group attention width 256
            # {"--grp-attn-w": 512},  # Group attention width 512

            # # Cache Management (zaawansowane)
            # {"--defrag-thold": 0.1},  # Niska wartość defragmentacji
            # {"--defrag-thold": 0.5},  # Średnia wartość defragmentacji
            # {"--cache-reuse": 4},  # Ponowne użycie cache dla 4 zapytań
            # {"--cache-reuse": 8},  # Ponowne użycie cache dla 8 zapytań

            # # Memory Management
            # {"--mlock": None},  # Zablokuj model w RAM (zapobiega swappingowi)

            # # GPU Optimization (Metal na Mac M1/M2/M3)
            # # {"--n-gpu-layers": 0},  # Baseline: Wszystko na CPU
            # # {"--n-gpu-layers": 10},  # 10 warstw na GPU (Metal)
            # # {"--n-gpu-layers": 20},  # 20 warstw na GPU (Metal)
            # # {"--n-gpu-layers": 32},  # 32 warstwy na GPU (Metal)
            # {"--n-gpu-layers": 99},  # Wszystko na GPU (safe: jeśli model ma <99 warstw)
            # {"--n-gpu-layers": -1},  # Wszystko na GPU (auto-detect: może być szybsze)
            # {"--split-mode": "none"},  # Brak podziału (domyślne dla Metal)
            # {"--split-mode": "layer"},  # Podział warstwowy (rzadko używane na Metal)
            # {"--split-mode": "row"},  # Podział wierszowy (rzadko używane na Metal)

            # # CPU Affinity & Priority (dla mobilnych urządzeń)
            # {"--prio": 0},  # Normalny priorytet
            # {"--prio": 1},  # Wysoki priorytet
            # {"--poll": 50},  # Polling co 50ms
            # {"--poll": 100},  # Polling co 100ms
            # {"--numa": None},  # Włącz NUMA support

            # # CPU Masks (zaawansowane CPU affinity)
            # {"--cpu-mask": "0-3"},  # Użyj rdzeni 0-3
            # {"--cpu-mask": "0-1"},  # Użyj tylko 2 rdzeni
            # {"--cpu-range": "0-7"},  # Zakres CPU 0-7
            # {"--cpu-strict": 1},  # Ścisłe CPU affinity

            # ========== ZAKOMENTOWANE - JUŻ PRZETESTOWANE ==========

            # # Podstawowe optymalizacje KV Cache - każda osobno
            # {"--cache-type-k": kv_cache},
            # {"--cache-type-v": kv_cache},
            # {"--cache-type-k": kv_cache, "--cache-type-v": kv_cache},  # KV Cache


            # {"--flash-attn": None},  # Flash Attention
            # {"--cont-batching": None},  # Continuous Batching

            # # Mobilne optymalizacje pamięci i wydajności
            # {"--no-kv-offload": None},  # Wyłączenie KV offload dla mobile bez GPU
            # {"--no-mmap": None},  # Wyłączenie memory mapping dla constrained memory
            # {"--threads": 1},  # Single thread dla bardzo ograniczonych urządzeń
            # {"--threads": 2},  # Conservative mobile
            # {"--threads": 4},  # Standard mobile
            # {"--batch-size": 4, "--ubatch-size": 4},  # Bardzo małe batches
            # {"--batch-size": 8, "--ubatch-size": 8},  # Małe batches mobile
            # {"--batch-size": 16, "--ubatch-size": 16},  # Standard mobile batches
            # {"--batch-size": 32, "--ubatch-size": 32},  # Większe batches dla edge

            # # Speculative decoding dla mobile (jeśli masz draft model)
            # {"--draft-max": 2},  # Minimalne speculative
            # {"--draft-max": 3},  # Standard speculative
            # {"--draft-max": 4},  # Agresywne speculative
            # {"--draft-max": 5},  # Bardzo agresywne speculative
            # {"--draft-max": 3, "--draft-p-min": 0.5},  # Speculative z probability
            # {"--draft-max": 3, "--draft-p-min": 0.7},  # Bardziej konserwatywne
            # {"--draft-max": 3, "--draft-p-split": 0.3},  # Split probability

            # # Kombinacje dwuskładnikowe - KV Cache + podstawowe
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--flash-attn": None
            # },
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--cont-batching": None
            # },
            # {
            #     "--flash-attn": None,
            #     "--cont-batching": None
            # },

            # # Kombinacje z thread i batch optimization
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--threads": 4,
            #     "--batch-size": 16,
            #     "--ubatch-size": 16
            # },
            # {
            #     "--flash-attn": None,
            #     "--threads": 4,
            #     "--batch-size": 32,
            #     "--ubatch-size": 32
            # },
            # {
            #     "--cont-batching": None,
            #     "--threads": 2,
            #     "--batch-size": 8,
            #     "--ubatch-size": 8
            # },

            # # Kombinacje memory-constrained dla mobile
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--no-mmap": None,
            #     "--threads": 2
            # },
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--no-kv-offload": None,
            #     "--threads": 1
            # },
            # {
            #     "--no-mmap": None,
            #     "--no-kv-offload": None,
            #     "--threads": 2,
            #     "--batch-size": 8,
            #     "--ubatch-size": 8
            # },

            # # Kombinacje speculative z optymalizacjami (jeśli draft model dostępny)
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--draft-max": 3
            # },
            # {
            #     "--flash-attn": None,
            #     "--draft-max": 3
            # },
            # {
            #     "--cont-batching": None,
            #     "--draft-max": 3
            # },

            # # Trójkowe kombinacje - podstawowe optymalizacje
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--flash-attn": None,
            #     "--cont-batching": None
            # },
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--flash-attn": None,
            #     "--threads": 4
            # },
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--cont-batching": None,
            #     "--threads": 4
            # },

            # # Ultra-conservative mobile (bardzo ograniczone urządzenia)
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--threads": 1,
            #     "--batch-size": 4,
            #     "--ubatch-size": 4,
            #     "--no-mmap": None,
            #     "--no-kv-offload": None
            # },

            # # Standard mobile optimization
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--flash-attn": None,
            #     "--threads": 4,
            #     "--batch-size": 16,
            #     "--ubatch-size": 16
            # },

            # # Memory-priority mobile
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--no-mmap": None,
            #     "--threads": 2,
            #     "--batch-size": 8,
            #     "--ubatch-size": 8
            # },

            # # Android CPU optimized
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--cont-batching": None,
            #     "--no-kv-offload": None,
            #     "--threads": 4
            # },

            # # Edge device (Raspberry Pi style)
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--threads": 2,
            #     "--batch-size": 4,
            #     "--ubatch-size": 4,
            #     "--no-mmap": None
            # },

            # # High-performance edge
            # {
            #     "--flash-attn": None,
            #     "--cont-batching": None,
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--threads": 8,
            #     "--batch-size": 32,
            #     "--ubatch-size": 32
            # },

            # # Pełne kombinacje mobilne z speculative (gdy draft model dostępny)
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--flash-attn": None,
            #     "--draft-max": 3,
            #     "--threads": 4
            # },
            # {
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--cont-batching": None,
            #     "--draft-max": 3,
            #     "--threads": 4
            # },

            # # Wszystkie optymalizacje razem - maksymalna wydajność
            # {
            #     "--flash-attn": None,
            #     "--cont-batching": None,
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--threads": 4,
            #     "--batch-size": 16,
            #     "--ubatch-size": 16
            # },

            # # Wszystkie optymalizacje + speculative
            # {
            #     "--flash-attn": None,
            #     "--cont-batching": None,
            #     "--cache-type-k": kv_cache,
            #     "--cache-type-v": kv_cache,
            #     "--draft-max": 3,
            #     "--threads": 4,
            #     "--batch-size": 16,
            #     "--ubatch-size": 16
            # }
        ]
        # FAZA 2: selected combinations
        selected_optimisation = [
            {
                # "--cache-type-k": kv_cache,
                # "--cache-type-v": kv_cache,
                # "--flash-attn": None
            },

        ]



        print(f" Optimization plan:")
        print(f"    Individual optimizations: {len(individual_optimizations)}")
        print(f"   🔹 Combination optimizations: {len(selected_optimisation)}")


        return individual_optimizations, selected_optimisation


    def get_inference_parameter_combinations(self, mobile=False):
        """
        Generuje kombinacje parametrów inferencji do testowania.

        Returns:
            List[Dict]: Lista słowników z parametrami inferencji
        """

        # Domyślne wartości z config

        if mobile:

            # Kombinacje do testowania - pełne dictionary z wszystkimi parametrami
            parameter_combinations = [
                {"context_size": 3072, "max_tokens": 512},   # Conservative – 3-5 tur, RAM ~4-5 GB
                {"context_size": 5120, "max_tokens": 1024},  # Deterministic – 5-7 tur, RAM ~5-6 GB
                {"context_size": 7168, "max_tokens": 1536},  # High-capacity – 7-10 tur, RAM ~6-7 GB
            ]
        parameter_combinations = [
            {"context_size": 4096, "max_tokens": 1024},  # Conservative – 5 tur, RAM ~7-9 GB
            {"context_size": 8192, "max_tokens": 2048},  # Deterministic – 10 tur, RAM ~10-11 GB
            {"context_size": 12288, "max_tokens": 4096}, # High-capacity – 10+ tur, RAM ~12 GB
        ]
        return parameter_combinations

    def get_the_most_accurate_model(self):
        """
        Zwraca Golden Model z ostatniego Category Winners Analysis.
        """
        # Po prostu zwróć aktualny model - golden model będzie ustalony w main
        # po uruchomieniu plot_mobile_analysis_visualizations
        return getattr(self, '_golden_model', self.model_name)

if __name__ == "__main__":
    from edge_llm_lab.utils.base_eval import Agent
    agent_type_enum = Agent.CONSTANT_DATA_EN

    print("\n KONFIGURACJA EWALUACJI")
    print("="*50)

    execution_mode = input("Tryb: [1] Logi [2] Logi+viz [3] Viz: ")

    # Pytania tylko gdy potrzebne
    if execution_mode == "3":  # Tylko wykresy
        mode_choice = "1"  # Domyślnie wszystkie modele dla trybu viz
        language_choice = input("Język wykresów: [1] Polski [2] Angielski: ")
        use_polish = language_choice == "1"
        # Domyślne wartości dla trybu tylko wykresy
        install_choice = "n"  # Nie instaluj modeli
        optimisations_choice = "selected"  # Użyj istniejących danych
        inference_params = True  # Nie testuj parametrów
    else:  # Tryby z logami
        mode_choice = input("Modele: [1] Wszystkie [2] Tested:true bez wyników: ")
        try:
            install_choice = input("Auto-pobieranie modeli: [y/n]: ").lower().strip()
            install_choice = "y" if install_choice in ["y", "yes", "tak"] else "n"
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️ Input error, defaulting to 'n'")
            install_choice = "n"

        # Pytaj o język tylko gdy będą generowane wykresy
        if execution_mode == "2":  # Logi+viz
            language_choice = input("Język wykresów: [1] Polski [2] Angielski: ")
            use_polish = language_choice == "1"
        else:
            use_polish = True  # Default dla trybu tylko logów

        # Parametry tylko dla trybów z logami
        optimisations_choice = "selected"
        inference_params = False

        optim_mode = input("Optymalizacje: [1] Test [2] Selected: ")
        optimisations_choice = "test" if optim_mode == "1" else "selected"

        params_mode = input("Parametry inferencji: [1] Standardowe [2] Testuj: ")
        inference_params = params_mode == "2"

    # Konwersja na parametry pipeline
    mode_map = {"1": "logs_only", "2": "logs_and_viz", "3": "viz_only"}
    mode = mode_map.get(execution_mode, "logs_and_viz")

    only_tested_true = mode_choice == "2"

    print(f"\n KONFIGURACJA:")
    print(f"   Modele: {'Tested:true bez wyników' if only_tested_true else 'Wszystkie'}")
    print(f"   Tryb: {mode}")
    print(f"   Optymalizacje: {optimisations_choice}")
    print(f"   Parametry inferencji: {'Tak' if inference_params else 'Nie'}")
    print("="*50)

    # Pobierz modele
    models_to_evaluate = EvalModelsReferenced.get_truly_untested_models(
        agent_type_enum.value, "referenced", only_tested_true=only_tested_true
    )

    if not models_to_evaluate:
        print(f"Brak modeli do testowania")
        exit(1)

    print(f" Znaleziono {len(models_to_evaluate)} modeli do testowania")

    # Testuj modele
    best_model = None
    for i, model_name in enumerate(models_to_evaluate, 1):
        print(f"\n{'='*50}")
        print(f"MODEL {i}/{len(models_to_evaluate)}: {model_name}")
        print("="*50)

        if not BaseEvaluation.check_model_availability(model_name, install_choice=install_choice):
            print(f"Pomijam {model_name}")
            continue

        evaluator = EvalModelsReferenced(model_name=model_name, agent=agent_type_enum)

        print(f" Testowanie {model_name}...")
        evaluator.pipeline_eval_model(
            mode=mode,
            use_cache=True,
            optimisations_choice=optimisations_choice,
            inference_params=inference_params,
            use_polish=use_polish
        )
        del evaluator


    print(f"\n EWALUACJA ZAKOŃCZONA!")
    print(f" Wyniki: examples/desktop/output/agents/{agent_type_enum.value}/referenced/")
