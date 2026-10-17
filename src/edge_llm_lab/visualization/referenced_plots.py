"""Extracted evaluator methods; historical plotting contracts are preserved."""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt

class ReferencedPlots:
    def per_round_plots(self, session_locations, timestamp, list_of_models=None, use_polish=True):
        """
        Tworzy wykresy dla każdej rundy dla wybranych optymalizacji.
        """
        def _create_round_plots_for_single_model(model_name):
            # Filter only by model_name and agent_type
            model_comparison_dict = {
                "model_name": model_name,
                "agent_type": self.agent_type
            }
            group_by_keys = ['optimisation']

            valid_session_data = self.get_last_sessions(
                key_dict=model_comparison_dict,
                log_file=session_locations["log_file"], 
                group_by_keys=group_by_keys
            )

            # Automatically generate per-round plots without user input
            # user_input = input(f"Czy chcesz wygenerować wykres plot_per_round_with_referenc? (y/n): ").lower().strip()
            # if user_input == 'y' or user_input == 'yes':
            if True: # Always generate per-round plots

                for optimisation, session_data in valid_session_data.items():

                    for round_data in session_data['rounds']:
                        # Ask user if they want to generate this plot
                        # Sanitize optimization string for filename
                        opt_str = str(optimisation)
                        if opt_str in ["{}", "()", "((),)", "None"]:
                             opt_str = "baseline"

                        plot_name = f"{opt_str}_round_{round_data['round']}"



                        summary_plot_path = self.plot_per_round_with_reference(
                            round_data=round_data,
                            optimisation_type=optimisation,
                            model_name=model_name,
                            agent_type=self.agent_type,
                            plotting_session_timestamp=timestamp,
                            metadata=metadata,
                            output_dir=output_dir,
                            output_file_name=plot_name,
                        )
                        print(f"Summary plot saved to: {summary_plot_path}")
            else:
                print(f"Pominięto generowanie wykresu '{plot_name}'")

        if list_of_models is None:
            output_dir = session_locations["model_output_directory"]
            metadata = self.current_model_metadata
            model_name = self.model_name
            _create_round_plots_for_single_model(self.model_name)
        else:
            # Use existing session_locations instead of creating new sessions
            # This ensures plots are saved to the folder that will be uploaded
            output_dir = session_locations["model_output_directory"]
            for model_name in list_of_models:
                metadata = self.all_model_metadata.get("model", {}).get(model_name, {}) if isinstance(self.all_model_metadata, dict) else {}
                _create_round_plots_for_single_model(model_name)


    def per_model_plots(self, session_locations, timestamp, list_of_models=None, use_polish=True):
        """
        Tworzy wykresy agregowane dla modelu - jeden wykres per optymalizacja.

        """
        def _create_model_plots_for_single_model(model_name):
            # Filter only by model_name and agent_type, not by full parameters dict
            # (which may not match exactly between evaluation and visualization runs)
            model_comparison_dict = {
                "model_name": model_name,
                "agent_type": self.agent_type
            }
            group_by_keys = ['optimisation']
            valid_session_data = self.get_last_sessions(key_dict=model_comparison_dict, log_file=session_locations["log_file"], group_by_keys=group_by_keys)

            for optimisation, session_data in valid_session_data.items():
                # 1. Timeline & Heatmap Plot (using the "over rounds" function)
                timeline_plot_path = self.plot_aggr_over_rounds_with_reference(
                    session_data=session_data,
                    optimisation_type=optimisation,
                    # model_name is not accepted by this function
                    agent_type=self.agent_type,
                    plotting_session_timestamp=timestamp,
                    metadata=metadata,
                    output_dir=output_dir,
                    output_file_name=f"{optimisation}_timeline_{timestamp}",
                )
                print(f"Timeline/Heatmap plot saved to: {timeline_plot_path}")

                # 2. Single Model Summary Plot (Metrics Bar Chart)
                summary_plot_path = self.plot_single_model_summary(
                    session_data=session_data,
                    optimisation_type=optimisation,
                    model_name=model_name,
                    agent_type=self.agent_type,
                    plotting_session_timestamp=timestamp,
                    metadata=metadata,
                    output_dir=output_dir,
                    output_file_name=f"{optimisation}_summary_{timestamp}",
                )
                print(f"Summary plot saved to: {summary_plot_path}")

                # --- NEW: Throttling Timeline Plot (per optimization) ---
                if isinstance(session_data, list) and session_data:
                    self.plot_throttling_timeline(session_data[-1], output_dir, timestamp)
                elif isinstance(session_data, dict):
                    self.plot_throttling_timeline(session_data, output_dir, timestamp)

            # --- NEW: Resource Health Check Plot ---
            all_sessions_list = list(valid_session_data.values())
            self.plot_resource_health_check(all_sessions_list, output_dir, timestamp)

            valid_session_data = self.get_last_sessions(key_dict=model_comparison_dict, log_file=session_locations["log_file"], group_by_keys=group_by_keys)

            round_by_round_plot_path = self.plot_round_by_round_comparison(
                session_data=valid_session_data,
                model_name=model_name,
                agent_type=self.agent_type,
                plotting_session_timestamp=timestamp,
                metadata=metadata,
                output_dir=output_dir,
                output_file_name=f"optimisations_round_by_round_{timestamp}",
                use_polish=use_polish
            )
            print(f"Round-by-round comparison plot saved to: {round_by_round_plot_path}")

            # Generate per-group plots
            print("\n🎨 Generating per-group optimization plots...")
            group_plots = self.generate_group_plots(
                session_data=valid_session_data,
                model_name=model_name,
                agent_type=self.agent_type,
                plotting_session_timestamp=timestamp,
                metadata=metadata,
                output_dir=output_dir,
                use_polish=use_polish
            )
            for group_name, plot_path in group_plots.items():
                print(f"  ✓ {group_name} plot saved to: {plot_path}")

            from edge_llm_lab.visualization.optimization_comparison import plot_optimization_comparison, plot_gartner_style_energy_analysis

            # Standard optimization comparison
            comparison_plot_path = plot_optimization_comparison(
                session_data=valid_session_data,
                model_name=model_name,
                plotting_session_timestamp=timestamp,
                metadata=metadata,
                output_dir=output_dir,
                output_file_name=f"optimizations_comparison_{timestamp}")
            print(f"Optimization comparison plot saved to: {comparison_plot_path}")

            # Gartner-style energy analysis
            gartner_plot_path = plot_gartner_style_energy_analysis(
                session_data=valid_session_data,
                model_name=model_name,
                plotting_session_timestamp=timestamp,
                metadata=metadata,
                output_dir=output_dir,
                output_file_name=f"optimizations_comparison_{timestamp}")
            print(f"Gartner-style energy analysis plot saved to: {gartner_plot_path}")

            # Generate inference parameters comparison plot if test data exists
            import os
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
                            model_name=model_name,
                            agent_type=self.agent_type,
                            plotting_session_timestamp=timestamp,
                            output_dir=output_dir,
                            output_file_name=f"inference_params_comparison_{timestamp}"
                        )
                        if plot_path:
                            print(f"  ✓ Inference parameters comparison plot saved to: {plot_path}")
                else:
                    print("  ℹ️ No inference parameters test data found")
            else:
                print("  ℹ️ No inference parameters test file found")

        if list_of_models is None:

            output_dir = session_locations["model_output_directory"]
            metadata = self.current_model_metadata
            model_name = self.model_name
            _create_model_plots_for_single_model(self.model_name)


    def _get_model_size_gradient_colors(self, model_names, model_sizes):
        """
        Generuje gradient kolorów według rozmiaru modelu - im mniejszy model, tym jaśniejszy kolor.

        Args:
            model_names: Lista nazw modeli
            model_sizes: Lista rozmiarów modeli w GB

        Returns:
            Lista kolorów dla każdego modelu
        """
        # Filter out None values and convert to float
        valid_sizes = [size for size in model_sizes if size is not None and isinstance(size, (int, float))]

        if valid_sizes and len(set(valid_sizes)) > 1:
            # Normalizuj rozmiary do zakresu 0-1 (mniejszy = bliżej 1)
            min_size = min(valid_sizes)
            max_size = max(valid_sizes)
            # Use tab10 for maximum contrast between categorical model names
            return plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        else:
            # Fallback do tab10 jeśli brak różnic w rozmiarze
            return plt.cm.tab10(np.linspace(0, 1, len(model_names)))


    def if_quantisation_comparison(self, session_data):
        """
        Sprawdza czy modele można skrócić - szuka wspólnych prefiksów w grupach modeli sesje z poprawionymi nazwami modlei i nazwe modelu lub none


        Args:
            session_data: Lista danych sesji z modelami

        Returns:
            tuple: (nowe_session_data, wspólny_prefiks) lub (None, None) jeśli nie można skrócić
        >>> # Przykład 1: Modele z tym samym prefiksem - powinno skrócić
        >>> session_data = [
        ...     {'model_name': 'granite3.1-dense:2b-instruct-q3_K_S', 'rounds': []},
        ...     {'model_name': 'granite3.1-dense:2b-instruct-q4_0', 'rounds': []},
        ...     {'model_name': 'granite3.1-dense:2b-instruct-q8_0', 'rounds': []}
        ... ]
        >>> be = EvalModelsReferenced(model_name="granite3.1-dense:8b-instruct-q4_K_S", agent=Agent.CONSTANT_DATA_EN)
        >>> new_session_data, common_prefix = be.if_quantisation_comparison(session_data)
        >>> print([s['model_name'] for s in new_session_data])
        ['q3_K_S', 'q4_0', 'q8_0']
        >>> print(common_prefix)
        'granite3.1-dense:2b-instruct'

        >>> # Przykład 2: Modele z różnymi prefiksami - nie skróci
        >>> session_data2 = [
        ...     {'model_name': 'granite3.1-dense:2b-instruct-q3_K_S', 'rounds': []},
        ...     {'model_name': 'granite3.1-dense:8b-instruct-q4_0', 'rounds': []}
        ... ]
        >>> new_session_data2, common_prefix2 = be.if_quantisation_comparison(session_data2)
        >>> print(new_session_data2)
        None
        >>> print(common_prefix2)
        None


        """
        if not isinstance(session_data, list) or len(session_data) < 1:
            return None, None

        # Sprawdź czy można skrócić nazwy - użyj grupowania
        model_groups = self._group_models_by_prefix(session_data)

        if not model_groups:
            return None, None

        # Jeśli jest więcej niż jedna grupa, nie można skrócić
        if len(model_groups) > 1:
            return None, None

        # Pobierz jedyną grupę
        group_prefix, group_sessions = next(iter(model_groups.items()))

        # Sprawdź czy wszystkie modele w grupie mają ten sam prefiks
        model_names = [session.get('model_name', '') for session in group_sessions]
        prefixes = []
        suffix_by_name = {}

        for name in model_names:
            if '-' not in name:
                return None, None
            prefix, suffix = name.rsplit('-', 1)
            prefixes.append(prefix)
            suffix_by_name[name] = suffix

        # Sprawdź czy wszystkie prefiksy są identyczne i niepuste
        unique_prefixes = set(prefixes)
        if len(unique_prefixes) == 1:
            # Wszystkie modele mają ten sam prefiks
            common_prefix = prefixes[0]
        elif len(model_names) == 1:
            # Dla pojedynczego modela, użyj jego prefiksu
            common_prefix = prefixes[0]
        else:
            # Różne prefiksy - nie można skrócić
            return None, None

        if not common_prefix:
            return None, None

        # Stwórz kopię group_sessions z skróconymi nazwami
        new_session_data = []
        for session in group_sessions:
            original_name = session.get('model_name')
            new_session = session.copy()
            new_session['model_name'] = suffix_by_name[original_name]
            new_session_data.append(new_session)

        print(f"📝 Skrócono nazwy modeli - wspólny prefiks: '{common_prefix}'")
        print(f"📝 Skrócone nazwy: {[s['model_name'] for s in new_session_data]}")

        return new_session_data, common_prefix


    def _group_models_by_prefix(self, session_data):
        """
        Grupuje modele według prefiksu przed ostatnim '-' w nazwie modelu.

        Args:
            session_data: Lista danych sesji z modelami

        Returns:
            dict: {prefix: [sessions]} - grupy modeli pogrupowane według prefiksu

        >>> # Przykład 1: Modele z różnymi prefiksami - powinno pogrupować
        >>> session_data = [
        ...     {'model_name': 'granite3.1-dense:2b-instruct-q3_K_S', 'rounds': []},
        ...     {'model_name': 'granite3.1-dense:2b-instruct-q4_0', 'rounds': []},
        ...     {'model_name': 'granite3.1-dense:8b-instruct-q3_K_S', 'rounds': []},
        ...     {'model_name': 'granite3.1-dense:8b-instruct-q4_0', 'rounds': []}
        ... ]
        >>> be = EvalModelsReferenced(model_name="granite3.1-dense:8b-instruct-q4_K_S", agent=Agent.CONSTANT_DATA_EN)
        >>> groups = be._group_models_by_prefix(session_data)
        >>> print(list(groups.keys()))
        ['granite3.1-dense:2b-instruct', 'granite3.1-dense:8b-instruct']
        >>> print(len(groups['granite3.1-dense:2b-instruct']))
        2
        >>> print(len(groups['granite3.1-dense:8b-instruct']))
        2

        >>> # Przykład 2: Modele bez '-' - traktuje jako pojedyncze
        >>> session_data2 = [
        ...     {'model_name': 'simple_model', 'rounds': []},
        ...     {'model_name': 'another_model', 'rounds': []}
        ... ]
        >>> groups2 = be._group_models_by_prefix(session_data2)
        >>> print(list(groups2.keys()))
        ['single']
        >>> print(len(groups2['single']))
        2
        """
        model_groups = {}
        for session in session_data:
            if isinstance(session, dict):
                model_name = session.get('model_name', 'unknown_model')
                if '-' in model_name:
                    prefix = model_name.rsplit('-', 1)[0]  # Wszystko przed ostatnim '-'
                    if prefix not in model_groups:
                        model_groups[prefix] = []
                    model_groups[prefix].append(session)
                else:
                    # Model bez '-' - traktuj jako pojedynczy
                    if 'single' not in model_groups:
                        model_groups['single'] = []
                    model_groups['single'].append(session)
        return model_groups


    def _filter_sessions_by_prefix(self, session_data, selected_prefix):
        """
        Filtruje sesje według wybranego prefiksu modelu.

        Args:
            session_data: Lista danych sesji z modelami
            selected_prefix: Prefix wybranego modelu (np. 'granite3.1-dense:2b-instruct')

        Returns:
            list: Lista sesji zawierająca tylko modele z tym prefiksem

        >>> # Przykład: Filtrowanie według prefiksu
        >>> session_data = [
        ...     {'model_name': 'granite3.1-dense:2b-instruct-q3_K_S', 'rounds': []},
        ...     {'model_name': 'granite3.1-dense:2b-instruct-q4_0', 'rounds': []},
        ...     {'model_name': 'granite3.1-dense:8b-instruct-q3_K_S', 'rounds': []}
        ... ]
        >>> be = EvalModelsReferenced(model_name="test", agent=Agent.CONSTANT_DATA_EN)
        >>> filtered = be._filter_sessions_by_prefix(session_data, 'granite3.1-dense:2b-instruct')
        >>> print(len(filtered))
        2
        >>> print(filtered[0]['model_name'])
        granite3.1-dense:2b-instruct-q3_K_S
        >>> print(filtered[1]['model_name'])
        granite3.1-dense:2b-instruct-q4_0

        >>> # Przykład: Prefix nie znaleziony
        >>> filtered2 = be._filter_sessions_by_prefix(session_data, 'nonexistent_prefix')
        >>> print(len(filtered2))
        0
        """
        filtered_sessions = []
        for session in session_data:
            if isinstance(session, dict):
                model_name = session.get('model_name', '')
                # Sprawdź czy model_name zaczyna się od selected_prefix
                if model_name.startswith(selected_prefix + '-') or model_name == selected_prefix:
                    filtered_sessions.append(session)
        return filtered_sessions


    def display_models_and_get_selection(self, session_data, interactive=True):
        """
        Wyświetla listę wykrytych grup modeli i pozwala użytkownikowi wybrać grupę lub konkretny model.

        Args:
            session_data: Lista danych sesji z modelami
            interactive: Czy pytać użytkownika (True) czy wybrać wszystko (False)

        Returns:
            tuple: (wybrany_session_data, nazwa_modelu) lub (None, None) jeśli anulowano
        """
        if not session_data:
            print("Nie znaleziono danych modeli do wyświetlenia")
            return None, None

        # Wyciągnij unikalne prefiksy (rodziny modeli)
        groups = self._group_models_by_prefix(session_data)

        if not groups:
            print("Nie znaleziono grup modeli")
            return None, None

        if not interactive:
            print("Automatycznie wybrano wszystkie modele (tryb nieinteraktywny)")
            return session_data, None

        print("\nWybierz grupę modeli:")
        print("0. Wszystkie modele (All models)")

        group_list = sorted(list(groups.keys()))
        for i, group in enumerate(group_list):
            print(f"{i+1}. {group}")

        while True:
            try:
                choice = input("\nWybierz numer (0-{}): ".format(len(group_list))).strip()
                if not choice:
                    choice = "0"

                choice_int = int(choice)
                if choice_int == 0:
                    print("Wybrano wszystkie modele")
                    return session_data, None
                elif 1 <= choice_int <= len(group_list):
                    selected_prefix = group_list[choice_int-1]
                    selected_sessions = groups[selected_prefix]
                    print(f"Wybrano grupę: {selected_prefix}")
                    return selected_sessions, selected_prefix
                else:
                    print(f"Nieprawidłowy numer. Wybierz od 0 do {len(group_list)}")
            except ValueError:
                print("Wprowadź prawidłowy numer")
            except (KeyboardInterrupt, EOFError):
                print("\nPrzerwano - wybrano wszystkie modele")
                return session_data, None


    def all_models_plots(self, session_locations, timestamp, use_polish=True, interactive=True):
        """
        Tworzy wykresy porównujące WSZYSTKIE modele dla każdej optymalizacji.

        """

        from collections import defaultdict
        # Filter only by agent_type for all models comparison
        all_models_comparison_dict = {
            "agent_type": self.agent_type
        }
        group_by_keys = ['optimisation', 'model_name']
        # Get ALL models for each optimization (not just latest session)
        valid_session_data = self.get_last_sessions(key_dict=all_models_comparison_dict, log_file=session_locations["log_file"], group_by_keys=group_by_keys)
        # filter optimisations
        output_dir = session_locations["all_models_output_directory"]
        # Generate a single timestamp for all files in this run
        metadata=self.all_model_metadata
        self._current_use_polish = use_polish
        # Aggregate ALL models and ALL optimizations into ONE plot
        all_session_data = []
        for session in valid_session_data.values():
            m_name = session.get('model_name', '').lower()
            # Strict filter for MOE models as they distort statistics
            if 'moe' in m_name:
                print(f"⏩ Globally filtering out failed MOE model from analysis: {m_name}")
                continue
            all_session_data.append(session)

        # Pozwól użytkownikowi wybrać grupę lub konkretny model
        selected_session_data, selected_model_name = self.display_models_and_get_selection(all_session_data, interactive=interactive)

        if selected_session_data is None:
            print("Anulowano generowanie wykresów")
            return

        # Sprawdź czy można skrócić nazwy modeli
        print(f" DEBUG: selected_model_name = {selected_model_name}")
        if selected_model_name is None:  # Wybrano wszystkie modele - zostaw pełne nazwy
            all_session_data = selected_session_data
            model_name = None  # Brak wspólnego prefiksu dla wielu modeli
            print(f" DEBUG: Wybrano wszystkie modele, model_name = {model_name}")
        else:  # Wybrano konkretną grupę (prefix)
            # Przefiltruj all_session_data według wybranego prefiksu
            filtered_session_data = self._filter_sessions_by_prefix(all_session_data, selected_model_name)

            # DEBUG: Pokaż co zostało wybrane
            print(f"\n DEBUG: Wybrano prefix: {selected_model_name}")
            print(f" DEBUG: Liczba sesji po filtrze: {len(filtered_session_data)}")
            if filtered_session_data:
                print(f" DEBUG: Przykładowy model: {filtered_session_data[0].get('model_name', 'unknown')}")

            # Sprawdź czy można skrócić nazwy
            new_sess, model_name = self.if_quantisation_comparison(filtered_session_data)
            if new_sess is not None:  # Udało się skrócić
                all_session_data = new_sess
            else:  # Nie udało się skrócić - użyj oryginalnej nazwy
                all_session_data = filtered_session_data
                model_name = selected_model_name

        print(f" DEBUG FINAL: model_name = {model_name}")
        # Generate ONE set of plots with ALL models and ALL optimizations

        # 3. Generate model comparison plot (ALL models, ALL optimizations)
        comparison_plot_path = self.plot_aggr_all_models_with_reference(
            session_data=all_session_data,
            optimisation_type="all_models_all_optimizations",
            agent_type=self.agent_type,
            plotting_session_timestamp=timestamp,
            metadata=metadata,
            output_dir=output_dir,
            output_file_name=f"all_models_with_reference_{timestamp}",
            use_polish=use_polish,
            model_name_prefix = model_name,

        )
        print(f" Comparison plot saved to: {comparison_plot_path}")

        # 4. Generate mobile analysis visualizations (ALL models, ALL optimizations)
        mobile_plots = self.plot_mobile_analysis_visualizations(
            session_data=all_session_data,
            optimisation_type="all_models_all_optimizations",
            agent_type=self.agent_type,
            plotting_session_timestamp=timestamp,
            metadata=metadata,
            output_dir=output_dir,
            output_file_name=f"all_models_mobile_analysis_{timestamp}",
            use_polish=use_polish,
            model_name_prefix = model_name,
        )
        print(f"📱 Mobile analysis saved to: {mobile_plots}")

        # Category winners are already generated inside plot_mobile_analysis_visualizations
        if isinstance(mobile_plots, dict) and 'category_winners' in mobile_plots:
            print(f" Category winners saved to: {mobile_plots['category_winners']}")

        # 5. Generate Resource Health Check (RAM/SWAP/Throttling) for all models
        try:
            print("\nGenerowanie diagnostyki Resource Health Check...")
            self.plot_resource_health_check(all_session_data, output_dir, timestamp)
        except Exception as e:
            print(f"Error in plot_resource_health_check: {str(e)}")

        # 6. Generate individual throttling timelines for each model
        try:
            print("\nGenerowanie przebiegów czasowych (throttling timelines)...")
            for session in all_session_data:
                self.plot_throttling_timeline(session, output_dir, timestamp)
        except Exception as e:
            print(f"Error in plot_throttling_timeline: {str(e)}")


    @staticmethod
    def generate_group_plots(session_data, model_name, agent_type, plotting_session_timestamp, metadata, output_dir, use_polish=True):
        """
        Generuje osobne wykresy dla każdej grupy optymalizacji.

        Args:
            session_data: Dict z optymalizacjami {optimization_key: session_data}
            model_name: Nazwa modelu
            agent_type: Typ agenta
            plotting_session_timestamp: Znacznik czasowy
            metadata: Metadane modelu
            output_dir: Katalog wyjściowy

        Returns:
            Dict: {group_name: plot_path}
        """
        if not session_data:
            print("No session data available")
            return {}

        # Przygotuj dane dla wszystkich optymalizacji
        all_optimizations = {}

        for opt_key, session in session_data.items():
            rounds_data = session.get('rounds', [])
            if not rounds_data:
                continue

            # Wyciągnij metryki z każdej rundy dla tej optymalizacji
            latencies = []
            throughputs = []
            gpt_scores = []
            memory_usage = []
            cpu_usage = []

            for round_data in rounds_data:
                latency_breakdown = round_data.get('latency_breakdown', {})
                total_ms = latency_breakdown.get('total_ms')
                tokens = latency_breakdown.get('tokens', {})
                throughput = tokens.get('throughput_tokens_per_sec')

                metrics = round_data.get('metrics', {})
                gpt_judge = metrics.get('gpt_judge', {})
                gpt_score = gpt_judge.get('score', 0) * 100

                # Use resource_differences for energy data (delta values)
                resource_differences = latency_breakdown.get('resource_differences', {})
                memory_diff = resource_differences.get('memory', {})
                energy_diff = resource_differences.get('energy', {})

                ram_used_gb = memory_diff.get('ram_delta_gb', 0)  # Change in RAM usage
                cpu_power_mw = abs(energy_diff.get('cpu_power_delta_mw', 0) or 0)  # Absolute delta value

                latencies.append(total_ms)
                throughputs.append(throughput)
                gpt_scores.append(gpt_score)
                memory_usage.append(ram_used_gb)
                cpu_usage.append(cpu_power_mw)

            # Sformatuj nazwę optymalizacji
            opt_name = str(session.get('optimisation', opt_key))
            if opt_name == '{}':
                opt_name = 'Baseline'

            all_optimizations[opt_name] = {
                'latencies': latencies,
                'throughputs': throughputs,
                'gpt_scores': gpt_scores,
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage
            }

        # Znajdź baseline
        baseline_data = None
        baseline_data_dict = None
        for opt_name, data in all_optimizations.items():
            if opt_name == 'Baseline':
                baseline_data = (opt_name, data)
                baseline_data_dict = data
                break

        # Grupuj optymalizacje według nowych kategorii
        groups = {}

        # 1. Grupa "Improved" - tylko optymalizacje które poprawiły wyniki
        improved_opts = {}
        for opt_name, data in all_optimizations.items():
            if opt_name == 'Baseline':
                continue
            if ReferencedPlots.is_optimization_improved(data, baseline_data_dict):
                improved_opts[opt_name] = data

        if improved_opts:
            groups['Improved'] = improved_opts

        # 2. Grupowanie per liczba optymalizacji (1, 2, 3, etc.)
        count_groups = {}
        for opt_name, data in all_optimizations.items():
            if opt_name == 'Baseline':
                continue
            count = ReferencedPlots.count_optimizations_in_name(opt_name)
            count_group_name = f'Count_{count}'

            if count_group_name not in count_groups:
                count_groups[count_group_name] = {}
            count_groups[count_group_name][opt_name] = data

        groups.update(count_groups)

        # 3. Grupowanie per typ spekulacji (tylko czyste grupy)
        pure_type_groups = {}
        for opt_name, data in all_optimizations.items():
            if opt_name == 'Baseline':
                continue
            pure_type = ReferencedPlots.get_pure_optimization_type(opt_name)

            # Tylko dodaj jeśli to nie jest 'Mixed' ani 'Other'
            if pure_type not in ['Mixed', 'Other', 'Baseline']:
                pure_group_name = f'Pure_{pure_type}'
                if pure_group_name not in pure_type_groups:
                    pure_type_groups[pure_group_name] = {}
                pure_type_groups[pure_group_name][opt_name] = data

        groups.update(pure_type_groups)

        # 4. Grupa "All" - wszystkie optymalizacje razem
        all_opts = {}
        for opt_name, data in all_optimizations.items():
            if opt_name != 'Baseline':
                all_opts[opt_name] = data

        if all_opts:
            groups['All'] = all_opts

        # Generuj wykres dla każdej grupy
        group_plot_paths = {}

        for group_name, group_optimizations in groups.items():
            if not group_optimizations:
                continue

            # Sanitize group name for filename
            safe_group_name = group_name.lower().replace(' ', '_').replace('/', '_')
            output_file_name = f"group_{safe_group_name}_{plotting_session_timestamp}"

            plot_path = ReferencedPlots.plot_optimization_group(
                group_name=group_name,
                group_optimizations=group_optimizations,
                baseline_data=baseline_data,
                model_name=model_name,
                agent_type=agent_type,
                plotting_session_timestamp=plotting_session_timestamp,
                metadata=metadata,
                output_dir=output_dir,
                output_file_name=output_file_name,
                use_polish=use_polish
            )

            if plot_path:
                group_plot_paths[group_name] = plot_path

        return group_plot_paths


    @staticmethod
    def get_optimization_groups():
        """
        Zwraca definicje semantycznych grup optymalizacji z kolorami bazowymi.
        Kolory będą automatycznie przyciemniane/rozjaśniane alfabetycznie dla wariantów w grupie.
        UWAGA: Kolejność grup ma znaczenie - bardziej specyficzne grupy MUSZĄ być pierwsze!
        """
        return {
            'Cache Optimization': {
                'color': '#1f77b4',  # blue - wszystko z cache
                'patterns': [
                    'cache-type-k', 'cache-type-v',  # Cache types
                    'defrag-thold',  # KV cache defragmentation
                    'cache-reuse',  # Cache reuse
                    'no-kv-offload'  # KV offload control
                ]
            },
            'Context & Attention': {
                'color': '#d62728',  # red - context, attention, RoPE
                'patterns': [
                    'flash-attn',  # Flash attention (NAJPIERW!)
                    'grp-attn-n', 'grp-attn-w',  # Group attention
                    'ctx-size', 'keep', 'no-context-shift',  # Context management
                    'rope-scaling', 'rope-scale', 'rope-freq-base', 'rope-freq-scale'  # RoPE
                ]
            },
            'Batch Processing': {
                'color': '#e377c2',  # pink - batching i parallel
                'patterns': [
                    'cont-batching',  # Continuous batching
                    'ubatch-size', 'batch-size',  # Batch sizes
                    'parallel', 'sequences'  # Parallel processing
                ]
            },
            'Speculative Decoding': {
                'color': '#17becf',  # cyan - speculative decoding
                'patterns': [
                    'draft-max', 'draft-min',  # Draft sizes
                    'draft-p-split', 'draft-p-min'  # Draft probabilities
                ]
            },
            'Memory & Storage': {
                'color': '#9467bd',  # purple - pamięć i storage
                'patterns': [
                    'no-mmap',  # Memory mapping
                    'mlock'  # Memory locking
                ]
            },
            'Hardware Optimization': {
                'color': '#7f7f7f',  # gray - CPU/GPU/Threading (NA KOŃCU!)
                'patterns': [
                    'gpu-layers', 'n-gpu-layers', 'split-mode',  # GPU
                    'threads', 'threads-batch',  # Threading
                    'prio', 'poll',  # Priority
                    'cpu-mask', 'cpu-range', 'cpu-strict',  # CPU Affinity
                    'numa'  # NUMA
                ]
            }
        }


    @staticmethod
    def categorize_optimization(opt_name):
        """
        Kategoryzuje nazwę optymalizacji do grupy semantycznej.

        Returns:
            tuple: (group_name, variant_value) lub (None, None) jeśli baseline
        """
        if opt_name == 'Baseline':
            return None, None

        groups = ReferencedPlots.get_optimization_groups()

        for group_name, group_info in groups.items():
            for pattern in group_info['patterns']:
                if pattern in opt_name:
                    # Wyciągnij wartość wariantu jeśli istnieje
                    variant_value = None
                    if '=' in opt_name:
                        try:
                            variant_str = opt_name.split('=')[-1].strip()
                            if variant_str.isdigit():
                                variant_value = int(variant_str)
                            else:
                                variant_value = variant_str
                        except:
                            pass

                    return group_name, variant_value

        return 'Other', None


    @staticmethod
    def is_optimization_improved(opt_data, baseline_data):
        """
        Sprawdza czy optymalizacja poprawiła wyniki względem baseline.

        Args:
            opt_data: Dict z danymi optymalizacji
            baseline_data: Dict z danymi baseline

        Returns:
            bool: True jeśli optymalizacja jest lepsza od baseline
        """
        if not baseline_data:
            return False

        # Funkcja pomocnicza do obliczania średniej bez None
        def safe_avg(values):
            valid = [v for v in values if v is not None]
            return sum(valid) / len(valid) if valid else 0

        # Porównaj średnie wyniki (pomijając None)
        opt_avg_latency = safe_avg(opt_data['latencies'])
        baseline_avg_latency = safe_avg(baseline_data['latencies'])

        opt_avg_throughput = safe_avg(opt_data['throughputs'])
        baseline_avg_throughput = safe_avg(baseline_data['throughputs'])

        opt_avg_gpt_score = safe_avg(opt_data['gpt_scores'])
        baseline_avg_gpt_score = safe_avg(baseline_data['gpt_scores'])

        # Optymalizacja jest "improved" jeśli:
        # - Ma niższy latency (lepsze) LUB
        # - Ma wyższy throughput (lepsze) LUB
        # - Ma wyższy GPT score (lepsze)
        # - I nie ma znacząco gorszych wyników w innych metrykach

        latency_improved = opt_avg_latency < baseline_avg_latency * 0.95  # 5% lepsze
        throughput_improved = opt_avg_throughput > baseline_avg_throughput * 1.05  # 5% lepsze
        score_improved = opt_avg_gpt_score > baseline_avg_gpt_score + 2  # 2 punkty lepsze

        # Sprawdź czy nie ma drastycznego pogorszenia w innych metrykach
        latency_not_much_worse = opt_avg_latency < baseline_avg_latency * 1.2  # max 20% gorsze
        throughput_not_much_worse = opt_avg_throughput > baseline_avg_throughput * 0.8  # max 20% gorsze
        score_not_much_worse = opt_avg_gpt_score > baseline_avg_gpt_score - 5  # max 5 punktów gorsze

        return (latency_improved or throughput_improved or score_improved) and \
               latency_not_much_worse and throughput_not_much_worse and score_not_much_worse


    @staticmethod
    def count_optimizations_in_name(opt_name):
        """
        Liczy ile optymalizacji jest w nazwie (po przecinkach i nawiasach).

        Args:
            opt_name: Nazwa optymalizacji

        Returns:
            int: Liczba optymalizacji
        """
        if opt_name == 'Baseline':
            return 0

        # Usuń nawiasy i podziel po przecinkach
        clean_name = opt_name.replace('(', '').replace(')', '').replace("'", '')
        parts = [part.strip() for part in clean_name.split(',') if part.strip()]

        # Licz unikalne optymalizacje (usuń duplikaty)
        unique_opts = set()
        for part in parts:
            if '=' in part:
                opt_key = part.split('=')[0].strip()
                unique_opts.add(opt_key)
            else:
                unique_opts.add(part)

        return len(unique_opts)


    @staticmethod
    def get_pure_optimization_type(opt_name):
        """
        Sprawdza czy optymalizacja składa się tylko z optymalizacji z jednej grupy.

        Args:
            opt_name: Nazwa optymalizacji

        Returns:
            str: Nazwa grupy jeśli wszystkie optymalizacje są z tej samej grupy, None jeśli mieszane
        """
        if opt_name == 'Baseline':
            return 'Baseline'

        # Usuń nawiasy i podziel po przecinkach
        clean_name = opt_name.replace('(', '').replace(')', '').replace("'", '')
        parts = [part.strip() for part in clean_name.split(',') if part.strip()]

        # Określ grupy dla każdej optymalizacji
        groups_found = set()
        for part in parts:
            if '=' in part:
                opt_key = part.split('=')[0].strip()
            else:
                opt_key = part

            group_name, _ = ReferencedPlots.categorize_optimization(opt_key)
            if group_name:
                groups_found.add(group_name)

        # Jeśli wszystkie optymalizacje są z tej samej grupy, zwróć nazwę grupy
        if len(groups_found) == 1:
            return list(groups_found)[0]
        elif len(groups_found) > 1:
            return 'Mixed'  # Mieszane grupy
        else:
            return 'Other'


    @staticmethod
    def get_optimization_color(opt_name, all_group_opts=None, group_name=None):
        """
        Zwraca kolor dla optymalizacji używając spójnych palet kolorów per grupa.

        Args:
            opt_name: Nazwa optymalizacji
            all_group_opts: Lista wszystkich optymalizacji w tej samej grupie (dla gradientu)
            group_name: Nazwa grupy (dla specjalnych grup jak 'Improved', 'Count_X', 'Pure_X', 'All')

        Returns:
            tuple: (color, order, linewidth, alpha)
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if opt_name == 'Baseline':
            return '#d62728', 1000, 3.5, 1.0  # red, last, thick, opaque

        # Definicja palet kolorów dla grup semantycznych (nie organizacyjnych!)
        # Używamy różnych palet dla lepszego rozróżnienia
        semantic_group_palettes = {
            'Cache Optimization': 'viridis',
            'Context & Attention': 'plasma',
            'Batch Processing': 'magma',
            'Speculative Decoding': 'inferno',
            'Memory & Storage': 'cividis',
            'Hardware Optimization': 'tab10',
            'Other': 'Set1'  # Użyj Set1 zamiast gray dla lepszego rozróżnienia
        }

        # ZAWSZE używaj grupy semantycznej optymalizacji, nie organizacyjnej kategorii
        detected_group, _ = ReferencedPlots.categorize_optimization(opt_name)
        palette_name = semantic_group_palettes.get(detected_group, 'gray')

        # Pobierz paletę kolorów
        if palette_name == 'gray':
            colors = ['#7f7f7f']
        else:
            try:
                # Użyj różnych palet dla lepszego rozróżnienia
                available_palettes = ['viridis', 'plasma', 'magma', 'inferno', 'cividis', 'tab10', 'Set1', 'Set2', 'Set3']
                palette_idx = hash(opt_name) % len(available_palettes)
                actual_palette = available_palettes[palette_idx]

                # Użyj nowej składni matplotlib
                cmap = plt.colormaps.get_cmap(actual_palette)
                colors = [cmap(i) for i in range(256)]  # 256 kolorów z palety
            except:
                colors = ['#7f7f7f']  # fallback

        # Oblicz indeks koloru w palecie - użyj hash dla bardziej różnorodnych kolorów
        if all_group_opts and len(all_group_opts) > 1:
            sorted_opts = sorted(all_group_opts)
            try:
                opt_idx = sorted_opts.index(opt_name)
                # Użyj hash nazwy optymalizacji dla bardziej różnorodnych kolorów
                hash_val = hash(opt_name) % 1000  # Hash 0-999
                color_idx = hash_val / 1000.0  # Normalizuj do 0.0-1.0
            except ValueError:
                color_idx = 0.5
        else:
            # Dla pojedynczych optymalizacji, użyj hash
            hash_val = hash(opt_name) % 1000
            color_idx = hash_val / 1000.0

        # Pobierz kolor z palety
        if palette_name == 'gray':
            color = colors[0]
        else:
            color_idx_int = int(color_idx * (len(colors) - 1))
            color = colors[color_idx_int]

        # Ustaw order i style
        order = 100 + hash(opt_name) % 100
        linewidth = 2.5
        alpha = 0.85

        return color, order, linewidth, alpha


    @staticmethod
    def plot_optimization_group(group_name, group_optimizations, baseline_data, model_name, agent_type, plotting_session_timestamp, metadata, output_dir, output_file_name, use_polish=True):
        """
        Tworzy wykres dla jednej grupy optymalizacji + baseline.

        Args:
            group_name: Nazwa grupy optymalizacji
            group_optimizations: Dict z optymalizacjami w grupie {opt_name: data}
            baseline_data: Tuple (opt_name, data) dla baseline
            model_name: Nazwa modelu
            agent_type: Typ agenta
            plotting_session_timestamp: Znacznik czasowy
            metadata: Metadane modelu
            output_dir: Katalog wyjściowy
            output_file_name: Nazwa pliku wyjściowego

        Returns:
            str: Ścieżka do wygenerowanego wykresu
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if not group_optimizations and not baseline_data:
            print(f"No data for group: {group_name}")
            return None

        # Twórz wykres - STAŁY ROZMIAR z 5 wykresami
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(14, 25))

        # Add agent to title
        agent_name = agent_type.value if hasattr(agent_type, 'value') else str(agent_type)
        title_suffix = f" - {model_name}" if model_name else ""
        fig.suptitle(f'{group_name} Performance Comparison\n{title_suffix} | {agent_name}', 
                     fontsize=16, fontweight='bold', y=0.98)

        # Get all optimization names in this group for color gradient
        all_opt_names = list(group_optimizations.keys())

        # Sort optimizations alphabetically for consistent ordering
        sorted_opts = []
        for opt_name, data in group_optimizations.items():
            color, order, linewidth, alpha = ReferencedPlots.get_optimization_color(opt_name, all_opt_names, group_name)
            sorted_opts.append((opt_name, data, color, order, linewidth, alpha))

        sorted_opts.sort(key=lambda x: x[0])  # Sort by name (alphabetically)

        # Plot group optimizations
        for opt_name, data, color, order, linewidth, alpha in sorted_opts:
            rounds = list(range(1, len(data['latencies']) + 1))

            # Latency
            ax1.plot(rounds, data['latencies'], 
                    marker='o', linewidth=linewidth, markersize=6,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

            # Throughput
            ax2.plot(rounds, data['throughputs'],
                    marker='s', linewidth=linewidth, markersize=6, 
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

            # GPT Score
            ax3.plot(rounds, data['gpt_scores'], 
                    marker='^', linewidth=linewidth, markersize=6,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

            # Memory
            ax4.plot(rounds, data['memory_usage'], 
                    marker='d', linewidth=linewidth, markersize=6,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

            # CPU Usage
            ax5.plot(rounds, data['cpu_usage'], 
                    marker='v', linewidth=linewidth, markersize=6,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        # Plot baseline last (on top)
        if baseline_data:
            opt_name, data = baseline_data
            color, order, linewidth, alpha = ReferencedPlots.get_optimization_color(opt_name)
            rounds = list(range(1, len(data['latencies']) + 1))

            ax1.plot(rounds, data['latencies'], 
                    marker='o', linewidth=linewidth, markersize=8,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

            ax2.plot(rounds, data['throughputs'],
                    marker='s', linewidth=linewidth, markersize=8, 
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

            ax3.plot(rounds, data['gpt_scores'], 
                    marker='^', linewidth=linewidth, markersize=8,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

            ax4.plot(rounds, data['memory_usage'], 
                    marker='d', linewidth=linewidth, markersize=8,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

            ax5.plot(rounds, data['cpu_usage'], 
                    marker='v', linewidth=linewidth, markersize=8,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        # Styling z wyborem języka
        if use_polish:
            ax1.set_xlabel('Runda')
            ax1.set_ylabel('Całkowita latencja (ms)')
            ax1.set_title('Latencja na rundę')
            ax2.set_xlabel('Runda')
            ax2.set_ylabel('Przepustowość (tokeny/s)')
            ax2.set_title('Przepustowość na rundę')
            ax3.set_xlabel('Runda')
            ax3.set_ylabel('Wynik GPT Judge (%)')
            ax3.set_title('Wynik GPT Judge na rundę')
            ax4.set_xlabel('Runda')
            ax4.set_ylabel('Użycie RAM (GB)')
            ax4.set_title('Użycie pamięci na rundę')
            ax5.set_xlabel('Runda')
            ax5.set_ylabel('Moc CPU (mW)')
            ax5.set_title('Moc CPU na rundę')
        else:
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Total Latency (ms)')
            ax1.set_title('Latency per Round')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Throughput (tokens/sec)')
            ax2.set_title('Throughput per Round')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('GPT Judge Score (%)')
            ax3.set_title('GPT Judge Score per Round')
            ax4.set_xlabel('Round')
            ax4.set_ylabel('RAM Usage (GB)')
            ax4.set_title('Memory Usage per Round')
            ax5.set_xlabel('Round')
            ax5.set_ylabel('CPU Power (mW)')
            ax5.set_title('CPU Power per Round')

        # Konfiguracja wszystkich wykresów
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')

        # Legenda na samym dole wykresu, poza obszarem wykresów
        # Użyj figlegend dla lepszego rozmieszczenia
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=min(4, len(labels)), fontsize=8, frameon=True, fancybox=True, shadow=True)

        # Dynamic y-axis: find min/max from all data and add margin
        all_gpt_scores = []
        for opt_name, data, *_ in sorted_opts:
            all_gpt_scores.extend(data['gpt_scores'])
        if baseline_data:
            all_gpt_scores.extend(baseline_data[1]['gpt_scores'])
        if all_gpt_scores:
            min_score = max(0, min(all_gpt_scores) - 10)
            max_score = min(100, max(all_gpt_scores) + 5)
            ax3.set_ylim([min_score, max_score])

        # Dostosuj layout z miejscem na legendę na dole i tytuł na górze
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Add timestamp
        fig.text(0.99, 0.01, f'Generated: {plotting_session_timestamp}', 
                ha='right', fontsize=8, style='italic')

        # Save plot
        plot_filename = f"{output_file_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path


    @staticmethod
    def plot_round_by_round_comparison(session_data, model_name, agent_type, plotting_session_timestamp, metadata, output_dir, output_file_name, use_polish=True):
        """
        Tworzy wykres porównujący wszystkie optymalizacje runda po rundzie.

        Args:
            session_data: Dict z optymalizacjami {optimization_key: session_data}
            model_name: Nazwa modelu
            agent_type: Typ agenta
            plotting_session_timestamp: Znacznik czasowy
            metadata: Metadane modelu
            output_dir: Katalog wyjściowy
            output_file_name: Nazwa pliku wyjściowego
            use_polish: Użyj polskich etykiet (domyślnie True)

        Returns:
            str: Ścieżka do wygenerowanego wykresu
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime
        import os

        if not session_data:
            print("No session data available")
            return None

        # Przygotuj dane dla wszystkich optymalizacji
        all_optimizations = {}

        for opt_key, session in session_data.items():
            rounds_data = session.get('rounds', [])
            if not rounds_data:
                continue

            # Wyciągnij metryki z każdej rundy dla tej optymalizacji
            latencies = []
            throughputs = []
            gpt_scores = []
            memory_usage = []  # Dodaj dane o pamięci
            cpu_usage = []  # Dodaj dane o CPU

            for round_data in rounds_data:
                latency_breakdown = round_data.get('latency_breakdown', {})
                total_ms = latency_breakdown.get('total_ms')
                tokens = latency_breakdown.get('tokens', {})
                throughput = tokens.get('throughput_tokens_per_sec')

                # Get GPT score if available (convert to percentage)
                metrics = round_data.get('metrics', {})
                gpt_judge = metrics.get('gpt_judge', {})
                gpt_score = gpt_judge.get('score', 0) * 100  # Convert from 0-10 to 0-100%

                # Get resource usage if available
                # Use resource_differences for energy data (delta values)
                resource_differences = latency_breakdown.get('resource_differences', {})
                memory_diff = resource_differences.get('memory', {})
                energy_diff = resource_differences.get('energy', {})

                ram_used_gb = memory_diff.get('ram_delta_gb', 0)  # Change in RAM usage
                cpu_power_mw = abs(energy_diff.get('cpu_power_delta_mw', 0) or 0)  # Absolute delta value

                latencies.append(total_ms)
                throughputs.append(throughput)
                gpt_scores.append(gpt_score)
                memory_usage.append(ram_used_gb)
                cpu_usage.append(cpu_power_mw)

            # Sformatuj nazwę optymalizacji
            opt_name = str(session.get('optimisation', opt_key))
            if opt_name == '{}':
                opt_name = 'Baseline'

            all_optimizations[opt_name] = {
                'latencies': latencies,
                'throughputs': throughputs,
                'gpt_scores': gpt_scores,
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage
            }

        if not all_optimizations:
            print("No valid optimization data found")
            return None

        # Twórz wykres z dodatkowym subplotem dla zasobów - STAŁY ROZMIAR z 5 wykresami
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(14, 25))

        # Add agent to title
        agent_name = agent_type.value if hasattr(agent_type, 'value') else str(agent_type)
        if use_polish:
            title_suffix = f" - {model_name}" if model_name else ""
            fig.suptitle(f'Wszystkie optymalizacje - Porównanie runda po rundzie\n{title_suffix} | {agent_name}', 
                         fontsize=16, fontweight='bold', y=0.98)
        else:
            fig.suptitle(f'All Optimizations - Round-by-Round Comparison\n{title_suffix} | {agent_name}', 
                         fontsize=16, fontweight='bold', y=0.98)

        # Sortuj optymalizacje: najpierw wszystkie poza Baseline, na końcu Baseline
        baseline_data = None
        other_optimizations = []

        # Grupuj optymalizacje według grupy dla poprawnego gradientu kolorów
        groups_dict = {}
        for opt_name, data in all_optimizations.items():
            if opt_name == 'Baseline':
                baseline_data = (opt_name, data)
            else:
                group_name, _ = ReferencedPlots.categorize_optimization(opt_name)
                if group_name not in groups_dict:
                    groups_dict[group_name] = []
                groups_dict[group_name].append((opt_name, data))

        # Dla każdej grupy, utwórz kolory z gradientem
        for group_name, group_opts_list in groups_dict.items():
            # Wyciągnij nazwy dla gradientu
            all_group_names = [opt[0] for opt in group_opts_list]

            for opt_name, data in group_opts_list:
                color, order, linewidth, alpha = ReferencedPlots.get_optimization_color(opt_name, all_group_names)
                other_optimizations.append((opt_name, data, color, order, linewidth, alpha))

        # Sortuj alfabetycznie
        other_optimizations.sort(key=lambda x: x[0])

        # Wykres 1: Latencja
        for opt_name, data, color, order, linewidth, alpha in other_optimizations:
            rounds = list(range(1, len(data['latencies']) + 1))
            ax1.plot(rounds, data['latencies'], 
                    marker='o', linewidth=linewidth, markersize=6,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        # Narysuj Baseline na końcu (na wierzchu)
        if baseline_data:
            opt_name, data = baseline_data
            color, order, linewidth, alpha = ReferencedPlots.get_optimization_color(opt_name)
            rounds = list(range(1, len(data['latencies']) + 1))
            ax1.plot(rounds, data['latencies'], 
                    marker='o', linewidth=linewidth, markersize=8,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        # Styling z wyborem języka
        if use_polish:
            ax1.set_xlabel('Runda')
            ax1.set_ylabel('Całkowita latencja (ms)')
            ax1.set_title('Latencja na rundę')
        else:
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Total Latency (ms)')
            ax1.set_title('Latency per Round')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')

        # Wykres 2: Throughput
        for opt_name, data, color, order, linewidth, alpha in other_optimizations:
            rounds = list(range(1, len(data['throughputs']) + 1))
            ax2.plot(rounds, data['throughputs'],
                    marker='s', linewidth=linewidth, markersize=6, 
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        if baseline_data:
            opt_name, data = baseline_data
            color, order, linewidth, alpha = ReferencedPlots.get_optimization_color(opt_name)
            rounds = list(range(1, len(data['throughputs']) + 1))
            ax2.plot(rounds, data['throughputs'],
                    marker='s', linewidth=linewidth, markersize=8, 
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        if use_polish:
            ax2.set_xlabel('Runda')
            ax2.set_ylabel('Przepustowość (tokeny/s)')
            ax2.set_title('Przepustowość na rundę')
        else:
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Throughput (tokens/sec)')
            ax2.set_title('Throughput per Round')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')

        # Wykres 3: GPT Judge Score
        for opt_name, data, color, order, linewidth, alpha in other_optimizations:
            rounds = list(range(1, len(data['gpt_scores']) + 1))
            ax3.plot(rounds, data['gpt_scores'], 
                    marker='^', linewidth=linewidth, markersize=6,
                    color=color, alpha=alpha,
                    label=f'{opt_name} - GPT Judge', zorder=order)

        if baseline_data:
            opt_name, data = baseline_data
            color, order, linewidth, alpha = ReferencedPlots.get_optimization_color(opt_name)
            rounds = list(range(1, len(data['gpt_scores']) + 1))
            ax3.plot(rounds, data['gpt_scores'], 
                    marker='^', linewidth=linewidth, markersize=8,
                    color=color, alpha=alpha,
                    label=f'{opt_name} - GPT Judge', zorder=order)

        if use_polish:
            ax3.set_xlabel('Runda')
            ax3.set_ylabel('Wynik GPT Judge (%)')
            ax3.set_title('Wynik GPT Judge na rundę')
        else:
            ax3.set_xlabel('Round')
            ax3.set_ylabel('GPT Judge Score (%)')
            ax3.set_title('GPT Judge Score per Round')
        ax3.grid(True, alpha=0.3)
        # Dynamic y-axis: find min/max from all data and add margin
        all_gpt_scores = []
        for opt_name, data, *_ in other_optimizations:
            all_gpt_scores.extend(data['gpt_scores'])
        if baseline_data:
            all_gpt_scores.extend(baseline_data[1]['gpt_scores'])
        if all_gpt_scores:
            min_score = max(0, min(all_gpt_scores) - 10)
            max_score = min(100, max(all_gpt_scores) + 5)
            ax3.set_ylim([min_score, max_score])
        ax3.set_facecolor('#f8f9fa')

        # Wykres 4: Memory Usage
        for opt_name, data, color, order, linewidth, alpha in other_optimizations:
            rounds = list(range(1, len(data['memory_usage']) + 1))
            ax4.plot(rounds, data['memory_usage'], 
                    marker='d', linewidth=linewidth, markersize=6,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        if baseline_data:
            opt_name, data = baseline_data
            color, order, linewidth, alpha = ReferencedPlots.get_optimization_color(opt_name)
            rounds = list(range(1, len(data['memory_usage']) + 1))
            ax4.plot(rounds, data['memory_usage'], 
                    marker='d', linewidth=linewidth, markersize=8,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        if use_polish:
            ax4.set_xlabel('Runda')
            ax4.set_ylabel('Użycie RAM (GB)')
            ax4.set_title('Użycie pamięci na rundę')
        else:
            ax4.set_xlabel('Round')
            ax4.set_ylabel('RAM Usage (GB)')
            ax4.set_title('Memory Usage per Round')
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor('#f8f9fa')

        # Wykres 5: CPU Usage
        for opt_name, data, color, order, linewidth, alpha in other_optimizations:
            rounds = list(range(1, len(data['cpu_usage']) + 1))
            ax5.plot(rounds, data['cpu_usage'], 
                    marker='v', linewidth=linewidth, markersize=6,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        if baseline_data:
            opt_name, data = baseline_data
            color, order, linewidth, alpha = ReferencedPlots.get_optimization_color(opt_name)
            rounds = list(range(1, len(data['cpu_usage']) + 1))
            ax5.plot(rounds, data['cpu_usage'], 
                    marker='v', linewidth=linewidth, markersize=8,
                    color=color, alpha=alpha,
                    label=opt_name, zorder=order)

        if use_polish:
            ax5.set_xlabel('Runda')
            ax5.set_ylabel('Moc CPU (mW)')
            ax5.set_title('Moc CPU na rundę')
        else:
            ax5.set_xlabel('Round')
            ax5.set_ylabel('CPU Power (mW)')
            ax5.set_title('CPU Power per Round')
        ax5.grid(True, alpha=0.3)
        ax5.set_facecolor('#f8f9fa')

        # Dodaj wspólną legendę na samym dole wykresu
        handles, labels = ax5.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=min(4, len(labels)), fontsize=8, frameon=True, fancybox=True, shadow=True)

        # Dostosuj layout z miejscem na legendę na dole i tytuł na górze
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Dodaj timestamp
        fig.text(0.99, 0.01, f'Generated: {plotting_session_timestamp}', 
                ha='right', fontsize=8, style='italic')

        # Zapisz wykres
        plot_filename = f"{output_file_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path


    @staticmethod
    def plot_inference_parameters_comparison(parameter_results, model_name, agent_type, plotting_session_timestamp, output_dir, output_file_name):
        """
        Tworzy wykres porównujący różne parametry inferencji.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if not parameter_results:
            print("No parameter results available")
            return None

        # Przygotuj dane dla wykresów
        param_data = {}

        for param_key, result_data in parameter_results.items():
            params = result_data['parameters']
            session_data = result_data['session_data']

            # Wyciągnij metryki z rounds
            rounds_data = session_data.get('rounds', [])
            if not rounds_data:
                continue

            latencies, throughputs, gpt_scores, memory_usage = [], [], [], []
            swap_usage, cpu_power, gpu_power = [], [], []
            loading_percent, prompt_percent, generation_percent = [], [], []

            for round_data in rounds_data:
                latency_breakdown = round_data.get('latency_breakdown', {})
                total_ms = latency_breakdown.get('total_ms')
                tokens = latency_breakdown.get('tokens', {})
                throughput = tokens.get('throughput_tokens_per_sec')

                # Breakdown percentages
                breakdown = latency_breakdown.get('breakdown_percentage', {})
                loading_pct = breakdown.get('loading', 0) or 0
                prompt_pct = breakdown.get('prompt_eval', 0) or 0
                generation_pct = breakdown.get('generation', 0) or 0

                metrics = round_data.get('metrics', {})
                gpt_judge = metrics.get('gpt_judge', {})
                gpt_score = gpt_judge.get('score', 0) * 100

                # Resource usage - use differences if available, otherwise fallback to end values
                resource_differences = latency_breakdown.get('resource_differences', {})
                if resource_differences:
                    memory_diff = resource_differences.get('memory', {})
                    energy_diff = resource_differences.get('energy', {})

                    ram_used_gb = memory_diff.get('ram_delta_gb', 0)  # Change in RAM usage
                    swap_used_gb = memory_diff.get('swap_delta_gb', 0)  # Change in swap usage
                    cpu_power_mw = abs(energy_diff.get('cpu_power_delta_mw', 0) or 0)  # Absolute change in CPU power
                    gpu_power_mw = abs(energy_diff.get('gpu_power_delta_mw', 0) or 0)  # Absolute change in GPU power
                else:
                    # Fallback to end values if differences not available
                    end_resources = latency_breakdown.get('end_resources', {})
                    memory_info = end_resources.get('memory', {})
                    energy_info = end_resources.get('energy', {})

                    ram_used_gb = memory_info.get('ram_used_gb', 0)
                    swap_used_gb = memory_info.get('swap_used_gb', 0)
                    # Use absolute values for power consumption
                    cpu_power_mw = abs(energy_info.get('cpu_power_mw', 0) or 0)
                    gpu_power_mw = abs(energy_info.get('gpu_power_mw', 0) or 0)

                latencies.append(total_ms)
                throughputs.append(throughput)
                gpt_scores.append(gpt_score)
                memory_usage.append(ram_used_gb)
                swap_usage.append(swap_used_gb)
                cpu_power.append(cpu_power_mw)
                gpu_power.append(gpu_power_mw)
                loading_percent.append(loading_pct)
                prompt_percent.append(prompt_pct)
                generation_percent.append(generation_pct)

            # Funkcja pomocnicza do obliczania średniej bez None
            def safe_mean(values):
                if not values:
                    return None
                valid = [v for v in values if v is not None]
                return np.mean(valid) if valid else None

            param_data[param_key] = {
                'parameters': params,
                'avg_latency': safe_mean(latencies),
                'avg_throughput': safe_mean(throughputs),
                'avg_gpt_score': safe_mean(gpt_scores),
                'avg_memory': safe_mean(memory_usage),
                'avg_swap': safe_mean(swap_usage),
                'avg_cpu_power': safe_mean(cpu_power),
                'avg_gpu_power': safe_mean(gpu_power),
                'avg_loading_pct': safe_mean(loading_percent),
                'avg_prompt_pct': safe_mean(prompt_percent),
                'avg_generation_pct': safe_mean(generation_percent)
            }

        # Sortuj według średniej latencji (None na końcu)
        sorted_params = sorted(param_data.items(), key=lambda x: (x[1]['avg_latency'] is None, x[1]['avg_latency'] or float('inf')))

        # Twórz wykres 3x3 - STAŁY ROZMIAR z więcej metrykami
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(20, 18))

        agent_name = agent_type.value if hasattr(agent_type, 'value') else str(agent_type)
        title_suffix = f" - {model_name}" if model_name else ""
        fig.suptitle(f'Inference Parameters Comparison\n{title_suffix} | {agent_name}', 
                     fontsize=16, fontweight='bold', y=0.98)

        # Dane dla wykresów
        short_labels = [f"#{i+1}" for i in range(len(sorted_params))]
        avg_latencies = [data['avg_latency'] for _, data in sorted_params]
        avg_throughputs = [data['avg_throughput'] for _, data in sorted_params]
        avg_gpt_scores = [data['avg_gpt_score'] for _, data in sorted_params]
        avg_memory = [data['avg_memory'] for _, data in sorted_params]
        avg_swap = [data['avg_swap'] for _, data in sorted_params]
        avg_cpu_power = [data['avg_cpu_power'] for _, data in sorted_params]
        avg_gpu_power = [data['avg_gpu_power'] for _, data in sorted_params]
        avg_loading_pct = [data['avg_loading_pct'] for _, data in sorted_params]
        avg_prompt_pct = [data['avg_prompt_pct'] for _, data in sorted_params]
        avg_generation_pct = [data['avg_generation_pct'] for _, data in sorted_params]

        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_params)))

        # Przygotuj legendę z parametrami
        legend_labels = []
        for i, (param_key, data) in enumerate(sorted_params):
            params = data['parameters']
            ctx = params.get('context_size', 0)
            max_tok = params.get('max_tokens', 0)
            temp = params.get('temperature', 0)
            top_p = params.get('top_p', 0)
            legend_labels.append(f"#{i+1}: ctx={ctx}, max={max_tok}, temp={temp}, p={top_p}")

        # Utwórz tekst legendy
        legend_text = "Parameters:\n" + "\n".join(legend_labels)

        # Wykresy słupkowe - wszystkie 9 metryk
        ax1.bar(short_labels, avg_latencies, color=colors, alpha=0.8)
        ax1.set_title('Average Latency', fontweight='bold')
        ax1.set_ylabel('Latency (ms)')

        ax2.bar(short_labels, avg_throughputs, color=colors, alpha=0.8)
        ax2.set_title('Average Throughput', fontweight='bold')
        ax2.set_ylabel('Tokens per Second')

        ax3.bar(short_labels, avg_gpt_scores, color=colors, alpha=0.8)
        ax3.set_title('Average GPT Score', fontweight='bold')
        ax3.set_ylabel('Score (%)')
        # Dynamic y-axis for better visibility
        if avg_gpt_scores:
            min_score = max(0, min(avg_gpt_scores) - 10)
            max_score = min(100, max(avg_gpt_scores) + 5)
            ax3.set_ylim([min_score, max_score])

        # Memory usage changes
        ax4.bar(short_labels, avg_memory, color=colors, alpha=0.8)
        ax4.set_title('Average RAM Change', fontweight='bold')
        ax4.set_ylabel('RAM Δ (GB)')

        # Swap usage changes
        ax5.bar(short_labels, avg_swap, color=colors, alpha=0.8)
        ax5.set_title('Average Swap Change', fontweight='bold')
        ax5.set_ylabel('Swap Δ (GB)')

        # CPU Power changes
        ax6.bar(short_labels, avg_cpu_power, color=colors, alpha=0.8)
        ax6.set_title('Average CPU Power Change', fontweight='bold')
        ax6.set_ylabel('CPU Power Δ (mW)')

        # GPU Power changes
        ax7.bar(short_labels, avg_gpu_power, color=colors, alpha=0.8)
        ax7.set_title('Average GPU Power Change', fontweight='bold')
        ax7.set_ylabel('GPU Power Δ (mW)')

        # Loading Percentage
        ax8.bar(short_labels, avg_loading_pct, color=colors, alpha=0.8)
        ax8.set_title('Average Loading %', fontweight='bold')
        ax8.set_ylabel('Loading (%)')

        # Prompt + Generation percentages (stacked)
        ax9.bar(short_labels, avg_prompt_pct, color=colors, alpha=0.6, label='Prompt Eval')
        ax9.bar(short_labels, avg_generation_pct, bottom=avg_prompt_pct, color=colors, alpha=0.8, label='Token Generation')
        ax9.set_title('Average Breakdown %', fontweight='bold')
        ax9.set_ylabel('Percentage (%)')
        ax9.legend()

        # Konfiguracja wszystkich wykresów
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')

        # Dostosuj layout z miejscem na legendę na dole i tytuł na górze
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Add legend with parameters
        fig.text(0.02, 0.02, legend_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        # Add timestamp
        fig.text(0.99, 0.01, f'Generated: {plotting_session_timestamp}', 
                ha='right', fontsize=8, style='italic')

        # Save plot
        plot_filename = f"{output_file_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f" Inference parameters plot saved: {plot_path}")
        return plot_path


    def plot_single_model_summary(self, session_data, optimisation_type, model_name, agent_type, plotting_session_timestamp, metadata, output_dir, output_file_name):

        """
        Agreguje i wizualizuje wyniki z wielu rund dla jednego modelu - uproszczona wersja.

        Ta funkcja tworzy wykres podsumowujący wydajność modelu na podstawie wszystkich
        przeprowadzonych rund ewaluacji. Skupia się na kluczowych informacjach bez
        szczegółowych opisów metryk:
        - Średnie wyniki metryk z odchyleniem standardowym (wykres słupkowy)
        - Średni rozkład latencji z procentowym udziałem komponentów
        - Metadane modelu (architektura, parametry, quantization, rozmiar)
        - Statystyki sesji (liczba rund, zakres wyników)

        Args:
            session_data (dict): Dane sesji zawierające wyniki z wielu rund
            model_name (str, optional): Nazwa ewaluowanego modelu
            agent_type (Agent, optional): Typ agenta użytego w ewaluacji
            output_dir (str, optional): Katalog do zapisu wykresu
            session_timestamp (str, optional): Timestamp sesji dla nazwy pliku

        Returns:
            str: Ścieżka do zapisanego pliku wykresu lub None w przypadku błędu

        Layout:
            1x3 grid - Panel metryk | Latencja breakdown | Model metadata
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Zbierz wszystkie dane ze wszystkich rund
        all_metrics = {}
        all_latency = []
        round_count = 0

        for round_data in session_data.get('rounds', []):
            round_count += 1
            metrics = round_data.get('metrics', {})
            latency = round_data.get('latency_breakdown', {})

            # Zbierz metryki
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'score' in metric_data:
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_data['score'])

            # Zbierz latencję
            if latency:
                all_latency.append(latency)

        if not all_metrics:
            print("No metrics data found for model aggregation plot")
            return None

        # Oblicz średnie metryk
        avg_metrics = {}
        for metric_name, scores in all_metrics.items():
            avg_metrics[metric_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }

        # Oblicz średnią latencję (tylko dla wartości non-None)
        avg_latency = {}
        if all_latency:
            def safe_mean(values):
                """Oblicz średnią pomijając None"""
                valid_values = [v for v in values if v is not None]
                return np.mean(valid_values) if valid_values else None

            avg_latency = {
                'total_ms': safe_mean([l.get('total_ms') for l in all_latency]),
                'model_loading_ms': safe_mean([l.get('model_loading_ms') for l in all_latency]),
                'prompt_evaluation_ms': safe_mean([l.get('prompt_evaluation_ms') for l in all_latency]),
                'token_generation_ms': safe_mean([l.get('token_generation_ms') for l in all_latency]),
                'avg_throughput': safe_mean([l.get('tokens', {}).get('throughput_tokens_per_sec') for l in all_latency])
            }

        # Utwórz wykres 1x3 - STAŁY ROZMIAR
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

        # === PANEL 1: Wykres metryk (kolumna 1, zajmuje oba wiersze)
        # Ensure a logical order for the bar chart
        preferred_order = [
            'json_validity', 'tool_names_match', 'tool_args_structure', 'tool_args_values',
            'levenshtein_similarity', 'p_bert', 'r_bert', 'bleu', 'rougeL', 'meteor',
            'gpt_judge'
        ]

        # Filter and sort based on preferred order, followed by remaining metrics
        sorted_names = [m for m in preferred_order if m in avg_metrics]
        remaining_names = [m for m in avg_metrics if m not in preferred_order]
        names = sorted_names + remaining_names

        values = [avg_metrics[m]['mean'] for m in names]
        stds = [avg_metrics[m]['std'] for m in names]
        colors = ['green' if v >= 0.7 else 'orange' if v >=
                  0.4 else 'red' for v in values]

        bars = ax1.bar(range(len(names)), values, color=colors,
                       alpha=0.7, yerr=stds, capsize=5)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Average Score')

        # Tytuł z informacjami
        title_parts = []
        if model_name:
            title_parts.append(f"Model: {model_name}")
        if agent_type:
            agent_name =agent_type
            title_parts.append(f"Agent: {agent_name}")

        # Add optimization type to title if available
        optimization = session_data.get('optimisation', {})
        if optimization:
            opt_str = ", ".join([f"{k}={v}" if v is not None else str(k) for k, v in optimization.items()])
            title_parts.append(f"Opt: {opt_str}")

        # title_parts.append(f"Rounds: {round_count}")
        title = " | ".join(title_parts)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)

        # Dodaj wartości na słupkach
        for i, (bar, value, std) in enumerate(zip(bars, values, stds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # === PANEL 2: Latency breakdown ===
        ax2.axis('off')
        latency_text = "ŚREDNIA LATENCJA\n\n"

        if avg_latency:
            total_ms = avg_latency.get('total_ms')
            loading_ms = avg_latency.get('model_loading_ms')
            prompt_ms = avg_latency.get('prompt_evaluation_ms') or avg_latency.get('prompt_eval_ms')
            generation_ms = avg_latency.get('token_generation_ms')

            # Oblicz procenty (tylko jeśli wartości nie są None)
            loading_pct = (loading_ms / total_ms) * 100 if (total_ms and loading_ms is not None) else None
            prompt_pct = (prompt_ms / total_ms) * 100 if (total_ms and prompt_ms is not None) else None
            generation_pct = (generation_ms / total_ms) * 100 if (total_ms and generation_ms is not None) else None

            # Formatuj wartości - pokaż "N/A" dla None
            def fmt(val, unit=""):
                return f"{val:.1f}{unit}" if val is not None else "N/A"

            latency_text += f"• Total: {fmt(total_ms, 'ms')} (100.0%)\n"
            latency_text += f"• Loading: {fmt(loading_ms, 'ms')} ({fmt(loading_pct, '%')})\n"
            latency_text += f"• Prompt Eval: {fmt(prompt_ms, 'ms')} ({fmt(prompt_pct, '%')})\n"
            latency_text += f"• Generation: {fmt(generation_ms, 'ms')} ({fmt(generation_pct, '%')})\n"
            avg_throughput = avg_latency.get('avg_throughput')
            latency_text += f"• Throughput: {fmt(avg_throughput, ' tok/s')}\n\n"

        # Dodaj statystyki sesji
        latency_text += f"SESSION STATS\n\n"
        latency_text += f"• Total rounds: {round_count}\n"
        if avg_metrics.get('gpt_judge'):
            gpt_stats = avg_metrics['gpt_judge']
            latency_text += f"• GPT Judge avg: {gpt_stats['mean']:.3f}±{gpt_stats['std']:.3f}\n"
            latency_text += f"• GPT Judge range: {gpt_stats['min']:.3f} - {gpt_stats['max']:.3f}\n"

        ax2.text(0.05, 0.95, latency_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        latency_title = 'Average Latency Breakdown (ms)'
        if optimization:
            opt_str = ", ".join([f"{k}={v}" if v is not None else str(k) for k, v in optimization.items()])
            latency_title = f"{latency_title} - {opt_str}"
        ax2.set_title(latency_title, fontsize=12, weight='bold')

        # === PANEL 3: Model metadata ===
        ax3.axis('off')
        metadata_text = "MODEL METADATA\n\n"


        metadata_text += f"• Name: {metadata.get('model_name', 'Unknown')}\n"
        metadata_text += f"• Architecture: {metadata.get('architecture', 'Unknown')}\n"
        metadata_text += f"• Parameters: {metadata.get('parameter_size_display', 'Unknown')}\n"
        metadata_text += f"• Quantization: {metadata.get('quantization_level', 'Unknown')}\n"
        metadata_text += f"• Format: {metadata.get('model_format', 'Unknown')}\n"
        metadata_text += f"• Size: {metadata.get('model_size_gb', 0)} GB\n"
        metadata_text += f"• Created: {metadata.get('created_at', 'Unknown')}\n"
        metadata_text += f"• Digest: {metadata.get('digest', 'Unknown')[:16]}...\n"


        metadata_title = 'Model Specifications'
        if optimization:
            opt_str = ", ".join([f"{k}={v}" if v is not None else str(k) for k, v in optimization.items()])
            metadata_title = f"{metadata_title} - {opt_str}"
        ax3.set_title(metadata_title, fontsize=12, pad=20)
        ax3.text(0.05, 0.95, metadata_text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax3.set_title('Model Information', fontsize=12, weight='bold')

        plt.tight_layout()

        # Save plot
        plot_filename = f"{output_file_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f" Model aggregation plot saved: {plot_path}")
        return plot_path


    def calculate_averages(self, session_data, metadata, model_name_prefix=None):
        """
        Oblicza średnie dla wszystkich modeli z danych per runda.

        Args:
            session_data: Lista danych sesji z modelami
            metadata: Metadane modeli
            model_name: Nazwa modelu (dla skróconych nazw)

        Returns:
            tuple: (models_data, model_names, avg_scores, avg_latencies, model_sizes, model_params)
        """
        # Initialize models data structure
        models_data = {}

        # Handle list of session data for multiple models
        if isinstance(session_data, list):
            # Process each model session
            for model_session in session_data:
                if not isinstance(model_session, dict):
                    continue

                model_name_item = model_session.get('model_name', 'unknown_model')
                 # Skip failed models (Granite MoE)
                model_name_lower = model_name_item.lower()
                if 'granite3-moe' in model_name_lower or 'moe' in model_name_lower:
                    continue

                # Initialize model data if not exists
                if model_name_item not in models_data:
                    models_data[model_name_item] = {
                        'gpt_scores': [],
                        'latency': [],
                        'metadata': metadata.get(model_name_item, {}) if isinstance(metadata, dict) else {},
                        'rounds_count': 0,
                        'avg_latency': 0,
                        'avg_gpt_score': 0,
                        'throughput': [],
                        'cpu_power': [],
                        'gpu_power': []
                    }

                # Process each round's data
                rounds = model_session.get('rounds', [])
                if not isinstance(rounds, list):
                    continue

                for round_idx, round_data in enumerate(rounds):
                    if not isinstance(round_data, dict):
                        continue

                    # Extract metrics and latency data
                    metrics = round_data.get('metrics', {})
                    latency = round_data.get('latency_breakdown', {})

                    # Track that we've processed a round
                    models_data[model_name_item]['rounds_count'] += 1

                    # Process GPT Judge score (support multiple field variants)
                    gpt_score = 0
                    if isinstance(metrics, dict):
                        if 'gpt_judge' in metrics:
                            gpt = metrics['gpt_judge']
                            if isinstance(gpt, dict):
                                # Prefer original_score (0-10) for visualization
                                gpt_score = gpt.get('original_score', gpt.get('score', 0))
                                # If we only have 'score' and it's 0-1 range, scale it
                                if 'original_score' not in gpt and gpt_score <= 1.0 and gpt_score > 0:
                                    gpt_score *= 10.0
                            else:
                                gpt_score = gpt
                        elif 'gpt_judge_score' in metrics:
                            gpt_score = metrics.get('gpt_judge_score', 0)
                            if gpt_score <= 1.0 and gpt_score > 0:
                                gpt_score *= 10.0
                        elif 'gpt_score' in metrics:
                            gpt_score = metrics.get('gpt_score', 0)
                            if gpt_score <= 1.0 and gpt_score > 0:
                                gpt_score *= 10.0
                    if isinstance(gpt_score, (int, float)) and gpt_score is not None:
                        models_data[model_name_item]['gpt_scores'].append(gpt_score)
                        if models_data[model_name_item]['gpt_scores']:
                            models_data[model_name_item]['avg_gpt_score'] = np.mean(models_data[model_name_item]['gpt_scores'])

                    # Process latency data if available
                    if isinstance(latency, dict):
                        # Handle different latency formats
                        total_latency = latency.get('total_ms')
                        if not isinstance(total_latency, (int, float)):
                            total_latency = latency.get('total_time_ms')
                        if not isinstance(total_latency, (int, float)):
                            # Fallback: sum known components if present
                            pe = latency.get('prompt_evaluation_ms', latency.get('prompt_eval_ms', 0))
                            tg = latency.get('token_generation_ms', 0)
                            ml = latency.get('model_loading_ms', 0)
                            try:
                                total_latency = (pe or 0) + (tg or 0) + (ml or 0)
                            except Exception:
                                total_latency = 0

                        if isinstance(total_latency, (int, float)) and total_latency > 0:
                            models_data[model_name_item]['latency'].append(total_latency)
                            models_data[model_name_item]['avg_latency'] = np.mean(models_data[model_name_item]['latency'])

                        # Extract throughput if available
                        tokens_info = latency.get('tokens', {})
                        throughput = tokens_info.get('throughput_tokens_per_sec')
                        if throughput is None:
                            throughput = tokens_info.get('predicted_per_second')
                        if throughput is None:
                            throughput = tokens_info.get('throughput', 0)

                        if throughput is not None and throughput > 0:
                            models_data[model_name_item]['throughput'].append(throughput)

                    # Extract energy data from end_resources (active power draw during inference)
                    end_resources = latency.get('end_resources', {})
                    energy_info = end_resources.get('energy', {})

                    cpu_power_mw = energy_info.get('cpu_power_mw', 0)
                    gpu_power_mw = energy_info.get('gpu_power_mw', 0)

                    # Debug energy data for radar
                    print(f"DEBUG CALCULATE_AVERAGES ENERGY (active): Model={model_name_item}, Round={round_idx}, CPU={cpu_power_mw}mW, GPU={gpu_power_mw}mW")

                    models_data[model_name_item]['cpu_power'].append(cpu_power_mw)
                    models_data[model_name_item]['gpu_power'].append(gpu_power_mw)
        else:
            print("session_data must be a list for all models comparison")
            return None

        if not models_data:
            print("No valid model data found")
            return None

        # Calculate averages and prepare data for visualization
        model_names = []
        avg_scores = []
        avg_latencies = []
        model_sizes = []
        model_params = []

        for model_name, data in models_data.items():
            # Compute averages if available; include model even if one metric missing

            # Calculate average scores and latencies
            avg_gpt = np.mean(data['gpt_scores']) if data['gpt_scores'] else 0
            avg_lat = np.mean(data['latency']) if data['latency'] else 0

            # Calculate average energy consumption
            avg_cpu_power = np.mean(data['cpu_power']) if data['cpu_power'] else 0
            avg_gpu_power = np.mean(data['gpu_power']) if data['gpu_power'] else 0

            # Only add if we have valid data (filter out models with score <= 1.0)
            if avg_gpt > 1.0:
                model_names.append(model_name)
                avg_scores.append(avg_gpt)
                avg_latencies.append(avg_lat)

                # Get model metadata - handle both full and shortened names
                if model_name_prefix is not None:
                    # For shortened names (e.g., "q2_K"), reconstruct full name with prefix
                    full_model_name = f"{model_name_prefix}-{model_name}"
                    full_model_name = full_model_name.replace("_", ":",1)
                else:
                    # For full model names, convert underscores to colons
                    full_model_name = model_name.replace("_", ":",1)

                # Robust metadata lookup
                model_meta = metadata.get('model', {}) if isinstance(metadata, dict) else {}
                model_metadata = model_meta.get(full_model_name)
                if not model_metadata:
                    # Try with underscores if colons failed
                    model_metadata = model_meta.get(full_model_name.replace(":", "_"))
                if not model_metadata:
                    # Try with colons if underscores failed (reverse)
                    model_metadata = model_meta.get(full_model_name.replace("_", ":"))
                if not model_metadata:
                    # Try original model_name as well
                    model_metadata = model_meta.get(model_name)

                if not model_metadata:
                    model_metadata = {}

                size_gb = model_metadata.get('model_size_gb', 0)

                if size_gb == 0:
                     # Try to find size in the original session data for this model
                     for sess in session_data:
                         if sess.get('model_name') == model_name:
                             size_gb = sess.get('metadata', {}).get('model_size_gb', 0)
                             if size_gb > 0: break

                print(f" DEBUG MODEL SIZE: model_name={model_name}, full_model_name={full_model_name}")
                print(f" DEBUG MODEL SIZE: model_metadata keys={model_metadata.keys() if model_metadata else 'EMPTY'}")
                print(f" DEBUG MODEL SIZE: size_gb={size_gb}")

                model_sizes.append(size_gb)

                # Store metadata and energy data in models_data for radar chart
                models_data[model_name]['metadata'] = {
                    'model_size_gb': size_gb,
                    'parameters': model_metadata.get('parameter_size_display', 0),
                    'architecture': model_metadata.get('architecture', 'Unknown'),
                    'quantization_level': model_metadata.get('quantization_level', 'Unknown')
                }

                # Add average energy data for radar chart and big table
                models_data[model_name]['avg_cpu_power'] = avg_cpu_power
                models_data[model_name]['avg_gpu_power'] = avg_gpu_power
                models_data[model_name]['total_power'] = avg_cpu_power + avg_gpu_power
                models_data[model_name]['avg_throughput'] = np.mean(data['throughput']) if data['throughput'] else 0
                models_data[model_name]['avg_memory'] = np.mean(data['memory_usage']) if data.get('memory_usage') else 0

                # Prepare parameters for display
                parameters_copy = self.MULTI_TURN_GLOBAL_CONFIG.copy() if isinstance(self.MULTI_TURN_GLOBAL_CONFIG, dict) else {}
                for key in ['cot_prompt_path', 'cot_schema_path', 'display_prompt', 'validation_cot_prompt_path', 'validation_schema_path']:
                    parameters_copy.pop(key, None)
                model_params.append(parameters_copy)

        # Generate comprehensive multi-model metrics table (LaTeX)
        self.generate_multi_model_metrics_table(models_data, model_names, session_data[0].get('session_timestamp') if session_data else "latest")

        if not model_names:
            print("No valid model data found for visualization")
            return None, None, None, None, None, None

        # Generate the master metrics table requested by the user
        self.generate_master_metrics_table(models_data, model_names)

        return models_data, model_names, avg_scores, avg_latencies, model_sizes, model_params


    def generate_master_metrics_table(self, models_data, model_names):
        """Generates a comprehensive LaTeX table with all key metrics for all models."""
        import os
        try:
            output_dir = os.path.dirname(self.report_path) if hasattr(self, 'report_path') else "output"
            os.makedirs(output_dir, exist_ok=True)
            table_path = os.path.join(output_dir, "master_metrics_table.tex")

            with open(table_path, 'w', encoding='utf-8') as f:
                f.write("\\begin{table}[!ht]\n\\centering\n")
                f.write("\\caption{Zbiorcze zestawienie metryk sędziowskich, wydajnościowych i energetycznych}\n")
                f.write("\\label{tab:master_metrics}\n")
                f.write("\\resizebox{\\linewidth}{!}{\n")
                f.write("\\begin{tabular}{|l|c|c|c|c|r|}\n\\hline\n")
                f.write("\\textbf{Model} & \\textbf{GPT Score} & \\textbf{Latencja (ms)} & \\textbf{TPS} & \\textbf{RAM (GB)} & \\textbf{Energia (Total mW)} \\\\\n\\hline\n")

                for name in model_names:
                    d = models_data[name]
                    safe_name = name.replace("_", "\\_").replace(":", ":")
                    gpt = d.get('avg_gpt_score', 0)
                    lat = d.get('avg_latency', 0)
                    tps = d.get('avg_throughput', [])
                    avg_tps = np.mean(tps) if tps else 0
                    mem = d.get('memory_usage', [])
                    avg_mem = np.mean(mem) if mem else 0
                    power = (np.mean(d.get('cpu_power', [0])) + np.mean(d.get('gpu_power', [0])))

                    f.write(f"{safe_name} & {gpt:.2f} & {lat:.0f} & {avg_tps:.1f} & {avg_mem:.2f} & {power:.0f}mW \\\\\n\\hline\n")

                f.write("\\end{tabular}\n}\n\\end{table}\n")
            print(f" Master metrics table saved: {table_path}")
        except Exception as e:
            print(f"Failed to generate master metrics table: {e}")


    def generate_multi_model_metrics_table(self, models_data, model_names, timestamp):
        """Generates a comprehensive LaTeX table with all key metrics for all models."""
        import os

        output_dir = os.path.dirname(self.report_path) if hasattr(self, 'report_path') else "output"
        os.makedirs(output_dir, exist_ok=True)

        table_path = os.path.join(output_dir, f"multi_model_metrics_{timestamp}.tex")

        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[!ht]\n\\centering\n")
            f.write("\\caption{Zbiorcze zestawienie metryk wydajnościowych i jakościowych dla testowanych modeli}\n")
            f.write("\\label{tab:multi_model_metrics}\n")
            f.write("\\resizebox{\\linewidth}{!}{\n")
            f.write("\\begin{tabular}{|l|c|c|c|c|c|r|}\n\\hline\n")
            f.write("\\textbf{Model} & \\textbf{GPT Score} & \\textbf{Latencja (ms)} & \\textbf{TPS} & \\textbf{Pamięć (GB)} & \\textbf{Energia (mW)} \\\\\n\\hline\n")

            for name in model_names:
                data = models_data[name]
                gpt = data.get('avg_gpt_score', 0)
                lat = data.get('avg_latency', 0)
                tps = data.get('avg_throughput', 0)
                mem = data.get('avg_memory', 0)
                power = data.get('total_power', 0)
                cpu = data.get('avg_cpu_power', 0)
                gpu = data.get('avg_gpu_power', 0)

                # Format latext row
                safe_name = name.replace("_", "\\_").replace(":", ":")
                f.write(f"{safe_name} & {gpt:.2f} & {lat:.0f} & {tps:.1f} & {mem:.2f} & CPU: {cpu:.0f} \\\\ GPU: {gpu:.0f} \\\\ Total: {power:.0f} \\\\\n\\hline\n")

            f.write("\\end{tabular}\n}\n\\end{table}\n")

        print(f" Comprehensive LaTeX table saved: {table_path}")


    def plot_aggr_all_models_with_reference(self, session_data, optimisation_type, agent_type, plotting_session_timestamp, metadata, output_dir, output_file_name, use_polish=True, model_name_prefix=None):
        """
        Compare multiple models using radar/quadrant visualization for model selection.

        This function creates a comprehensive model comparison focusing on key production
        selection criteria:
        - Average GPT Judge score across all rounds
        - Model size (GB) and parameter count
        - Average latency (total ms) and throughput
        - Quantization level and architecture
        - Gartner Magic Quadrant or radar chart visualization

        Args:
            session_data (dict): Dictionary containing evaluation data for all models
            output_file_name (str): Base name for the output file
            output_dir (str): Directory to save the comparison plot

        Returns:
            str: Path to the saved plot file or None in case of error

        Layout:
            2x2 grid - Radar Chart | Performance vs Size Scatter
                     Latency Comparison | Model Specifications Table
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Calculate averages using centralized function
        print(f"DEBUG: plot_aggr_all_models_with_reference called")
        models_data, model_names, avg_scores, avg_latencies, model_sizes, model_params = self.calculate_averages(session_data, metadata, model_name_prefix)

        if models_data is None:
            print("DEBUG: models_data is None, returning early")
            return None

        # Create separate visualizations instead of one combined plot
        saved_plots = {}

        # Set language preference for this function and its children
        self._current_use_polish = use_polish
        os.makedirs(output_dir, exist_ok=True)

        # 1. Bar chart: Latency Comparison
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        self._create_latency_bars(ax1, model_names, avg_latencies, model_sizes, model_name=model_name_prefix)
        # Add model_name_prefix to title if provided
        title_suffix = f" - {model_name_prefix}" if model_name_prefix else ""

        if use_polish:
            fig1.suptitle(f'Porównanie Latencji Modeli{title_suffix}', fontsize=16, fontweight='bold')
        else:
            fig1.suptitle(f'Model Latency Comparison{title_suffix}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plot1_path = os.path.join(output_dir, f"{output_file_name}_latency_bars.png")
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots['latency_bars'] = plot1_path
        print(f" Latency bars saved: {plot1_path}")

        # 2. Scatter plot: Performance vs Model Size
        if any(size > 0 for size in model_sizes):
            fig2 = plt.figure(figsize=(10, 8))
            ax2 = fig2.add_subplot(111)
            self._create_scatter_plot(ax2, model_names, avg_scores, model_sizes, avg_latencies)
            if use_polish:
                fig2.suptitle(f'Wydajność vs Latencja{title_suffix}', fontsize=16, fontweight='bold')
            else:
                fig2.suptitle(f'Performance vs Latency{title_suffix}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plot2_path = os.path.join(output_dir, f"{output_file_name}_scatter.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots['scatter'] = plot2_path
            print(f"📈 Scatter plot saved: {plot2_path}")

        # 3. Radar chart for multiple metrics (also works for single model)
        if len(model_names) >= 1:
            print(f" DEBUG: Generating radar chart for {len(model_names)} model(s): {model_names}")
            fig3 = plt.figure(figsize=(10, 10))
            ax3 = fig3.add_subplot(111, polar=True)
            # Radar chart needs raw session data, not processed data
            radar_data = session_data
            self._create_radar_chart(ax3, radar_data, metadata, model_name_prefix)
            if use_polish:
                fig3.suptitle(f'Radar Wydajności Mobilnej{title_suffix}', fontsize=16, fontweight='bold')
            else:
                fig3.suptitle(f'Mobile Performance Radar{title_suffix}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plot3_path = os.path.join(output_dir, f"{output_file_name}_radar.png")
            plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots['radar'] = plot3_path
            print(f" Radar chart saved: {plot3_path}")

        # 4. Create LaTeX table
        latex_table_path = self._create_latex_table(model_names, models_data, output_dir, f"{output_file_name}_table.tex")
        if latex_table_path:
            saved_plots['latex_table'] = latex_table_path
            print(f"  LaTeX table saved: {latex_table_path}")

        # 5. Generate category winners analysis
        category_winners_plot = self._create_category_winners_analysis_referenced(
            models_data=models_data,
            output_file_name=f"{optimisation_type}_mobile_category_winners_{plotting_session_timestamp}",
            output_dir=output_dir,
            timestamp=plotting_session_timestamp,
            optimisation_type=optimisation_type,
            model_name_prefix=model_name_prefix
        )
        if category_winners_plot:
            saved_plots['category_winners'] = category_winners_plot
            print(f"Category winners saved: {category_winners_plot}")

        # 6. Score Comparison
        score_plot = self._create_score_comparison(model_names, models_data, output_dir, output_file_name, use_polish)
        if score_plot:
            saved_plots['score_comparison'] = score_plot

        # 7. Resource Comparison
        resource_plot = self._create_resource_comparison(model_names, models_data, output_dir, output_file_name, use_polish)
        if resource_plot:
            saved_plots['resource_comparison'] = resource_plot

        return saved_plots


    def _create_latex_table(self, model_names, models_data, output_dir, filename):
        """Create a LaTeX table with model specifications."""
        try:
            latex_content = []
            latex_content.append("\\begin{table}[h!]")
            latex_content.append("\\centering")
            latex_content.append("\\caption{Szczegóły Modeli}")
            latex_content.append("\\label{tab:model_details}")
            latex_content.append("\\resizebox{\\linewidth}{!}{")
            latex_content.append("\\begin{tabular}{|l|l|l|l|l|p{4cm}|}")
            latex_content.append("\\hline")
            latex_content.append("Model & Architektura & Parametry & Rozmiar & Kwantyzacja & Energia (mW) \\\\")
            latex_content.append("\\hline")

            for model in model_names:
                model_data = models_data.get(model, {})

                # Extract model info
                model_name = model.replace("_", ":",1)
                architecture = "granite"
                parameters = model_data.get('parameters', 'N/A')
                size_gb = model_data.get('model_size_gb', 0)
                quantization = model_data.get('quantization', 'N/A')

                # Energy data
                avg_cpu = model_data.get('avg_cpu_power', 0)
                avg_gpu = model_data.get('avg_gpu_power', 0) or 0
                avg_total = avg_cpu + avg_gpu

                if avg_cpu > 0 or avg_gpu > 0:
                    gpu_str = f"{avg_gpu:.0f}" if avg_gpu > 0 else "0"
                    energy_str = f"CPU: {avg_cpu:.0f}mW\\newline GPU: {gpu_str}mW\\newline Total: {avg_total:.0f}mW"
                else:
                    energy_str = "Brak danych"

                # Format size
                size_str = f"{size_gb:.1f} GB" if size_gb > 0 else "N/A"

                # Add row
                latex_content.append(f"{model_name} & {architecture} & {parameters} & {size_str} & {quantization} & {energy_str} \\\\")
                latex_content.append("\\hline")

            latex_content.append("\\end{tabular}")
            latex_content.append("}")
            latex_content.append("\\end{table}")

            # Save to file
            table_path = os.path.join(output_dir, filename)
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(latex_content))

            return table_path

        except Exception as e:
            print(f"Error creating LaTeX table: {str(e)}")
            return None


    def _create_radar_chart(self, ax, session_data, metadata, model_name_prefix=None):
        """Create a radar chart comparing multiple metrics across models."""
        # Calculate averages using centralized function
        models_data, model_names, avg_scores, avg_latencies, model_sizes, model_params = self.calculate_averages(session_data, metadata, model_name_prefix)

        if models_data is None:
            print("No data for radar chart")
            return

        # Debug info
        print(f"DEBUG RADAR: model_names={model_names}")
        print(f"DEBUG RADAR: avg_scores={avg_scores}")
        print(f"DEBUG RADAR: avg_latencies={avg_latencies}")
        print(f"DEBUG RADAR: model_sizes={model_sizes}")



        # Create gradient colors based on model size
        model_colors = self._get_model_size_gradient_colors(model_names, model_sizes)

        # Create color mapping for models
        model_color_map = {}
        for i, model_name in enumerate(model_names):
            model_color_map[model_name] = model_colors[i]

        # Prepare data for radar chart with absolute scales
        categories = ['GPT Score (%)', 'Latency (Mobile)', 'Model Size (Mobile)', 'Energy Efficiency (Mobile)']
        num_vars = len(categories)

        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Close the loop

        # Define mobile-friendly limits based on actual data
        MIN_MOBILE_LATENCY_MS = min(avg_latencies)  # Best case (minimum latency)
        MAX_MOBILE_LATENCY_MS = max(avg_latencies)  # Worst case
        MIN_MOBILE_SIZE_GB = min(model_sizes)  # Smallest model = 100%
        MAX_MOBILE_SIZE_GB = max(model_sizes)  # Largest model = 0%

        # Calculate energy efficiency (lower power = better)
        # Get total_power from models_data (already calculated in calculate_averages)
        # Define mobile-friendly limits based on actual data with safety margins
        # Using max * 1.5 ensures nothing is exactly in the center (0) unless it's genuinely terrible.
        MAX_LATENCY_LIMIT = max(avg_latencies) * 1.5 if avg_latencies else 1000
        MAX_SIZE_LIMIT = max(model_sizes) * 1.5 if model_sizes else 10

        # Calculate energy limits
        total_powers = []
        for m in model_names:
            p_list = models_data[m].get('cpu_power', [0])
            g_list = models_data[m].get('gpu_power', [0])
            total_powers.append(np.mean(p_list) + np.mean(g_list))
        MAX_POWER_LIMIT = max(total_powers) * 1.5 if total_powers else 5000

        # Consistent labels min/max for textual display
        MIN_MOBILE_LATENCY_MS = min(avg_latencies) if avg_latencies else 0
        MAX_MOBILE_LATENCY_MS = max(avg_latencies) if avg_latencies else 1000
        MIN_MOBILE_SIZE_GB = min(model_sizes) if model_sizes else 0
        MAX_MOBILE_SIZE_GB = max(model_sizes) if model_sizes else 10
        MIN_TOTAL_POWER = min(total_powers) if total_powers else 0
        MAX_TOTAL_POWER = max(total_powers) if total_powers else 1000

        # Calculate GPT Score range for proper scaling
        MIN_GPT_SCORE = min(avg_scores)
        MAX_GPT_SCORE = max(avg_scores)

        print(f"DEBUG SCALES: Latency range {MIN_MOBILE_LATENCY_MS:.1f}ms - {MAX_MOBILE_LATENCY_MS:.1f}ms")
        print(f"DEBUG SCALES: Size range {MIN_MOBILE_SIZE_GB:.1f}GB - {MAX_MOBILE_SIZE_GB:.1f}GB")
        print(f"DEBUG SCALES: Power range {MIN_TOTAL_POWER:.1f}mW - {MAX_TOTAL_POWER:.1f}mW")

        # Calculate GPT Score range for proper scaling
        MIN_GPT_SCORE = min(avg_scores)
        MAX_GPT_SCORE = max(avg_scores)

        print(f"DEBUG SCALES: GPT Score range {MIN_GPT_SCORE:.3f} - {MAX_GPT_SCORE:.3f}")

        # Plot each model with consistent colors
        for i, model in enumerate(model_names):
            # Scale each metric to 0-1
            # GPT Score: Use absolute 0-10 scale
            # To provide better visual separation for high scores (e.g. 9.1 vs 9.4),
            # we can use a slightly shifted scale if all scores are very high,
            # but for now let's stick to absolute 0-10 as requested.
            gpt_score_scaled = avg_scores[i] / 10.0
            gpt_score_scaled = max(0, min(1, gpt_score_scaled))

            # To make differences more visible if clustering at the edge:
            # gpt_score_scaled = (avg_scores[i] - 5) / 5  # Map 5-10 to 0-1
            # But user asked for absolute 0-10.

            # Latency: scale relative to 0 (closer to 0ms = better = closer to edge)
            latency_ms = avg_latencies[i]
            latency_scaled = 1 - (latency_ms / MAX_LATENCY_LIMIT)
            latency_scaled = max(0, min(1, latency_scaled))

            # Model size: scale relative to 0 (smaller = better = closer to edge)
            size_gb = model_sizes[i]
            size_scaled = 1 - (size_gb / MAX_SIZE_LIMIT)
            size_scaled = max(0, min(1, size_scaled))

            # Energy efficiency: scale relative to 0 (lower power = better = closer to edge)
            model_power = 0
            if models_data and model in models_data:
                # Use the calculated total power from the list
                model_power = total_powers[i]

            power_scaled = 1 - (model_power / MAX_POWER_LIMIT)
            power_scaled = max(0, min(1, power_scaled))

            values = [gpt_score_scaled, latency_scaled, size_scaled, power_scaled]
            values += values[:1]  # Close the loop

            # Debug radar values - uncommented to diagnose the "all maximum" issue
            print(f" DEBUG RADAR VALUES: Model={model}, GPT_raw={avg_scores[i]:.3f}, GPT_scaled={gpt_score_scaled:.3f}, Latency={latency_ms:.0f}ms→{latency_scaled:.2f}, Size={model_sizes[i]:.1f}GB→{size_scaled:.2f}, Power={model_power:.0f}mW→{power_scaled:.2f}")

            # Use consistent color for this model
            color = model_color_map[model]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=color)
            ax.fill(angles, values, color=color, alpha=0.1)

        # Set labels and title
        ax.set_xticks(angles[:-1])
        # Create detailed labels with units and scales (check if use_polish is available)
        try:
            # Try to get use_polish from function parameters
            use_polish = getattr(self, '_current_use_polish', True)  # Default to Polish
        except:
            use_polish = True  # Default to Polish

        if use_polish:
            detailed_categories = [
                f'Wynik GPT\n(0-10)',
                f'Latencja\n({MIN_MOBILE_LATENCY_MS:.0f}-{MAX_MOBILE_LATENCY_MS:.0f}ms)',
                f'Rozmiar modelu\n({MIN_MOBILE_SIZE_GB:.1f}-{MAX_MOBILE_SIZE_GB:.1f}GB)',
                f'Efektywność energetyczna\n({MIN_TOTAL_POWER:.0f}-{MAX_TOTAL_POWER:.0f}mW)'
            ]
            title_text = 'Radar Wydajności Mobilnej\n(Zewnętrzna krawędź = Lepsze)'
        else:
            detailed_categories = [
                f'GPT Score\n(0-10)',
                f'Latency\n({MIN_MOBILE_LATENCY_MS:.0f}-{MAX_MOBILE_LATENCY_MS:.0f}ms)',
                f'Model Size\n({MIN_MOBILE_SIZE_GB:.1f}-{MAX_MOBILE_SIZE_GB:.1f}GB)',
                f'Energy Efficiency\n({MIN_TOTAL_POWER:.0f}-{MAX_TOTAL_POWER:.0f}mW)'
            ]
            title_text = 'Mobile Performance Radar\n(Outer edge = Better)'

        ax.set_xticklabels(detailed_categories)
        ax.set_title(title_text, pad=30, fontsize=12, fontweight='bold')
        # Add legend further outside to avoid overlapping
        ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=8, frameon=True)

        # Set y-axis limits but remove percentage labels
        ax.set_ylim(0, 1)
        ax.set_yticks([])  # Remove all y-axis tick labels

        # Make axis labels visible on top of plot lines and adjust positioning
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_zorder(1000)
            label.set_fontweight('bold')
            label.set_fontsize(9)
            # Add small padding/offset to labels to avoid overlapping with axes
            # For polar plots, we can adjust alignment based on angle
            angle = angles[i]
            if angle == 0:
                label.set_ha('left')
            elif angle == np.pi/2:
                label.set_va('bottom')
            elif angle == np.pi:
                label.set_ha('right')
            elif angle == 3*np.pi/2:
                label.set_va('top')

        # Add grid for better readability
        ax.grid(True, alpha=0.3)


    def _create_scatter_plot(self, ax, model_names, avg_scores, model_sizes, avg_latencies):
        """Create a scatter plot of performance vs latency with model size as bubble size."""
        # GUARD: Check for empty data
        if not model_names or not avg_scores or not model_sizes or not avg_latencies:
            print("  Skipping scatter plot: Missing data")
            return

        # GUARD: Check for empty data
        if not model_names or not avg_scores or not model_sizes or not avg_latencies:
            print("  Skipping scatter plot: Missing data")
            return

        # GUARD to prevent plotting if arrays are effectively empty
        if len(model_names) == 0:
            return
        # GUARD: Check for empty data
        if not model_names or not avg_scores or not model_sizes or not avg_latencies:
            print("  Skipping scatter plot: Missing data")
            return

        # GUARD: Check if we have valid non-zero data for plotting
        if len(model_names) == 0:
             return
        # Create gradient colors based on model size
        scatter_colors = self._get_model_size_gradient_colors(model_names, model_sizes)

        # Keep latencies in milliseconds for the axis
        latencies_ms = avg_latencies

        # Scale model sizes for better visualization
        min_size = 100  # Minimum bubble size
        max_size = 1000  # Maximum bubble size
        if len(set(model_sizes)) > 1:  # If we have different model sizes
            # Scale sizes to range [min_size, max_size]
            size_min = min(model_sizes)
            size_range = max(model_sizes) - size_min
            bubble_sizes = [min_size + (size - size_min) / size_range * (max_size - min_size)
                          for size in model_sizes]
        else:
            # If all models have the same size, use medium size
            bubble_sizes = [(min_size + max_size) / 2] * len(model_sizes)

        # Use Line2D for a clean legend with fixed-size markers
        from matplotlib.lines import Line2D
        legend_elements = []
        for i, model in enumerate(model_names):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         label=model,
                                         markerfacecolor=scatter_colors[i],
                                         markersize=10, alpha=0.7))

        # Create scatter plot with unique colors for each model
        for i, (lat, score, size, color) in enumerate(zip(latencies_ms, avg_scores, bubble_sizes, scatter_colors)):
            ax.scatter(
                lat, score, s=size, c=[color], alpha=0.7,
                edgecolors='black', linewidth=0.5
            )

        # Check language preference
        try:
            use_polish = getattr(self, '_current_use_polish', True)
        except:
            use_polish = True

        if use_polish:
            ax.set_xlabel('Latencja (ms) — mniej znaczy lepiej')
            ax.set_ylabel('Jakość (Wynik GPT 0-10) → Góra lepiej')
            ax.set_title('Jakość a latencja')
        else:
            ax.set_xlabel('Latency (ms; lower is better)')
            ax.set_ylabel('Quality (GPT Score 0-10) → Up is better')
            ax.set_title('Quality vs latency')

        ax.grid(True, linestyle='--', alpha=0.6)

        # Set axis limits with padding for 0-10 scale
        if latencies_ms and avg_scores:
            x_padding = (max(latencies_ms) - min(latencies_ms)) * 0.2 if len(latencies_ms) > 1 else 100
            y_padding = (max(avg_scores) - min(avg_scores)) * 0.2 if len(avg_scores) > 1 else 1

            ax.set_xlim(max(0, min(latencies_ms) - x_padding),
                       max(latencies_ms) + x_padding)
            ax.set_ylim(max(0, min(avg_scores) - y_padding),
                       min(10.5, max(avg_scores) + y_padding))

        # Add legend for model colors with standardized sizes - Move outside
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)


    def _shorten_model_labels(self, model_names):
        """Shorten model labels if they differ only after the last '-' character."""
        if not model_names:
            return model_names, None

        # Check if all models have the same prefix before the last '-'
        prefixes = []
        suffixes = []

        for name in model_names:
            if '-' in name:
                parts = name.rsplit('-', 1)  # Split on last '-'
                prefixes.append(parts[0])
                suffixes.append(parts[1])
            else:
                # If no '-', use the whole name as suffix
                prefixes.append('')
                suffixes.append(name)

        # Check if all prefixes are the same (and not empty)
        if len(set(prefixes)) == 1 and prefixes[0]:
            # All models have the same prefix, return shortened labels and common prefix
            return suffixes, prefixes[0]
        # Models have different prefixes, return original names
        return model_names, None


    def _model_name_to_id(self, name):
        """Konwertuje nazwę modelu na bezpieczny identyfikator pliku."""
        if not name: return "unknown"
        return str(name).replace(":", "_").replace("/", "_").replace(".", "_")


    def _shorten_model_name(self, name):
        """Skraca nazwę modelu do wyświetlania na osiach wykresu."""
        if not name: return "unknown"
        # Bierzemy nazwę po ostatnim '/' i zamieniamy ':' na '-'
        clean = str(name).split('/')[-1].replace(':', '-')
        # Skróć jeśli za długie dla osi wykresu
        if len(clean) > 20:
            return clean[:17] + "..."
        return clean


    def _create_latency_bars(self, ax, model_names, avg_latencies, model_sizes, model_name=None):
        """Create a bar chart of average latencies sorted by model size."""
        print(f" DEBUG: _create_latency_bars called with {len(model_names)} models: {model_names}")
        # Sort by model size for better visualization
        sorted_indices = sorted(range(len(model_names)), key=lambda i: model_sizes[i])
        names = [model_names[i] for i in sorted_indices]
        lats = [avg_latencies[i] for i in sorted_indices]
        sizes = [model_sizes[i] for i in sorted_indices]

        # Shorten labels if possible


        # Create gradient colors based on model size
        bar_colors = self._get_model_size_gradient_colors(names, sizes)

        # Create bars (lats are in milliseconds)
        bars = ax.barh(range(len(names)), lats, color=bar_colors)

        # Add value labels in seconds (convert from ms to s)
        for i, bar in enumerate(bars):
            value_seconds = bar.get_width() / 1000  # Convert ms to s
            ax.text(bar.get_width(), i, f'{value_seconds:.3f}s', 
                   va='center', ha='left', fontsize=8)

        # Set labels and title
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        # Check language preference
        try:
            use_polish = getattr(self, '_current_use_polish', True)
        except:
            use_polish = True

        if use_polish:
            ax.set_xlabel('Średnia latencja (ms)')
            if model_name is not None:
                ax.set_title(f'Porównanie latencji modeli - {model_name}')
            else:
                ax.set_title('Porównanie latencji modeli')
        else:
            ax.set_xlabel('Average Latency (ms)')
            if model_name is not None:
                ax.set_title(f'Model Latency Comparison - {model_name}')
            else:
                ax.set_title('Model Latency Comparison')
        ax.grid(axis='x', linestyle='--', alpha=0.6)


    def _create_model_table(self, ax, model_names, models_data, model_name_prefix=None):
        """Create a table with model specifications."""
        # Prepare data for the table
        table_data = []

        for name in model_names:
            # Handle both full model names and shortened names
            if model_name_prefix is not None:
                # For shortened names (e.g., "q2_K"), reconstruct full name with prefix
                full_name = f"{model_name_prefix}-{name}"
                full_name = full_name.replace("_", ":", 1)
                print(f"Full name short: {full_name}")
            else:
                # For full model names, convert underscores to colons
                full_name = name.replace("_", ":", 1)
                print(f"Full name long: {full_name}")

            model_data = models_data[name]
            metadata = self.all_model_metadata.get("model", {}).get(full_name, {})


            # Get model size with unit
            size_gb = metadata.get('model_size_gb', 0)
            size_str = f"{size_gb:.1f} GB" if size_gb > 0 else "N/A"

            # Get parameter count with unit
            params = metadata.get('parameter_size_display', 0)
            params_str = params

            # Get quantization info
            quant = metadata.get('quantization_level', 'Unknown')

            # Get architecture
            arch = metadata.get('architecture', 'Unknown')

            # Get model parameters (context_size, max_tokens, etc.)
            parameters = self.MULTI_TURN_GLOBAL_CONFIG.copy()
            for key in ['cot_prompt_path', 'cot_schema_path', 'display_prompt', 
                       'validation_cot_prompt_path', 'validation_schema_path']:
                parameters.pop(key, None)

            # Format parameters as string (shorter version)
            params_config = ', '.join([f"{k}: {v}" for k, v in list(parameters.items())[:3]])  # Only first 3 params

            # Get energy data
            cpu_power_values = model_data.get('cpu_power', [])
            gpu_power_values = model_data.get('gpu_power', [])

            avg_cpu = np.mean(cpu_power_values) if cpu_power_values else 0
            avg_gpu = np.mean(gpu_power_values) if gpu_power_values else None
            avg_total = avg_cpu + (avg_gpu if avg_gpu is not None else 0)

            # Format energy data with better display
            if avg_cpu > 0 or (avg_gpu is not None and avg_gpu > 0):
                gpu_str = f"{avg_gpu:.0f}" if avg_gpu is not None and avg_gpu > 0 else "0"
                energy_str = f"CPU: {avg_cpu:.0f}mW\nGPU: {gpu_str}mW\nTotal: {avg_total:.0f}mW"
            else:
                energy_str = "Brak danych\nenergii"

            # Add row to table
            table_data.append([
                name,
                arch,
                params_str,
                size_str,
                quant,
                energy_str
            ])

        # Set column widths based on content (adjusted for energy data)
        col_widths = [0.25, 0.15, 0.1, 0.1, 0.1, 0.3]

        # Check language preference
        try:
            use_polish = getattr(self, '_current_use_polish', True)
        except:
            use_polish = True

        if use_polish:
            col_labels = ['Model', 'Architektura', 'Parametry', 'Rozmiar', 'Kwantyzacja', 'Energia (mW)']
        else:
            col_labels = ['Model', 'Architecture', 'Parameters', 'Size', 'Quantization', 'Energy (mW)']
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colWidths=col_widths
        )

        # Style the table with better layout
        table.auto_set_font_size(False)
        table.set_fontsize(7)  # Slightly larger but still compact
        table.scale(1, 2.0)  # Increase row height for better readability

        # Adjust cell properties
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_fontsize(8)
                cell.set_text_props(weight='bold')
            # Enable text wrapping for energy column
            if j == 5:  # Energy column
                cell.set_text_props(wrap=True)
                cell.set_height(0.15)  # Make energy cells taller

        ax.axis('off')

        # Check language preference for title
        try:
            use_polish = getattr(self, '_current_use_polish', True)
        except:
            use_polish = True

        title_text = 'Specyfikacje Modeli' if use_polish else 'Model Specifications'
        ax.set_title(title_text, pad=15, fontsize=10)


    def _calculate_efficiency_metrics(self, models_data):
        """Calculate efficiency metrics for model comparison."""
        efficiency_data = {}
        for model_name, data in models_data.items():
            # Calculate average scores and latencies
            avg_gpt_score = np.mean(data['gpt_scores']) if data.get('gpt_scores') else 0

            # Calculate average latency and throughput
            latencies = [l for l in data.get('latency', []) if isinstance(l, (int, float))]
            avg_latency = np.mean(latencies) if latencies else 0

            # Get metadata
            metadata = self.all_model_metadata.get("model", {}).get(model_name, {})

            # Store calculated metrics
            efficiency_data[model_name] = {
                'gpt_score': avg_gpt_score,
                'latency_ms': avg_latency,
                'model_size_gb': metadata.get('model_size_gb', 0),
                'parameters': metadata.get('parameter_size_display', 0),
                'quantization_level': metadata.get('quantization_level', 'N/A')
            }

        return efficiency_data


    def _create_score_comparison(self, model_names, models_data, output_dir, filename_prefix, use_polish=True):
        """Creates a bar chart comparing GPT Judge scores across all models."""
        import matplotlib.pyplot as plt
        import numpy as np

        scores = []
        names = []
        for name in model_names:
            score = models_data[name].get('avg_gpt_score', 0)
            scores.append(score)
            names.append(name)

        # Sort by score desc
        if scores:
            sorted_indices = np.argsort(scores)[::-1]
            scores = [scores[i] for i in sorted_indices]
            names = [names[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, scores, color='skyblue')

        # Labels
        ax.set_ylabel('GPT Judge Score' if not use_polish else 'Ocena GPT Judge')
        ax.set_title('Model Score Comparison' if not use_polish else 'Porównanie Ocen Modeli')
        ax.set_ylim(0, 10.5)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)

        # Values
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        path = os.path.join(output_dir, f"{filename_prefix}_scores.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f" Score comparison saved: {path}")
        return path


    def _create_resource_comparison(self, model_names, models_data, output_dir, filename_prefix, use_polish=True):
        """Creates a grouped bar chart comparing CPU and GPU power usage across all models."""
        import matplotlib.pyplot as plt
        import numpy as np

        cpu_powers = []
        gpu_powers = []
        names = []

        # Collect data
        for name in model_names:
            # Try to get average CPU/GPU power
            cpu = models_data[name].get('avg_cpu_power', 0)
            gpu = models_data[name].get('avg_gpu_power', 0)

            # If 0, try computing from list if present
            if cpu == 0 and 'cpu_power' in models_data[name]:
                l = models_data[name]['cpu_power']
                cpu = np.mean(l) if l else 0

            if gpu == 0 and 'gpu_power' in models_data[name]:
                l = models_data[name]['gpu_power']
                gpu = np.mean(l) if l else 0

            cpu_powers.append(cpu)
            gpu_powers.append(gpu)
            names.append(name)

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, cpu_powers, width, label='CPU', color='royalblue')
        rects2 = ax.bar(x + width/2, gpu_powers, width, label='GPU', color='orange')

        ax.set_ylabel('Power (mW)' if not use_polish else 'Moc (mW)')
        ax.set_title('Resource Usage Comparison' if not use_polish else 'Porównanie Zużycia Zasobów')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.text(rect.get_x() + rect.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=8)

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        path = os.path.join(output_dir, f"{filename_prefix}_resources.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"⚡ Resource comparison saved: {path}")
        return path


    def plot_per_round_with_reference(self, round_data, optimisation_type, model_name, agent_type, plotting_session_timestamp, metadata, output_dir, output_file_name, use_polish=True):
        """
        Wizualizuje wyniki pojedynczej rundy ewaluacji LLM.

        Ta funkcja tworzy szczegółowy wykres dla jednej rundy ewaluacji, pokazując:
        - Wyniki wszystkich metryk w formie wykresu słupkowego
        - Szczegółowe opisy metryk bez wartości liczbowych w legendzie
        - Rozkład latencji z procentowym udziałem każdego komponentu
        - Metadane modelu (architektura, parametry, quantization)
        - Szczegóły oceny GPT Judge z uzasadnieniami
        - Porównanie JSON: odpowiedź modelu vs referencja

        Args:
            metrics_dict (dict): Słownik z wynikami metryk dla rundy
            model_name (str, optional): Nazwa ewaluowanego modelu
            agent_type (Agent, optional): Typ agenta użytego w ewaluacji
            round_number (self.RoundNumber, optional): Numer rundy (0-3)
            llm_response_preview (str, optional): Odpowiedź modelu do wyświetlenia
            reference_response_preview (str, optional): Odpowiedź referencyjna
            latency_breakdown (dict, optional): Szczegółowy rozkład latencji
            metadata (dict, optional): Metadane modelu z Ollama
            output_dir (str, optional): Katalog do zapisu wykresu
            session_timestamp (str, optional): Timestamp sesji dla nazwy pliku

        Returns:
            str: Ścieżka do zapisanego pliku wykresu lub None w przypadku błędu

        Layout:
            2x3 grid - Panel metryk (2 rzędy) | Legenda/Metadata | GPT Judge
                                              | JSON Referencja  | JSON Model
        """
        import matplotlib.pyplot as plt
        import textwrap

        metrics_dict = round_data['metrics']
        round_number = round_data['round']
        llm_response_preview = round_data['llm_response']
        reference_response_preview = round_data['reference_response']
        latency_breakdown = round_data['latency_breakdown']

        valid_metrics = {k: v for k, v in metrics_dict.items()
                        if isinstance(v, dict) and 'score' in v}

        # Ensure a logical order for the bar chart
        preferred_order = [
            'json_validity', 'tool_names_match', 'tool_args_structure', 'tool_args_values',
            'levenshtein_similarity', 'p_bert', 'r_bert', 'bleu', 'rougeL', 'meteor',
            'gpt_judge'
        ]

        # Filter and sort based on preferred order, followed by remaining metrics
        sorted_names = [m for m in preferred_order if m in valid_metrics]
        remaining_names = [m for m in valid_metrics if m not in preferred_order]
        names = sorted_names + remaining_names

        values = [valid_metrics[m]['score'] for m in names]
        colors = ['green' if v >= 0.7 else 'orange' if v >=
                  0.4 else 'red' for v in values]

        # Tworzenie subplotów - layout 3 kolumny x 2 wiersze
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[
                              1, 1], hspace=0.4, wspace=0.3)

        # Panel 1: Wykres metryk (kolumna 1, zajmuje oba wiersze)
        ax1 = fig.add_subplot(gs[:, 0])

        # Panel 2: Metadane modelu + latencja + legenda metryk
        ax2 = fig.add_subplot(gs[0, 1])

        # Panel 3: GPT Judge szczegóły
        ax3 = fig.add_subplot(gs[0, 2])

        # Panel 4: Referencja JSON
        ax4 = fig.add_subplot(gs[1, 1])

        # Panel 5: Model JSON
        ax5 = fig.add_subplot(gs[1, 2])

        # === PANEL 1: Wykres metryk ===
        if not names or not values:
             print("[WARN] No valid metrics to plot in per-round analysis")
             # Plot empty placeholder
             ax1.text(0.5, 0.5, "Brak danych metryk", ha='center', va='center')
             bars = []
        else:
             bars = ax1.bar(names, values, color=colors)

        if use_polish:
            ax1.set_xlabel('Metryki')
            ax1.set_ylabel('Wynik (0-1)')
        else:
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Score (0-1)')

        # Tytuł z informacjami o modelu, agencie i rundzie
        title_parts = []
        if model_name:
            title_parts.append(f"Model: {model_name}")
        if agent_type:
            agent_name = agent_type.value if hasattr(
                agent_type, 'value') else str(agent_type)
            title_parts.append(f"Agent: {agent_name}")
        if round_number is not None:
            round_val = round_number.value if hasattr(
                round_number, 'value') else round_number
            if use_polish:
                title_parts.append(f"Runda: {round_val}")
            else:
                title_parts.append(f"Round: {round_val}")

        if title_parts:
            if use_polish:
                ax1.set_title(
                    f"Analiza Metryk LLM - Pojedyncza Runda\n{' | '.join(title_parts)}")
            else:
                ax1.set_title(
                    f"LLM Evaluation Metrics - Single Round Analysis\n{' | '.join(title_parts)}")
        else:
            if use_polish:
                ax1.set_title("Analiza Metryk LLM - Pojedyncza Runda")
            else:
                ax1.set_title("LLM Evaluation Metrics - Single Round Analysis")

        # Dodanie wartości na słupkach
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        if not bars:
            ax1.set_ylim(0, 1)
        else:
            ax1.set_ylim(0, 1.1)
        ax1.tick_params(axis='x', rotation=90, labelsize=8)
        ax1.grid(axis='y', alpha=0.3)

        # === PANEL 2: Metadane modelu + latencja + legenda metryk ===
        info_text = "MODEL METADATA\n\n"

        if metadata:
            info_text += f"• Name: {model_name}\n"
            info_text += f"• Architecture: {metadata.get('architecture', 'Unknown')}\n"
            info_text += f"• Parameters: {metadata.get('parameter_size_display', 'Unknown')}\n"
            info_text += f"• Quantization: {metadata.get('quantization_level', 'Unknown')}\n"
            info_text += f"• Format: {metadata.get('model_format', 'Unknown')}\n"
            info_text += f"• Size: {metadata.get('model_size_gb', 0)} GB\n\n"

        if latency_breakdown:
            info_text += "LATENCY BREAKDOWN\n\n"
            total_ms = latency_breakdown.get('total_ms')
            loading_ms = latency_breakdown.get('model_loading_ms')
            prompt_ms = latency_breakdown.get('prompt_evaluation_ms')
            generation_ms = latency_breakdown.get('token_generation_ms')

            # Oblicz procenty (tylko jeśli wartości nie są None)
            loading_pct = (loading_ms / total_ms) * 100 if (total_ms and loading_ms is not None) else None
            prompt_pct = (prompt_ms / total_ms) * 100 if (total_ms and prompt_ms is not None) else None
            generation_pct = (generation_ms / total_ms) * 100 if (total_ms and generation_ms is not None) else None

            # Formatuj wartości - pokaż "N/A" dla None
            def fmt(val, unit=""):
                return f"{val:.1f}{unit}" if val is not None else "N/A"

            info_text += f"• Total: {fmt(total_ms, 'ms')} (100.0%)\n"
            info_text += f"• Loading: {fmt(loading_ms, 'ms')} ({fmt(loading_pct, '%')})\n"
            info_text += f"• Prompt Eval: {fmt(prompt_ms, 'ms')} ({fmt(prompt_pct, '%')})\n"
            info_text += f"• Generation: {fmt(generation_ms, 'ms')} ({fmt(generation_pct, '%')})\n"
            if 'tokens' in latency_breakdown:
                tokens = latency_breakdown['tokens']
                throughput = tokens.get('throughput_tokens_per_sec')
                info_text += f"• Throughput: {fmt(throughput, ' tok/s')}\n\n"

        # Dodaj legendę metryk z długimi opisami
        info_text += "METRICS LEGEND\n\n"
        metric_descriptions = {
            'json_validity': 'Sprawdza czy odpowiedź zawiera poprawną składnię JSON bez błędów parsowania',
            'required_fields': 'Weryfikuje obecność wszystkich wymaganych pól w strukturze odpowiedzi',
            'tool_calls_format': 'Ocenia poprawność formatu wywołań narzędzi zgodnie ze specyfikacją',
            'tool_calls_count_match': 'Porównuje liczbę wywołań narzędzi z referencją',
            'tool_calls_names_match': 'Sprawdza zgodność nazw funkcji w wywołaniach narzędzi',
            'tool_calls_args_structure': 'Analizuje strukturę argumentów przekazanych do narzędzi',
            'exact_match': 'Dokładne dopasowanie tekstowe odpowiedzi z referencją',
            'jaccard_similarity': 'Podobieństwo Jaccard oparte na wspólnych tokenach',
            'levenshtein_similarity': 'Odległość edycyjna między tekstami odpowiedzi',
            'tool_args_values': 'Bezpośrednie porównanie wartości argumentów (case-insensitive)',
            'p_bert': 'BERTScore Precision - miara halucynacji (podobieństwo semantyczne)',
            'r_bert': 'BERTScore Recall - miara kompletności przekazu względem referencji',
            'bleu': 'Metryka BLEU używana w ocenie jakości tłumaczeń maszynowych',
            'rougeL': 'ROUGE-L mierzy najdłuższy wspólny podciąg dla oceny podsumowań',
            'meteor': 'METEOR uwzględnia synonimy i stemming w porównaniu tekstów',
            'gpt_context_correctness': 'Ocena GPT Judge dotycząca poprawności struktury JSON',
            'gpt_tool_call_correctness': 'Ocena GPT Judge jakości wywołań narzędzi',
            'gpt_reasoning_logic': 'Ocena GPT Judge logiki rozumowania w odpowiedzi',
            'gpt_question_naturalness': 'Ocena GPT Judge naturalności zadawanych pytań',
            'gpt_context_relevance': 'Ocena GPT Judge trafności odpowiedzi w kontekście',
            'gpt_judge': 'Średnia ocena ze wszystkich kryteriów GPT Judge'
        }

        for metric_name in names:
            if metric_name in metric_descriptions:
                desc = metric_descriptions[metric_name]
                if not use_polish:
                    # Simple mapping for common Polish descriptions to English
                    eng_desc = {
                        'json_validity': 'Checks if response has valid JSON syntax without parsing errors',
                        'required_fields': 'Verifies presence of all required fields in response structure',
                        'tool_calls_format': 'Evaluates tool call format correctness according to spec',
                        'tool_calls_count_match': 'Compares tool call count with reference',
                        'tool_calls_names_match': 'Checks tool function name matches',
                        'tool_calls_args_structure': 'Analyzes tool argument structure',
                        'exact_match': 'Exact text match between response and reference',
                        'jaccard_similarity': 'Jaccard similarity based on shared tokens',
                        'levenshtein_similarity': 'Edit distance similarity between texts',
                        'tool_args_values': 'Direct argument value comparison (case-insensitive)',
                        'p_bert': 'BERTScore Precision - hallucination measure (semantic similarity)',
                        'r_bert': 'BERTScore Recall - completeness measure vs reference',
                        'bleu': 'BLEU metric used in machine translation quality',
                        'rougeL': 'ROUGE-L measures longest common subsequence for summarization',
                        'meteor': 'METEOR considers synonyms and stemming in text comparison',
                        'gpt_context_correctness': 'GPT Judge score for JSON structure correctness',
                        'gpt_tool_call_correctness': 'GPT Judge score for tool call quality',
                        'gpt_reasoning_logic': 'GPT Judge score for reasoning logic',
                        'gpt_question_naturalness': 'GPT Judge score for question naturalness',
                        'gpt_context_relevance': 'GPT Judge score for context relevance',
                        'gpt_judge': 'Average score from all GPT Judge criteria'
                    }
                    if metric_name in eng_desc:
                        desc = eng_desc[metric_name]

                info_text += f"• {metric_name}: {desc}\n"

        ax2.text(0.1, 0.95, info_text, transform=ax2.transAxes, fontsize=6,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax2.axis('off')

        # === PANEL 3: GPT Judge szczegóły ===
        judge_text = "GPT JUDGE SZCZEGÓŁY\n\n"

        if 'gpt_judge' in valid_metrics and isinstance(valid_metrics['gpt_judge'], dict):
            gpt_data = valid_metrics['gpt_judge']
            judge_text += f"Ogólna ocena: {gpt_data.get('original_score', 0.0):.1f}/10\n\n"

            print(f" DEBUG: gpt_data keys: {gpt_data.keys()}")
            print(f" DEBUG: 'criteria_scores' in gpt_data: {'criteria_scores' in gpt_data}")

            if 'criteria_scores' in gpt_data:
                criteria_scores = gpt_data['criteria_scores']
                print(f" DEBUG: criteria_scores keys: {criteria_scores.keys()}")
                category_names = {
                    'json_correctness': 'JSON Correctness',
                    'tool_call_correctness': 'Tool Calls',
                    'reasoning_logic': 'Reasoning Logic',
                    'question_naturalness': 'Question Natural',
                    'context_relevance': 'Context Relevance'
                }

                for category in ['json_correctness', 'tool_call_correctness', 'reasoning_logic', 'question_naturalness', 'context_relevance']:
                    if category in criteria_scores:
                        score_data = criteria_scores[category]
                        name = category_names.get(category, category)

                        reason = score_data.get('explanation', 'Brak uzasadnienia') if isinstance(
                            score_data, dict) else 'Brak uzasadnienia'
                        score_value = score_data.get('score', score_data) if isinstance(
                            score_data, dict) else score_data

                        wrapped_reason = textwrap.fill(reason, width=40)
                        indented_reason = '\n'.join(
                            ['  ' + line for line in wrapped_reason.split('\n')])
                        judge_text += f"• {name}: {score_value}/10\n{indented_reason}\n\n"
            else:
                print("  DEBUG: 'criteria_scores' NOT FOUND in gpt_data!")

        ax3.text(0.0, 0.95, judge_text, transform=ax3.transAxes, fontsize=6,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        ax3.axis('off')

        # === PANEL 4: Referencja JSON ===
        ax4.text(0.2, 1, "REFERENCJA", transform=ax4.transAxes,
                 fontsize=10, ha='center', va='top', weight='bold', color='blue')

        if reference_response_preview:
            ref_json = self.pretty_json(reference_response_preview, 'str')
            ref_json = ref_json.replace('$', '\\$')
            ax4.text(0.0, 0.95, ref_json, transform=ax4.transAxes, fontsize=5,
                     verticalalignment='top', fontfamily='monospace', color='blue',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.2))

        ax4.axis('off')

        # === PANEL 5: Model JSON ===
        ax5.text(0.1, 1, "MODEL", transform=ax5.transAxes,
                 fontsize=10, ha='center', va='top', weight='bold', color='red')

        if llm_response_preview:
            llm_json = self.pretty_json(llm_response_preview, 'str')
            llm_json = llm_json.replace('$', '\\$')
            ax5.text(0.0, 0.95, llm_json, transform=ax5.transAxes, fontsize=5,
                     verticalalignment='top', fontfamily='monospace', color='red',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.2))

        ax5.axis('off')

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot
        plot_filename = f"{output_file_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f" Single round plot saved: {plot_path}")
        return plot_path


    def plot_latency_breakdown_timeline(self, session_data, agent_type, plotting_session_timestamp, output_dir, output_file_name, use_polish=True, model_name_prefix=None):
        """
        Creates a timeline visualization of latency breakdown and trends for referenced evaluation.
        Can handle both single session (dict) and multi-model sessions (list).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Debug: print session_data type and structure
        print(f"[DEBUG plot_latency_performance] session_data type: {type(session_data)}")
        if isinstance(session_data, list):
            print(f"[DEBUG plot_latency_performance] session_data is list with {len(session_data)} items")
            if len(session_data) > 0:
                print(f"[DEBUG plot_latency_performance] First item type: {type(session_data[0])}")
                if isinstance(session_data[0], dict):
                    print(f"[DEBUG plot_latency_performance] First item keys: {list(session_data[0].keys())[:10]}")
        elif isinstance(session_data, dict):
            print(f"[DEBUG plot_latency_performance] session_data is dict with keys: {list(session_data.keys())[:10]}")



        # Extract detailed latency and energy data
        models = []
        avg_latency_per_round = []
        total_completion_latency = []
        min_latency_per_round = []
        max_latency_per_round = []
        latency_std_dev = []
        round_latency_distributions = []

        # Energy data
        avg_cpu_power = []
        avg_gpu_power = []
        avg_total_power = []

        # GPT Judge data
        avg_gpt_scores = []
        gpt_score_distributions = []

        # Handle case where session_data contains multiple models (all_sessions)
        if 'all_sessions' in session_data:
            print(f"[DEBUG] Processing multiple models data: {len(session_data['all_sessions'])} models")
            all_sessions = session_data.get('all_sessions', [])

            for session in all_sessions:
                model_name = session.get('model_name', 'unknown_model')

                # Process rounds data for this model
                rounds_data = session.get('rounds', [])
                all_round_latencies = []
                all_cpu_power = []
                all_gpu_power = []
                all_gpt_scores = []

                for round_data in rounds_data:
                    if not isinstance(round_data, dict):
                        print(f"[WARN] Round data is not a dict: {round_data}")
                        continue

                    latency_breakdown = round_data.get('latency_breakdown', {})
                    if not isinstance(latency_breakdown, dict):
                        print(f"[WARN] latency_breakdown is not a dict: {latency_breakdown}")
                        continue

                    total_latency = latency_breakdown.get('total_ms', 0)
                    if isinstance(total_latency, (int, float)) and total_latency > 0:
                        all_round_latencies.append(total_latency)

                    # Extract energy data from resource_differences (delta values)
                    end_resources = latency_breakdown.get('end_resources', {})
                    energy_info = end_resources.get('energy', {})

                    cpu_power_mw = energy_info.get('cpu_power_mw', 0)
                    gpu_power_mw = energy_info.get('gpu_power_mw', 0)

                    # Debug energy data for radar
                    print(f"DEBUG CALCULATE_AVERAGES ENERGY (active): Model={model_name}, Round={round_data.get('round', 'N/A')}, CPU={cpu_power_mw}mW, GPU={gpu_power_mw}mW")

                    all_cpu_power.append(cpu_power_mw)
                    all_gpu_power.append(gpu_power_mw)

                    # Extract GPT Judge score
                    metrics = round_data.get('metrics', {})
                    gpt_judge = metrics.get('gpt_judge', {})
                    gpt_score = gpt_judge.get('score', 0) * 100  # Convert from 0-1 to 0-100%
                    all_gpt_scores.append(gpt_score)

                # Calculate statistics for this model
                if all_round_latencies:
                    avg = sum(all_round_latencies) / len(all_round_latencies)
                    min_lat = min(all_round_latencies)
                    max_lat = max(all_round_latencies)
                    std_dev = np.std(all_round_latencies) if len(all_round_latencies) > 1 else 0

                    avg_latency_per_round.append(avg)
                    min_latency_per_round.append(min_lat)
                    max_latency_per_round.append(max_lat)
                    latency_std_dev.append(std_dev)
                    round_latency_distributions.append(all_round_latencies)
                    total_completion_latency.append(sum(all_round_latencies))
                else:
                    print(f"[WARN] No valid latency data for model {model_name}")
                    avg_latency_per_round.append(0)
                    min_latency_per_round.append(0)
                    max_latency_per_round.append(0)
                    latency_std_dev.append(0)
                    round_latency_distributions.append([])
                    total_completion_latency.append(0)

                # Calculate energy statistics
                avg_cpu = np.mean(all_cpu_power) if all_cpu_power else 0
                avg_gpu = np.mean(all_gpu_power) if all_gpu_power else None
                avg_total = avg_cpu + (avg_gpu if avg_gpu is not None else 0)

                avg_cpu_power.append(avg_cpu)
                avg_gpu_power.append(avg_gpu)
                avg_total_power.append(avg_total)

                # Calculate GPT Judge statistics
                if all_gpt_scores:
                    avg_gpt = np.mean(all_gpt_scores)
                    avg_gpt_scores.append(avg_gpt)
                    gpt_score_distributions.append(all_gpt_scores)
                else:
                    avg_gpt_scores.append(0)
                    gpt_score_distributions.append([])

        # Handle case where session_data is the session data directly (single model)
        elif 'rounds' in session_data:
            print("[DEBUG] Processing single model data with rounds")
            model_name_from_session = session_data.get('model_name', model_name_prefix or 'unknown_model')
            models.append(model_name_from_session.replace(":", "_"))

            # Process rounds data
            rounds_data = session_data.get('rounds', [])
            all_round_latencies = []
            all_cpu_power = []
            all_gpu_power = []
            all_gpt_scores = []

            for round_data in rounds_data:
                if not isinstance(round_data, dict):
                    print(f"[WARN] Round data is not a dict: {round_data}")
                    continue

                latency_breakdown = round_data.get('latency_breakdown', {})
                if not isinstance(latency_breakdown, dict):
                    print(f"[WARN] latency_breakdown is not a dict: {latency_breakdown}")
                    continue

                total_latency = latency_breakdown.get('total_ms', 0)
                if isinstance(total_latency, (int, float)) and total_latency > 0:
                    all_round_latencies.append(total_latency)

                # Extract energy data from latency_breakdown.resource_differences (delta values)
                latency_breakdown = round_data.get('latency_breakdown', {})
                resource_differences = latency_breakdown.get('resource_differences', {})
                energy_diff = resource_differences.get('energy', {})
                cpu_delta = energy_diff.get('cpu_power_delta_mw')
                gpu_delta = energy_diff.get('gpu_power_delta_mw')
                cpu_power_mw = abs(cpu_delta) if cpu_delta is not None else 0
                gpu_power_mw = abs(gpu_delta) if gpu_delta is not None else 0

                # Debug energy data extraction (commented out for cleaner output)
                # if cpu_power_mw > 0 or gpu_power_mw > 0:
                #     print(f"[DEBUG ENERGY A] Found energy: CPU={cpu_power_mw}mW, GPU={gpu_power_mw}mW")

                all_cpu_power.append(cpu_power_mw)
                all_gpu_power.append(gpu_power_mw)

                # Extract GPT Judge score
                metrics = round_data.get('metrics', {})
                gpt_judge = metrics.get('gpt_judge', {})
                gpt_score = gpt_judge.get('score', 0) * 100  # Convert from 0-1 to 0-100%
                all_gpt_scores.append(gpt_score)

            # Calculate statistics
            if all_round_latencies:
                avg = sum(all_round_latencies) / len(all_round_latencies)
                min_lat = min(all_round_latencies)
                max_lat = max(all_round_latencies)
                std_dev = np.std(all_round_latencies) if len(all_round_latencies) > 1 else 0

                avg_latency_per_round.append(avg)
                min_latency_per_round.append(min_lat)
                max_latency_per_round.append(max_lat)
                latency_std_dev.append(std_dev)
                round_latency_distributions.append(all_round_latencies)

                print(f"[DEBUG] Processed {len(all_round_latencies)} rounds for {model_name_prefix}")
                print(f"  - Avg latency: {avg:.2f}ms")
                print(f"  - Min latency: {min_lat:.2f}ms")
                print(f"  - Max latency: {max_lat:.2f}ms")

                # If we have completion time, use it, otherwise estimate from latencies
                completion_time = session_data.get('avg_completion_time_sec', 0) * 1000  # Convert to ms
                if completion_time <= 0 and all_round_latencies:
                    completion_time = sum(all_round_latencies)
                total_completion_latency.append(completion_time)

                # Calculate energy statistics
                avg_cpu = np.mean(all_cpu_power) if all_cpu_power else 0
                avg_gpu = np.mean(all_gpu_power) if all_gpu_power else None
                avg_total = avg_cpu + (avg_gpu if avg_gpu is not None else 0)

                avg_cpu_power.append(avg_cpu)
                avg_gpu_power.append(avg_gpu)
                avg_total_power.append(avg_total)

                # Calculate GPT Judge statistics
                if all_gpt_scores:
                    avg_gpt = np.mean(all_gpt_scores)
                    avg_gpt_scores.append(avg_gpt)
                    gpt_score_distributions.append(all_gpt_scores)
                else:
                    avg_gpt_scores.append(0)
                    gpt_score_distributions.append([])
            else:
                print("[WARN] No valid latency data found in rounds")
                return None

        else:
            # Handle list of session data (for all_models_plots)
            print("[DEBUG] Processing list of models data")
            if isinstance(session_data, list):
                for model_session in session_data:
                    if not isinstance(model_session, dict):
                        continue

                    model_name_item = model_session.get('model_name', 'unknown_model')
                    models.append(model_name_item.replace(":", "_"))

                    # Process rounds data for this model
                    rounds_data = model_session.get('rounds', [])
                    print(f"[DEBUG] Model {model_name_item}: found {len(rounds_data)} rounds")
                    all_round_latencies = []
                    all_cpu_power = []
                    all_gpu_power = []
                    all_gpt_scores = []

                    for round_data in rounds_data:
                        if not isinstance(round_data, dict):
                            continue
                        latency_breakdown = round_data.get('latency_breakdown', {})
                        if not isinstance(latency_breakdown, dict):
                            continue
                        total_latency = latency_breakdown.get('total_ms', 0)
                        if isinstance(total_latency, (int, float)) and total_latency > 0:
                            all_round_latencies.append(total_latency)

                        # Extract energy data from latency_breakdown.resource_differences (delta values)
                        latency_breakdown = round_data.get('latency_breakdown', {})
                        resource_differences = latency_breakdown.get('resource_differences', {})
                        energy_diff = resource_differences.get('energy', {})
                        cpu_delta = energy_diff.get('cpu_power_delta_mw')
                        gpu_delta = energy_diff.get('gpu_power_delta_mw')
                        cpu_power_mw = abs(cpu_delta) if cpu_delta is not None else 0
                        gpu_power_mw = abs(gpu_delta) if gpu_delta is not None else 0

                        # Debug energy data extraction (commented out for cleaner output)
                        # if cpu_power_mw > 0 or gpu_power_mw > 0:
                        #     print(f"[DEBUG ENERGY B] Found energy: CPU={cpu_power_mw}mW, GPU={gpu_power_mw}mW")

                        all_cpu_power.append(cpu_power_mw)
                        all_gpu_power.append(gpu_power_mw)

                        # Extract GPT Judge score
                        metrics = round_data.get('metrics', {})
                        gpt_judge = metrics.get('gpt_judge', {})
                        gpt_score = gpt_judge.get('score', 0) * 100  # Convert from 0-1 to 0-100%
                        all_gpt_scores.append(gpt_score)

                    # Calculate statistics for this model
                    if all_round_latencies:
                        avg = sum(all_round_latencies) / len(all_round_latencies)
                        min_lat = min(all_round_latencies)
                        max_lat = max(all_round_latencies)
                        std_dev = np.std(all_round_latencies) if len(all_round_latencies) > 1 else 0

                        avg_latency_per_round.append(avg)
                        min_latency_per_round.append(min_lat)
                        max_latency_per_round.append(max_lat)
                        latency_std_dev.append(std_dev)
                        round_latency_distributions.append(all_round_latencies)
                        total_completion_latency.append(sum(all_round_latencies))

                        # Calculate energy statistics
                        avg_cpu = np.mean(all_cpu_power) if all_cpu_power else 0
                        avg_gpu = np.mean(all_gpu_power) if all_gpu_power else None
                        avg_total = avg_cpu + (avg_gpu if avg_gpu is not None else 0)

                        avg_cpu_power.append(avg_cpu)
                        avg_gpu_power.append(avg_gpu)
                        avg_total_power.append(avg_total)

                        # Calculate GPT Judge statistics
                        if all_gpt_scores:
                            avg_gpt = np.mean(all_gpt_scores)
                            avg_gpt_scores.append(avg_gpt)
                            gpt_score_distributions.append(all_gpt_scores)
                        else:
                            avg_gpt_scores.append(0)
                            gpt_score_distributions.append([])
                    else:
                        avg_latency_per_round.append(0)
                        min_latency_per_round.append(0)
                        max_latency_per_round.append(0)
                        latency_std_dev.append(0)
                        round_latency_distributions.append([])
                        total_completion_latency.append(0)

                        # Add default values for energy and GPT Judge
                        avg_cpu_power.append(0)
                        avg_gpu_power.append(None)
                        avg_total_power.append(0)
                        avg_gpt_scores.append(0)
                        gpt_score_distributions.append([])
            else:
                # Single session data without 'rounds' key
                print("[WARN] Unexpected session_data format")
                return None

        if not models:
            print("[ERROR] No valid model data found")
            return None

        # Calculate model sizes for gradient colors using full model names from session_data
        model_sizes = []
        for model in models:
            # Handle both full model names and shortened names
            if model_name_prefix is not None:
                # For shortened names (e.g., "q2_K"), reconstruct full name with prefix
                full_model_name = f"{model_name_prefix}-{model}"
                full_model_name = full_model_name.replace("_", ":", 1)
                print(f"Full model name short: {full_model_name}")
            else:
                # For full model names, convert underscores to colons
                full_model_name = model.replace("_", ":", 1)
                print(f"Full model name long: {full_model_name}")

            # Get model metadata using converted name
            metadata = getattr(self, 'all_model_metadata', {})
            model_metadata = metadata.get('model', {}).get(full_model_name, {}) if isinstance(metadata, dict) else {}
            size_gb = model_metadata.get('model_size_gb', 0)
            print(f"Model {model} -> {full_model_name} -> size: {size_gb}GB")

            model_sizes.append(size_gb)

        # Get gradient colors based on model size
        model_colors = self._get_model_size_gradient_colors(models, model_sizes)

        # Sort models alphabetically and reorder all data accordingly
        sorted_indices = sorted(range(len(models)), key=lambda i: models[i])
        models = [models[i] for i in sorted_indices]
        model_sizes = [model_sizes[i] for i in sorted_indices]
        model_colors = [model_colors[i] for i in sorted_indices]
        avg_latency_per_round = [avg_latency_per_round[i] for i in sorted_indices]
        min_latency_per_round = [min_latency_per_round[i] for i in sorted_indices]
        max_latency_per_round = [max_latency_per_round[i] for i in sorted_indices]
        latency_std_dev = [latency_std_dev[i] for i in sorted_indices]
        round_latency_distributions = [round_latency_distributions[i] for i in sorted_indices]
        total_completion_latency = [total_completion_latency[i] for i in sorted_indices]
        avg_cpu_power = [avg_cpu_power[i] for i in sorted_indices]
        avg_gpu_power = [avg_gpu_power[i] for i in sorted_indices]
        avg_total_power = [avg_total_power[i] for i in sorted_indices]
        avg_gpt_scores = [avg_gpt_scores[i] for i in sorted_indices]
        gpt_score_distributions = [gpt_score_distributions[i] for i in sorted_indices]

        # Create comprehensive latency visualization (2x2 grid)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        # Add model_name_prefix to title if provided
        title_suffix = f" - {model_name_prefix}" if model_name_prefix else ""

        if use_polish:
            fig.suptitle(f'Kompleksowa Analiza Latencji\nEwaluacja z Referencją{title_suffix}', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'Comprehensive Latency Analysis\nReferenced Evaluation{title_suffix}', 
                        fontsize=16, fontweight='bold')

        # Debug energy data
        print(f"DEBUG FINAL ENERGY DATA: avg_cpu_power={avg_cpu_power}, avg_gpu_power={avg_gpu_power}")

        # 1. Energy Consumption Comparison (CPU vs GPU Power)
        if len(avg_cpu_power) > 0 and len(avg_gpu_power) > 0:
            # Use gradient colors based on model size
            for i, (cpu, gpu) in enumerate(zip(avg_cpu_power, avg_gpu_power)):
                ax1.scatter(cpu, gpu, s=100, alpha=0.7, c=[model_colors[i]],
                           label=models[i] if i < len(models) else f'Model {i}')

            # Add legend for models
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        elif len(avg_cpu_power) == 0:
            ax1.text(0.5, 0.5, 'Brak danych energii (Empty Data)', transform=ax1.transAxes, 
                    ha='center', va='center', fontsize=12)
        else:
             # Case where we have lists but they might be mismatched or contain Nones that caused issues
             ax1.text(0.5, 0.5, 'Insufficient Energy Data', transform=ax1.transAxes,
                     ha='center', va='center', fontsize=12)

        # Check language preference (use_polish parameter already passed to function)
        if use_polish:
            ax1.set_xlabel('Średnia moc CPU (mW)', fontweight='bold')
            ax1.set_ylabel('Średnia moc GPU (mW)', fontweight='bold')
            ax1.set_title('Porównanie Zużycia Energii (CPU vs GPU)', fontweight='bold')
        else:
            ax1.set_xlabel('Average CPU Power (mW)', fontweight='bold')
            ax1.set_ylabel('Average GPU Power (mW)', fontweight='bold')
            ax1.set_title('Energy Consumption Comparison (CPU vs GPU)', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Latency Range (Min-Max) with Error Bars
        x_pos = np.arange(len(models))
        bars = ax2.bar(x_pos, avg_latency_per_round,
               yerr=[np.array(avg_latency_per_round) - np.array(min_latency_per_round),
                     np.array(max_latency_per_round) - np.array(avg_latency_per_round)],
               capsize=5, alpha=0.7)

        # Color bars with gradient colors based on model size
        for bar, color in zip(bars, model_colors):
            bar.set_color(color)

        if use_polish:
            ax2.set_xlabel('Modele', fontweight='bold')
            ax2.set_ylabel('Latencja na rundę (ms)', fontweight='bold')
            ax2.set_title('Zakres latencji (Min-Śr-Maks)', fontweight='bold')
        else:
            ax2.set_xlabel('Models', fontweight='bold')
            ax2.set_ylabel('Latency per Round (ms)', fontweight='bold')
            ax2.set_title('Latency Range (Min-Avg-Max)', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=6)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels with smaller font and better positioning
        for i, (avg, std) in enumerate(zip(avg_latency_per_round, latency_std_dev)):
            ax2.text(i, avg + std + max(avg_latency_per_round) * 0.02,
                    f'{avg:.0f}ms\n±{std:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=6)

        # 3. GPT Judge Score Distribution Box Plot
        if any(gpt_score_distributions):
            valid_distributions = [dist for dist in gpt_score_distributions if dist]
            valid_models = [models[i] for i, dist in enumerate(gpt_score_distributions) if dist]
            valid_colors = [model_colors[i] for i, dist in enumerate(gpt_score_distributions) if dist]
            box_plot = ax3.boxplot(
                valid_distributions,
                tick_labels=valid_models,
                patch_artist=True,
                notch=True,
                showmeans=True
            )

            # Color the boxes with gradient colors based on model size
            for patch, color in zip(box_plot['boxes'], valid_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        if use_polish:
            ax3.set_xlabel('Modele', fontweight='bold')
            ax3.set_ylabel('Rozkład wyniku GPT Judge (%)', fontweight='bold')
            ax3.set_title('Rozkład wyniku GPT Judge na model', fontweight='bold')
        else:
            ax3.set_xlabel('Models', fontweight='bold')
            ax3.set_ylabel('GPT Judge Score Distribution (%)', fontweight='bold')
            ax3.set_title('GPT Judge Score Distribution per Model', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45, labelsize=6)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Latency Performance Categories
        latency_categories = []
        category_colors = []

        for avg_lat in avg_latency_per_round:
            if avg_lat < 10000:  # < 10s
                latency_categories.append('Excellent\n(<10s)')
                category_colors.append('#2E8B57')  # Sea green
            elif avg_lat < 15000:  # 10-15s
                latency_categories.append('Good\n(10-15s)')
                category_colors.append('#FFD700')  # Gold
            elif avg_lat < 20000:  # 15-20s
                latency_categories.append('Fair\n(15-20s)')
                category_colors.append('#FF8C00')  # Dark orange
            else:  # > 20s
                latency_categories.append('Poor\n(>20s)')
                category_colors.append('#DC143C')  # Crimson

        bars4 = ax4.bar(models, avg_latency_per_round, color=category_colors, alpha=0.8)

        # Add value labels with smaller font
        for bar, latency in zip(bars4, avg_latency_per_round):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(avg_latency_per_round) * 0.01,
                    f'{latency:.0f}ms', ha='center', va='bottom', fontweight='bold', fontsize=6)

        if use_polish:
            ax4.set_xlabel('Modele', fontweight='bold')
            ax4.set_ylabel('Śr. latencja na rundę (ms)', fontweight='bold')
            ax4.set_title('Kategorie wydajności latencji', fontweight='bold')
        else:
            ax4.set_xlabel('Models', fontweight='bold')
            ax4.set_ylabel('Avg Latency per Round (ms)', fontweight='bold')
            ax4.set_title('Latency Performance Categories', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45, labelsize=6)

        # Add category legend
        from matplotlib.patches import Patch
        if use_polish:
            legend_elements = [
                Patch(facecolor='#2E8B57', label='Doskonała (<10s)'),
                Patch(facecolor='#FFD700', label='Dobra (10-15s)'),
                Patch(facecolor='#FF8C00', label='Przeciętna (15-20s)'),
                Patch(facecolor='#DC143C', label='Słaba (>20s)')
            ]
        else:
            legend_elements = [
                Patch(facecolor='#2E8B57', label='Excellent (<10s)'),
                Patch(facecolor='#FFD700', label='Good (10-15s)'),
                Patch(facecolor='#FF8C00', label='Fair (15-20s)'),
                Patch(facecolor='#DC143C', label='Poor (>20s)')
            ]
        # Move legend below the plot
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)

        # Add model legend at the bottom
        if models:
            # Create model legend with colors
            model_colors = self._get_model_size_gradient_colors(models, [0] * len(models))  # Use default colors
            model_legend_elements = []
            for i, model in enumerate(models):
                model_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                      markerfacecolor=model_colors[i],
                                                      markersize=8, label=model.replace(":", "_")))

            # Add model legend below the performance legend
            fig.legend(handles=model_legend_elements, loc='lower center', 
                      bbox_to_anchor=(0.5, -0.15), ncol=min(6, len(models)),
                      title='Modele (Models)', fontsize=8)

        plt.tight_layout(rect=[0, 0.2, 1, 0.95])

        # Save plot
        plot_filename = f"{output_file_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"⏱️ Comprehensive Latency Analysis saved: {plot_path}")
        return plot_path


    def plot_aggr_over_rounds_with_reference(self, session_data, optimisation_type, agent_type, plotting_session_timestamp, metadata, output_dir, output_file_name, use_polish=True, model_name_prefix=None):
        """
        Tworzy zbiorczy wykres porównujący postęp latencji w kolejnych rundach dla wszystkich modeli.

        Args:
            results_data: Dictionary containing evaluation results data with 'rounds' key
            output_dir: Directory to save the output plot
            timestamp: Timestamp string for the plot title
            optimisation_type: String identifier for the optimization type (used in filename)

        Returns:
            str: Path to the saved plot, or None if no valid data was found
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from datetime import datetime



        # Handle list of session data for multiple models
        if isinstance(session_data, list):
            all_sessions = [s for s in session_data if isinstance(s, dict)]
            if not all_sessions:
                print("  all_sessions is empty or invalid")
                return None

            # Build per-model round latency map
            model_round_latencies = {}
            for session in all_sessions:
                model_name = session.get('model_name', 'unknown_model')
                rounds_data = session.get('rounds', [])
                per_round = {}
                for i, round_data in enumerate(rounds_data, 1):
                    if not isinstance(round_data, dict):
                        continue
                    latency_breakdown = round_data.get('latency_breakdown', {})
                    if not isinstance(latency_breakdown, dict):
                        continue
                    latency = latency_breakdown.get('total_ms', 0)
                    if isinstance(latency, (int, float)) and latency > 0:
                        per_round[i] = latency if i not in per_round else np.mean([per_round[i], latency])
                if per_round:
                    model_round_latencies[model_name] = per_round

            if not model_round_latencies:
                print("  No valid latency data in all_sessions")
                return None

            try:
                plt.style.use('ggplot')
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
                agent_name = agent_type.value if hasattr(agent_type, 'value') else str(agent_type)
                title_suffix = f" - {model_name_prefix}" if model_name_prefix else ""
                if use_polish:
                    fig.suptitle(f'Analiza czasu wykonania\n{agent_name} | Ewaluacja z Referencją{title_suffix}', 
                                fontsize=16, fontweight='bold')
                else:
                    fig.suptitle(f'Latency Timeline Analysis\n{agent_name} | Referenced Evaluation{title_suffix}', 
                                fontsize=16, fontweight='bold')


                # 1) Line plot for each model with gradient colors based on model size
                model_names = list(model_round_latencies.keys())
                model_sizes = []
                for model_name in model_names:
                    # Handle both full model names and shortened names with prefix3
                    if model_name_prefix is not None:
                        # For shortened names (e.g., "q2_K"), reconstruct full name with prefix
                        full_model_name = f"{model_name_prefix}-{model_name}"
                        full_model_name = full_model_name.replace("_", ":", 1)
                    else:
                        # For full model names, convert underscores to colons
                        full_model_name = model_name.replace("_", ":",1)

                    # Get model size from metadata
                        model_metadata = metadata.get('model', {}).get(full_model_name, {}) if isinstance(metadata, dict) else {}
                    size_gb = model_metadata.get('model_size_gb', 0)
                    model_sizes.append(size_gb)

                colors = self._get_model_size_gradient_colors(model_names, model_sizes)
                for idx, (mname, per_round) in enumerate(model_round_latencies.items()):
                    rounds_sorted = sorted(per_round.keys())
                    lat_list = [per_round[r] for r in rounds_sorted]
                    ax1.plot(rounds_sorted, lat_list, 'o-', linewidth=2, markersize=6,
                             label=full_model_name, color=colors[idx])
                    if len(rounds_sorted) > 1:
                        try:
                            z = np.polyfit(rounds_sorted, lat_list, 1)
                            p = np.poly1d(z)
                            ax1.plot(rounds_sorted, p(rounds_sorted), "--", alpha=0.5, color=colors[idx])
                        except Exception:
                            pass
                if use_polish:
                    ax1.set_xlabel('Numer rundy', fontweight='bold')
                    ax1.set_ylabel('Średnia latencja (ms)', fontweight='bold')
                    ax1.set_title('Postęp latencji przez rundy', fontweight='bold')
                else:
                    ax1.set_xlabel('Round Number', fontweight='bold')
                    ax1.set_ylabel('Average Latency (ms)', fontweight='bold')
                    ax1.set_title('Latency Progression Through Rounds', fontweight='bold')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid(True, alpha=0.3)

                # 2. Heatmap: models x rounds
                all_rounds = sorted({r for per_round in model_round_latencies.values() for r in per_round.keys()})
                heatmap_data = []
                model_names_order = []
                for mname, per_round in model_round_latencies.items():
                    model_names_order.append(mname.replace(":", "_"))
                    model_row = [per_round.get(r, 0) for r in all_rounds]
                    heatmap_data.append(model_row)
                im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                ax2.set_xticks(range(len(all_rounds)))
                ax2.set_xticklabels([f'Round {r}' for r in all_rounds])
                ax2.set_yticks(range(len(model_names_order)))
                ax2.set_yticklabels(model_names_order)
                # annotate cells
                max_val = max((max(row) for row in heatmap_data if row), default=0)
                for i in range(len(model_names_order)):
                    for j in range(len(all_rounds)):
                        val = heatmap_data[i][j]
                        if isinstance(val, (int, float)) and val > 0:
                            ax2.text(j, i, f'{val:.0f}', ha='center', va='center',
                                     color='black' if max_val == 0 or val < max_val/2 else 'white',
                                     fontweight='bold', fontsize=8)

                if use_polish:
                    ax2.set_title('Mapa czasu wykonania przez model & rundę (ms)', fontweight='bold')
                    cbar = plt.colorbar(im, ax=ax2)
                    cbar.set_label('Latencja (ms)', fontweight='bold')
                else:
                    ax2.set_title('Latency Heatmap by Model & Round (ms)', fontweight='bold')
                    cbar = plt.colorbar(im, ax=ax2)
                    cbar.set_label('Latency (ms)', fontweight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_filename = f"{output_file_name}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📈 Latency Timeline Analysis saved (multi-model): {plot_path}")
                return plot_path
            except Exception as e:
                print(f"Error creating multi-model latency timeline analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                return None

        # Handle single session data
        elif 'rounds' in session_data and isinstance(session_data['rounds'], list):
            rounds_data = session_data['rounds']
            model_name = session_data.get('model_name', 'unknown_model')

            # Extract round-by-round latency data
            model_round_latencies = {model_name: {}}

            # Process each round's data
            round_latencies = {}

            for i, round_data in enumerate(rounds_data, 1):
                if not isinstance(round_data, dict):
                    print(f"  Skipping invalid round data at index {i}")
                    continue

                # Get latency breakdown, default to empty dict if not present
                latency_breakdown = round_data.get('latency_breakdown', {})
                if not isinstance(latency_breakdown, dict):
                    latency_breakdown = {}

                # Get total latency, default to 0 if not available
                latency = latency_breakdown.get('total_ms', 0)

                # Only include valid latencies
                if isinstance(latency, (int, float)) and latency > 0:
                    round_latencies[i] = [latency]  # Use 1-based round numbering

            # Calculate average latency per round (in this case, just one value per round)
            avg_round_latencies = {}
            for round_num, latencies in round_latencies.items():
                if latencies:  # Should always be true here since we filtered above
                    avg_round_latencies[round_num] = np.mean(latencies)

            if not avg_round_latencies:
                print("  No valid latency data found in any rounds")
                return None

            model_round_latencies[model_name] = avg_round_latencies
        else:
            print("  Unexpected session_data format for latency timeline")
            return None

        # Check if we have any valid data to plot
        if not model_round_latencies:
            print("  No valid latency data found to create timeline analysis")
            return None

        try:
            # Create timeline plot with a built-in style
            plt.style.use('ggplot')  # Using ggplot style which is built into matplotlib
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            # Add model_name to title if provided
            title_suffix = f" - {model_name}" if model_name else ""
            if use_polish:
                fig.suptitle(f'Analiza czasu wykonania\nEwaluacja z Referencją {title_suffix}', 
                            fontsize=16, fontweight='bold')
            else:
                fig.suptitle(f'Latency Timeline Analysis\nReferenced Evaluation {title_suffix}', 
                        fontsize=16, fontweight='bold')

            # 1. Line plot - latency progression through rounds
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_round_latencies)))

            for i, (model_name, round_latencies) in enumerate(model_round_latencies.items()):
                if round_latencies:
                    rounds = sorted(round_latencies.keys())
                    latencies = [round_latencies[r] for r in rounds]

                    ax1.plot(rounds, latencies, 'o-', linewidth=2, markersize=6,
                            label=model_name.replace(":", "_"), color=colors[i])

                    # Add trend line if we have enough points
                    if len(rounds) > 1:
                        try:
                            z = np.polyfit(rounds, latencies, 1)
                            p = np.poly1d(z)
                            ax1.plot(rounds, p(rounds), "--", alpha=0.5, color=colors[i])
                        except Exception as e:
                            print(f"  Could not add trend line for {model_name}: {str(e)}")

            ax1.set_xlabel('Round Number', fontweight='bold')
            ax1.set_ylabel('Latency (ms)', fontweight='bold')
            ax1.set_title('Latency Progression Through Rounds', fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

            # 2. Heatmap - latency by round (single model)
            all_rounds = sorted(avg_round_latencies.keys())
            if not all_rounds:
                print("  No valid round data for heatmap")
                return None

            # Create matrix with just one row for the single model
            heatmap_data = [[avg_round_latencies[r] for r in all_rounds]]

            # Create heatmap
            im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

            # Set ticks and labels
            ax2.set_xticks(range(len(all_rounds)))
            ax2.set_xticklabels([f'Round {r}' for r in all_rounds])
            ax2.set_yticks([0])
            ax2.set_yticklabels([model_name.replace(":", "_")])

            # Add text annotations
            max_val = max([max(row) for row in heatmap_data]) if heatmap_data else 0
            for j in range(len(all_rounds)):
                value = heatmap_data[0][j]
                if value > 0:  # Only add text for non-zero values
                    text = ax2.text(j, 0, f'{value:.0f}',
                                 ha="center", va="center", 
                                 color="black" if value < max_val/2 else "white",
                                 fontweight='bold')

            if use_polish:
                ax2.set_title('Mapa czasu wykonania przez model & rundę (ms)', fontweight='bold')
            else:
                ax2.set_title('Latency Heatmap (ms)', fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)

            cbar.set_label('Latency (ms)', fontweight='bold')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Generate single model timeline
            plot_filename = f"{output_file_name}.png"
            plot_path = os.path.join(output_dir, plot_filename)

            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📈 Latency Timeline Analysis saved: {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error creating latency timeline analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


    def plot_throttling_timeline(self, session, output_dir, timestamp):
        """
        Generuje liniowy wykres zasobów w czasie (rundach) dla konkretnego modelu.
        Pozwala zaobserwować narastanie użycia SWAP i spadek częstotliwości (throttling).
        """
        if not session or 'rounds' not in session or not session['rounds']:
            return

        model_name = session.get('model_name', 'model')
        rounds_idx = []
        ram_used = []
        swap_used = []
        cpu_freq = []
        gpu_util = []
        cpu_freq_max = 0

        for r in session['rounds']:
            lb = r.get('latency_breakdown', {})
            start_res = lb.get('start_resources', {})
            mem = start_res.get('memory', {})
            cpu = start_res.get('device', {}).get('cpu_freq', {})
            gpu = start_res.get('device', {}).get('gpu_util', 0)

            rounds_idx.append(r.get('round'))
            ram_used.append(mem.get('ram_used_gb', 0))
            swap_used.append(mem.get('swap_used_gb', 0))
            cpu_freq.append(cpu.get('current', 0))
            gpu_util.append(gpu if gpu is not None else 0)

            if cpu.get('max', 0) > cpu_freq_max:
                cpu_freq_max = cpu.get('max', 0)

        if not rounds_idx:
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot 1: Memory Pressure
        ax1.plot(rounds_idx, ram_used, 'o-', color='#2ecc71', label='RAM (GB)', linewidth=2)
        ax1.plot(rounds_idx, swap_used, 's--', color='#e74c3c', label='SWAP (GB)', linewidth=2)
        ax1.set_ylabel('Użycie Pamięci (GB)')
        ax1.set_title(f'📈 Oś czasu zasobów: {model_name}\nRAM & Memory Pressure', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: CPU Frequency (Throttling)
        if cpu_freq_max > 0:
            cpu_freq_pct = [f/cpu_freq_max*100 for f in cpu_freq]
            ax2.plot(rounds_idx, cpu_freq_pct, '^-', color='#f1c40f', label='CPU Freq (%)', linewidth=2)
            ax2.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
            ax2.set_ylim(0, 110)
            ax2.set_ylabel('% Maks. Częstotliwości')
            ax2.set_title('Stabilność CPU (Thermal Throttling)', fontsize=10)
        else:
            ax2.text(0.5, 0.5, "Brak danych o CPU", ha='center')

        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: GPU Utilization
        ax3.plot(rounds_idx, gpu_util, 'd-', color='#9b59b6', label='GPU Util (%)', linewidth=2)
        ax3.set_ylabel('GPU %')
        ax3.set_ylim(0, 105)
        ax3.set_title('Użycie GPU', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        ax3.set_xlabel('Numer Rundy')

        plt.tight_layout()
        # Use session's model name for the filename to avoid overwrites in all_models mode
        session_model_name = self._model_name_to_id(session.get('model_name', 'unknown'))
        plot_path = os.path.join(output_dir, f"throttling_timeline_{session_model_name}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Throttling timeline plot saved: {plot_path}")


    def plot_resource_health_check(self, session_data, output_dir, timestamp):
        """
        Generuje ujednolicony wykres diagnostyczny (RAM, Swap, CPU) z dwiema liniami limitu.
        """
        if not session_data:
            return

        models = []
        ram_used = []
        swap_used = []
        cpu_freq_start = []
        cpu_freq_end = []
        cpu_freq_max = []

        for session in session_data:
            model_name = self._shorten_model_name(session.get('model_name', 'unknown'))
            if 'rounds' in session and session['rounds']:
                last_round = session['rounds'][-1]
                if 'latency_breakdown' in last_round:
                    lb = last_round['latency_breakdown']
                    start_res = lb.get('start_resources', {})
                    mem = start_res.get('memory', {})
                    models.append(model_name)
                    ram_used.append(mem.get('ram_used_gb', 0))
                    swap_used.append(mem.get('swap_used_gb', 0))

                    cpu = start_res.get('device', {}).get('cpu_freq', {})
                    cpu_end_data = lb.get('end_resources', {}).get('device', {}).get('cpu_freq', {})
                    cpu_freq_start.append(cpu.get('current', 0))
                    cpu_freq_max.append(cpu.get('max', 0))
                    cpu_freq_end.append(cpu_end_data.get('current', 0))

        if not models:
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        x = np.arange(len(models))
        width = 0.6

        # --- Subplot 1: RAM Usage ---
        # Fixed color for legend consistency
        ax1.bar(x, ram_used, width, label='Użycie RAM', color='#2ecc71', alpha=0.8)

        # Dual Limit Lines
        ax1.axhline(y=16, color='#2c3e50', linestyle='--', linewidth=2, label='Limit MacBook (16GB)')
        ax1.axhline(y=8, color='#c0392b', linestyle='-.', linewidth=2, label='Limit iPhone (8GB)')

        ax1.set_ylabel('Użycie RAM (GB)')
        ax1.set_title('Resource Health Check: RAM (MacBook Air vs iPhone)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15, ha='right')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.set_ylim(0, max(ram_used + [16]) * 1.1)

        # Highlight critical bars with a border instead of changing fill color
        for i, ram in enumerate(ram_used):
            if ram > 8.0:
                ax1.bar(x[i], ram, width, color='#2ecc71', edgecolor='#c0392b', linewidth=2, alpha=0.9, label='Model > 8GB' if i == 0 else "")

        # --- Subplot 2: SWAP Usage ---
        ax2.bar(x, swap_used, width, label='SWAP (Pamięć Wirtualna)', color='#95a5a6', hatch='//', alpha=0.8)
        # Standard system-level thresholds
        ax2.axhline(y=1.0, color='#f1c40f', linestyle=':', label='Ostrzeżenie (1GB)')
        ax2.axhline(y=2.0, color='#e67e22', linestyle=':', label='Wysokie Obciążenie (2GB)')
        ax2.axhline(y=4.0, color='#c0392b', linestyle='--', label='Krytyczne Obciążenie (4GB)')

        ax2.set_ylabel('Użycie SWAP (GB)')
        ax2.set_title('Obciążenie Pamięci: SWAP / SSD Dysk', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=15, ha='right')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, axis='y', alpha=0.3)

        for i, swap in enumerate(swap_used):
            if swap > 1.0:
                ax2.text(i, swap + 0.1, f'{swap:.1f} GB', ha='center', color='#c0392b', fontweight='bold')

        # --- Subplot 3: CPU Stability ---
        freq_start_pct = [s/m*100 if m else 0 for s, m in zip(cpu_freq_start, cpu_freq_max)]
        freq_end_pct = [e/m*100 if m else 0 for e, m in zip(cpu_freq_end, cpu_freq_max)]

        width_bar = 0.35
        ax3.bar(x - width_bar/2, freq_start_pct, width_bar, label='Taktowanie Start', color='#3498db')
        ax3.bar(x + width_bar/2, freq_end_pct, width_bar, label='Taktowanie Koniec', color='#f1c40f')
        ax3.axhline(y=100, color='red', linestyle=':', label='Częstotliwość Nominalna (100%)')

        ax3.set_ylabel('% Maks. Częstotliwości')
        ax3.set_title('Stabilność CPU (Brak Throttlingu = 100%)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=15, ha='right')
        ax3.set_ylim(0, 115)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"resource_health_check_unified_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Updated Unified Resource Health Check saved: {plot_path}")


    def plot_mobile_analysis_visualizations(self, session_data, optimisation_type, agent_type, plotting_session_timestamp, metadata, output_dir, output_file_name, use_polish=True, model_name_prefix=None):
        """
         MOBILE-FOCUSED ANALYSIS: Golden Model Selection for Mobile Deployment
        Analyzes model performance from a mobile deployment perspective.

        Args:
            session_data (dict): The evaluation session data
            output_dir (str): Directory to save the visualizations
            timestamp (str, optional): Timestamp for the analysis. Defaults to current time.

        Returns:
            str: Path to the saved visualization or None if failed
        """
        import os
        import numpy as np

        # Set language preference for this function
        self._current_use_polish = use_polish
        import matplotlib.pyplot as plt
        from datetime import datetime

        if not session_data:
            print("No session data provided for mobile analysis")
            return None




        # Extract model data from session
        models_data = {}

        # Create consistent color mapping for models
        model_color_map = {}

        # Handle list of session data for multiple models (all_models_plots)
        if isinstance(session_data, list):
            # Multiple models evaluation
            for model_session in session_data:
                if not isinstance(model_session, dict):
                    continue

                model_name = model_session.get('model_name', 'unknown_model')
                # Skip failed models (Granite MoE)
                if 'granite3-moe' in model_name.lower() or 'moe' in model_name.lower():
                    continue

                rounds = model_session.get('rounds', [])


                if not rounds:
                    continue

                # Process latency, GPT scores, and energy data for this model
                latencies = []
                gpt_scores = []
                all_cpu_power = []
                all_gpu_power = []
                all_throughputs = []

                # Initialize lists for models_data entry
                models_data[model_name] = {
                    'latencies': [],
                    'gpt_scores': [],
                    'cpu_power': [],
                    'gpu_power': [],
                    'throughput': [],
                    'memory_usage': []
                }

                for round_data in rounds:
                    # Get latency
                    if isinstance(round_data, dict) and 'latency_breakdown' in round_data:
                        latency = round_data['latency_breakdown'].get('total_ms', 0)
                        if isinstance(latency, (int, float)) and latency > 0:
                            models_data[model_name]['latencies'].append(latency)

                        # Get token throughput (Tokens per second)
                        tokens_data = round_data['latency_breakdown'].get('tokens', {})
                        if isinstance(tokens_data, dict):
                            # Try multiple keys for throughput for robustness
                            tps = tokens_data.get('throughput_tokens_per_sec')
                            if tps is None:
                                tps = tokens_data.get('predicted_per_second')
                            if tps is None:
                                tps = tokens_data.get('throughput', 0)

                            if isinstance(tps, (int, float)) and tps > 0:
                                models_data[model_name]['throughput'].append(tps)

                    # Get GPT judge score
                    if isinstance(round_data, dict) and 'metrics' in round_data:
                        gpt_data = round_data['metrics'].get('gpt_judge', {})
                        if isinstance(gpt_data, dict):
                            gpt_score = gpt_data.get('original_score')
                            if gpt_score is None:
                                gpt_score = gpt_data.get('score')
                                if gpt_score is not None and gpt_score <= 1.0 and gpt_score > 0:
                                    gpt_score *= 10.0
                        else:
                            gpt_score = gpt_data

                        if gpt_score is not None:
                            models_data[model_name]['gpt_scores'].append(gpt_score)

                    # Extract energy data from end_resources (active power draw during inference)
                    end_resources = round_data.get('latency_breakdown', {}).get('end_resources', {})
                    if not end_resources:
                        # Try direct from round_data if nested failed
                        end_resources = round_data.get('end_resources', {})

                    energy_info = end_resources.get('energy', {})
                    cpu_power_mw = energy_info.get('cpu_power_mw', 0)
                    gpu_power_mw = energy_info.get('gpu_power_mw', 0)

                    # Store as absolute power
                    models_data[model_name]['cpu_power'].append(cpu_power_mw)
                    models_data[model_name]['gpu_power'].append(gpu_power_mw)

                    # Memory usage for health check table
                    mem_usage = round_data.get('metrics', {}).get('memory_used_gb', 0)
                    if not mem_usage:
                        mem_usage = end_resources.get('memory', {}).get('ram_used_gb', 0)
                    if mem_usage:
                        models_data[model_name]['memory_usage'].append(mem_usage)

                # Calculate averages for models_data entry
                latencies = models_data[model_name]['latencies']
                gpt_scores = models_data[model_name]['gpt_scores']
                all_cpu_power = models_data[model_name]['cpu_power']
                all_gpu_power = models_data[model_name]['gpu_power']
                all_throughputs = models_data[model_name]['throughput']

                if not latencies:  # Skip if no valid latency data
                    del models_data[model_name] # Remove entry if no valid data
                    continue

                # Define metadata_model_name for lookup
                if model_name_prefix is not None:
                    metadata_model_name = f"{model_name_prefix}-{model_name}".replace("_", ":", 1)
                else:
                    metadata_model_name = model_name

                # Robust metadata lookup
                model_meta = metadata.get('model', {}) if isinstance(metadata, dict) else {}
                model_metadata = model_meta.get(metadata_model_name, {})
                if not model_metadata:
                    model_metadata = model_meta.get(metadata_model_name.replace(":", "_"), {})
                if not model_metadata:
                    model_metadata = model_meta.get(metadata_model_name.replace("_", ":"), {})
                if not model_metadata:
                    model_metadata = model_meta.get(model_name, {})

                # Check if size_gb is 0, try to find it in model_session data as fallback
                if isinstance(model_metadata, dict) and model_metadata.get('model_size_gb', 0) == 0 and 'metadata' in model_session:
                    model_metadata['model_size_gb'] = model_session['metadata'].get('model_size_gb', 0)
                print(f"DEBUG MOBILE ANALYSIS METADATA: model_name={model_name}, metadata_model_name={metadata_model_name}, model_size_gb={model_metadata.get('model_size_gb')}")

                # Calculate energy statistics
                avg_cpu_power = np.mean(all_cpu_power) if all_cpu_power else 0
                avg_gpu_power = np.mean(all_gpu_power) if all_gpu_power else 0

                models_data[model_name] = {
                    'latencies': latencies,
                    'avg_latency': np.mean(latencies) if latencies else 0,
                    'min_latency': min(latencies) if latencies else 0,
                    'max_latency': max(latencies) if latencies else 0,
                    'std_dev': np.std(latencies) if len(latencies) > 1 else 0,
                    'num_rounds': len(rounds),
                    'gpt_scores': gpt_scores,
                    'avg_gpt_score': np.mean(gpt_scores) if gpt_scores else 0,
                    'min_gpt_score': min(gpt_scores) if gpt_scores else 0,
                    'max_gpt_score': max(gpt_scores) if gpt_scores else 0,
                    'gpt_judge_score': np.mean(gpt_scores) if gpt_scores else 0,  # Alias for compatibility
                    # Add energy data
                    'cpu_power': all_cpu_power,
                    'gpu_power': all_gpu_power,
                    'avg_cpu_power': avg_cpu_power,
                    'avg_gpu_power': avg_gpu_power,
                    'avg_token_throughput': np.mean(all_throughputs) if all_throughputs else 0,
                    # Add model metadata
                    'model_size_gb': model_metadata.get('model_size_gb'),
                    'parameter_size_display': model_metadata.get('parameter_size_display'),
                    'architecture': model_metadata.get('architecture'),
                    'quantization_level': model_metadata.get('quantization_level')
                }

                # Debug energy data
                if avg_cpu_power > 0 or avg_gpu_power > 0:
                    print(f"DEBUG MODEL ENERGY for {model_name}: CPU={avg_cpu_power:.0f}mW, GPU={avg_gpu_power:.0f}mW")

            # Create gradient color mapping based on model size
            sorted_model_names = sorted(models_data.keys())
            model_sizes = []
            for name in sorted_model_names:
                # model_size_gb is stored directly in models_data[name], not in metadata sub-dict
                size = models_data[name].get('model_size_gb', 0)
                if size is None:
                    size = 0
                model_sizes.append(size)
                print(f"Mobile analysis - Model {name}: size = {size}GB")
            colors = self._get_model_size_gradient_colors(sorted_model_names, model_sizes)
            for i, model_name in enumerate(sorted_model_names):
                model_color_map[model_name] = colors[i]

            # print(f"Models data: {models_data}")


        # Generate mobile-focused visualizations
        try:
            # Create a 2x1 grid for mobile performance metrics
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
            # Add model_name_prefix to title if provided
            title_suffix = f" - {model_name_prefix}" if model_name_prefix else ""
            fig.suptitle(f'Mobile Performance Analysis{title_suffix}\n', fontsize=16, fontweight='bold')

            # Sort models by average latency (best to worst)
            sorted_models = sorted(models_data.items(), key=lambda x: x[1]['avg_latency'])
            model_names = [m[0] for m in sorted_models]
            avg_latencies = [m[1]['avg_latency'] for m in sorted_models]
            min_latencies = [m[1]['min_latency'] for m in sorted_models]
            max_latencies = [m[1]['max_latency'] for m in sorted_models]
            std_devs = [m[1]['std_dev'] for m in sorted_models]

            # 1. Latency Comparison (Bar chart with error bars)
            x = np.arange(len(model_names))
            width = 0.6

            # Use consistent colors for models
            bar_colors = [model_color_map[name] for name in model_names]

            bars = ax1.bar(x, avg_latencies, width,
                         yerr=std_devs,
                         color=bar_colors,
                         alpha=0.7)

            # Add min/max markers
            for i, (min_lat, max_lat) in enumerate(zip(min_latencies, max_latencies)):
                ax1.plot([i - width/3, i + width/3], [min_lat, min_lat], 'k_')
                ax1.plot([i - width/3, i + width/3], [max_lat, max_lat], 'k_')
                ax1.plot([i, i], [min_lat, max_lat], 'k-')

            # Add value labels
            for i, (rect, avg, std) in enumerate(zip(bars, avg_latencies, std_devs)):
                height = rect.get_height()
                ax1.text(rect.get_x() + rect.get_width()/2., height + max(avg_latencies)*0.05,
                        f'{avg:.1f}ms\n±{std:.1f}',
                        ha='center', va='bottom', fontsize=9)

            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            if use_polish:
                ax1.set_ylabel('Latencja (ms)')
                ax1.set_title('Porównanie latencji modeli (Niżej lepiej)')
            else:
                ax1.set_ylabel('Latency (ms)')
                ax1.set_title('Model Latency Comparison (Lower is Better)')
            ax1.grid(True, alpha=0.3, axis='y')

            # 2. Throughput vs Latency (Scatter plot)
            # Calculate throughput (requests per second) using avg latency only
            # rps ≈ 1000 / avg_latency_ms
            throughputs = []
            model_sizes_gb = []
            for name in model_names:
                # Use actual token throughput (TPS) instead of requests per second
                tps = models_data[name].get('avg_token_throughput', 0)

                # Fallback if TPS is zero but latency is available (simplistic RPS)
                if tps == 0:
                    avg_lat = models_data[name]['avg_latency']
                    tps = 1000.0 / avg_lat if isinstance(avg_lat, (int, float)) and avg_lat > 0 else 0.0

                throughputs.append(tps)

                # Get model size for bubble size - stored directly in models_data
                model_size = models_data[name].get('model_size_gb', 0)
                if model_size is None:
                    model_size = 0
                model_sizes_gb.append(model_size)

            # Use model size as bubble size (normalize to 50-2000 range for better visibility)
            if len(model_sizes_gb) > 0:
                # Filter out None values just in case
                safe_sizes = [s if s is not None and s > 0 else 0.1 for s in model_sizes_gb]

                min_size = min(safe_sizes) if safe_sizes else 0.1
                max_size = max(safe_sizes) if safe_sizes else 0.1

                # Avoid division by zero
                range_size = (max_size - min_size)
                if range_size <= 0:
                    range_size = 1.0 # arbitrary non-zero value

                bubble_sizes = [50 + ((s - min_size) / range_size) * 1950 for s in safe_sizes]
            else:
                bubble_sizes = [500] * len(model_names)  # Default size if no size data

            # Use consistent colors for models
            scatter_colors = [model_color_map[name] for name in model_names]

            if len(avg_latencies) > 0 and len(throughputs) > 0:
                 scatter = ax2.scatter(avg_latencies, throughputs,
                                     s=bubble_sizes,
                                     c=scatter_colors,
                                     alpha=0.6)
            else:
                 print("  Skipping scatter plot in mobile analysis: empty data arrays")
                 return None

            if model_name_prefix is not None:
                # Add labels for each point
                for i, (lat, thr) in enumerate(zip(avg_latencies, throughputs)):
                    ax2.text(lat, thr, model_names[i],
                            fontsize=9, ha='center', va='center')

            if use_polish:
                ax2.set_xlabel('Średnia latencja (ms)')
                ax2.set_ylabel('Przepustowość (tokeny/sekundę)')
                ax2.set_title('Analiza Wydajności: TPS vs Latencja (Rozmiar ∝ Rozmiar modelu)')
            else:
                ax2.set_xlabel('Average Latency (ms)')
                ax2.set_ylabel('Throughput (tokens/sec)')
                ax2.set_title('Performance Analysis: TPS vs Latency (Bubble size ∝ Model Size)')
            ax2.grid(True, alpha=0.3)
            # Legend is redundant here since bar chart above identifies models
            # Add legend for bubble sizes
            size_legend_elements = []
            if model_sizes_gb:
                min_size_gb = min(model_sizes_gb) if min(model_sizes_gb) > 0 else 0.1
                max_size_gb = max(model_sizes_gb)
                mid_size_gb = (min_size_gb + max_size_gb) / 2

                # Use Line2D for model color legend (fixed size)
                from matplotlib.lines import Line2D
                color_legend_elements = []
                for i, name in enumerate(model_names):
                    color_legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                                     label=name,
                                                     markerfacecolor=model_color_map[name],
                                                     markersize=10, alpha=0.6))

                # Use Line2D for SIZE legend instead of empty scatter
                size_legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, alpha=0.6, label=f'{min_size_gb:.1f}GB'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, alpha=0.6, label=f'{mid_size_gb:.1f}GB'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=16, alpha=0.6, label=f'{max_size_gb:.1f}GB')
                ]

                # Only add legend for Model Size since model names are in the bar chart above
                ax2.legend(handles=size_legend_elements, title='Model Size', loc='upper right', fontsize=8)

            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            # Save the plot
            output_file_name = f"{optimisation_type}_mobile_performance_analysis_{plotting_session_timestamp}"
            plot_filename = f"{output_file_name}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Return a dictionary with plot type and path
            mobile_plots = {
                'performance_analysis': plot_path
            }

            # If we have category winners analysis, add it to the plots
            output_file_name = f"{optimisation_type}_mobile_category_winners_{plotting_session_timestamp}"
            category_winners_plot = self._create_category_winners_analysis_referenced(
                models_data = models_data,
                output_file_name = output_file_name,
                output_dir = output_dir,
                timestamp = plotting_session_timestamp,
                optimisation_type = optimisation_type
            )
            if category_winners_plot:
                mobile_plots['category_winners'] = category_winners_plot

            print(f"📱 Mobile performance analysis saved: {plot_path}")
            return mobile_plots

        except Exception as e:
            print(f"Error generating mobile analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}


    def _calculate_mobile_score_referenced(self, gpt_score, latency_ms, model_size_gb):
        """Calculate mobile readiness score for referenced evaluation

        Args:
            gpt_score (float): GPT evaluation score (0-1)
            latency_ms (float): Average latency in milliseconds
            model_size_gb (float): Model size in GB

        Returns:
            float: Mobile readiness score (0-100)
        """
        # Mobile scoring weights
        success_weight = 0.4
        speed_weight = 0.3  # Lower latency = better
        size_weight = 0.3   # Smaller size = better

        # Normalize scores (0-1)
        success_score = max(0, min(1, gpt_score))  # Ensure 0-1 range
        speed_score = max(0, 1 - (latency_ms / 30000))  # 30s as max acceptable

        # Handle None model_size_gb
        if model_size_gb is None:
            model_size_gb = 0
        size_score = max(0, 1 - (model_size_gb / 10))  # 10GB as max acceptable

        # Calculate weighted score and convert to 0-100 scale
        mobile_score = (
            (success_score * success_weight) +
            (speed_score * speed_weight) +
            (size_score * size_weight)
        ) * 100

        return min(100, max(0, mobile_score))  # Ensure 0-100 range


    def _create_category_winners_analysis_referenced(self, models_data,output_dir,timestamp, optimisation_type, output_file_name='category_winners', model_name_prefix=None):
        """C) Category Winners Analysis for Referenced

        Returns:
            str: Path to the saved plot or None if failed
        """
        if not models_data:
            return None
        # print("models_data!!!")
        # print(models_data)
        # Find winners in each category using REAL data
        best_gpt = None
        best_latency = None
        best_size = None
        best_mobile = None

        for mdl_name, model_data in models_data.items():
            # Get real data from models_data
            # mdl_name from models_data already has correct format from JSON
            if model_name_prefix is not None:
                full_model_name = f"{model_name_prefix}-{mdl_name}"
                full_model_name = full_model_name.replace("_", ":", 1)
            else:
                # mdl_name already has correct format, no replacement needed
                full_model_name = mdl_name
            gpt_score = model_data.get('avg_gpt_score', 0)  # Already in 0-1 range
            avg_latency = model_data.get('avg_latency', 0)
            model_size = model_data.get('model_size_gb', 0)
            print(f"DEBUG CATEGORY WINNERS: mdl_name={mdl_name}, full_model_name={full_model_name}, model_size={model_size}")

            # Calculate mobile score
            mobile_score = self._calculate_mobile_score_referenced(gpt_score, avg_latency, model_size)

            # Find best GPT Judge score (HIGHER is better)
            if best_gpt is None or gpt_score > best_gpt[1].get('gpt_judge_score', 0):
                best_gpt = (mdl_name, {
                    'gpt_judge_score': gpt_score,
                    'avg_latency_ms': avg_latency,
                    'model_size_gb': model_size,
                    'mobile_score': mobile_score,
                    'latencies': model_data.get('latencies', []),
                    'min_latency': model_data.get('min_latency', 0),
                    'max_latency': model_data.get('max_latency', 0),
                    'std_dev': model_data.get('std_dev', 0),
                    'num_rounds': model_data.get('num_rounds', 0)
                })

            # Find best latency (LOWER is better)
            if best_latency is None or avg_latency < best_latency[1].get('avg_latency_ms', float('inf')):
                best_latency = (mdl_name, {
                    'gpt_judge_score': gpt_score,
                    'avg_latency_ms': avg_latency,
                    'model_size_gb': model_size,
                    'mobile_score': mobile_score,
                    'latencies': model_data.get('latencies', []),
                    'min_latency': model_data.get('min_latency', 0),
                    'max_latency': model_data.get('max_latency', 0),
                    'std_dev': model_data.get('std_dev', 0),
                    'num_rounds': model_data.get('num_rounds', 0)
                })


            if best_size is None or (model_size is not None and model_size < best_size[1].get('model_size_gb', float('inf'))):
                best_size = (mdl_name, {
                    'gpt_judge_score': gpt_score,
                    'avg_latency_ms': avg_latency,
                    'model_size_gb': model_size,
                    'mobile_score': mobile_score,
                    'latencies': model_data.get('latencies', []),
                    'min_latency': model_data.get('min_latency', 0),
                    'max_latency': model_data.get('max_latency', 0),
                    'std_dev': model_data.get('std_dev', 0),
                    'num_rounds': model_data.get('num_rounds', 0)
                })

            # Find best mobile score (HIGHER is better)
            if best_mobile is None or mobile_score > best_mobile[1].get('mobile_score', 0):
                best_mobile = (mdl_name, {
                    'gpt_judge_score': gpt_score,
                    'avg_latency_ms': avg_latency,
                    'model_size_gb': model_size,
                    'mobile_score': mobile_score,
                    'latencies': model_data.get('latencies', []),
                    'min_latency': model_data.get('min_latency', 0),
                    'max_latency': model_data.get('max_latency', 0),
                    'std_dev': model_data.get('std_dev', 0),
                    'num_rounds': model_data.get('num_rounds', 0)
                })

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        title_suffix = f" - {model_name_prefix}" if model_name_prefix else ""
        fig.suptitle(f'ANALIZA ZWYCIĘZCÓW KATEGORII\nNajlepsze modele wg kryteriów mobilnych (Referenced Evaluation){title_suffix}', 
                    fontsize=16, fontweight='bold')

        # 1. GPT Judge Score Winner

        gpt_score = best_gpt[1].get('gpt_judge_score', 0) * 100  # Convert to percentage
        ax1.bar(['Zwycięzca'], [gpt_score], 
               color='#2E8B57', alpha=0.8, width=0.5)
        ax1.set_title(f'[CEL] ZWYCIĘZCA GPT JUDGE\n{best_gpt[0].replace(":", "_")}', fontweight='bold')
        ax1.set_ylabel('Wynik GPT Judge (%)')
        ax1.text(0, gpt_score + 2,
                f'{gpt_score:.1f}%', ha='center', fontweight='bold', fontsize=14)
        ax1.set_ylim(0, 100)

        # 2. Latency Winner (Speed)
        latency_s = best_latency[1].get('avg_latency_ms', 0) / 1000  # Default to 0 if missing
        ax2.bar(['Zwycięzca'], [latency_s], 
               color='#FFD700', alpha=0.8, width=0.5)
        ax2.set_title(f'[CEL] ZWYCIĘZCA LATENCJI (Najszybszy)\n{best_latency[0].replace(":", "_")}', fontweight='bold')
        ax2.set_ylabel('Latencja na rundę (s)')
        if latency_s > 0:
            ax2.text(0, latency_s + 0.5,
                    f'{latency_s:.1f}s', ha='center', fontweight='bold', fontsize=14)
        else:
            ax2.text(0, 0.5, 'N/A', ha='center', fontweight='bold', fontsize=14)

        # 3. Size Winner (Efficiency)
        size_gb = best_size[1].get('model_size_gb')
        if size_gb is None:
            size_gb = 0.0  # Default to 0.0 if missing
        ax3.bar(['Zwycięzca'], [size_gb], 
               color='#4169E1', alpha=0.8, width=0.5)
        ax3.set_title(f'[DYSK] ZWYCIĘZCA ROZMIARU (Najmniejszy)\n{best_size[0].replace(":", "_")}', fontweight='bold')
        ax3.set_ylabel('Rozmiar modelu (GB)')
        if size_gb > 0:
            ax3.text(0, size_gb + 0.1,
                    f'{size_gb:.1f}GB', ha='center', fontweight='bold', fontsize=14)
        else:
            ax3.text(0, 0.5, 'N/A', ha='center', fontweight='bold', fontsize=14)

        # 4. Golden Model (Mobile Score)
        mobile_score = best_mobile[1].get('mobile_score', 0)  # Default to 0 if missing
        ax4.bar(['Zwycięzca'], [mobile_score],    
               color='#FF6347', alpha=0.8, width=0.5)
        ax4.set_title(f'[PUCHAR] ZŁOTY MODEL (Ranking Mobilny)\n{best_mobile[0].replace(":", "_")}', fontweight='bold')
        ax4.set_ylabel('Wynik Gotowości Mobilnej')
        if mobile_score > 0:
            ax4.text(0, mobile_score + 2,
                    f'{mobile_score:.0f}/100', ha='center', fontweight='bold', fontsize=14)
        else:
            ax4.text(0, 50, 'N/A', ha='center', fontweight='bold', fontsize=14)
        ax4.set_ylim(0, 100)

        # Get model metadata
        # best_mobile[0] already has correct format from JSON
        if model_name_prefix is not None:
            # For shortened names (e.g., "fp16"), reconstruct full name with prefix
            metadata_model_name = f"{model_name_prefix}-{best_mobile[0]}"
            metadata_model_name = metadata_model_name.replace("_", ":", 1)
        else:
            # best_mobile[0] already has correct format, no replacement needed
            metadata_model_name = best_mobile[0]

        model_metadata = self.all_model_metadata.get("model", {}).get(metadata_model_name, {})
        params = model_metadata.get('parameter_size_display', 'N/A')
        size_gb = best_mobile[1].get('model_size_gb', 0)

        # Format parameters for display
        if isinstance(params, (int, float)):
            if params >= 1e9:
                params_str = f"{params/1e9:.1f}B"
            elif params >= 1e6:
                params_str = f"{params/1e6:.1f}M"
            elif params >= 1e3:
                params_str = f"{params/1e3:.1f}K"
            else:
                params_str = str(params)
        else:
            params_str = str(params)


        # Add golden model details with parameters and size (handle None values)
        size_display = f"{size_gb:.1f}GB" if size_gb is not None else "N/A"
        gpt_score = best_mobile[1].get('gpt_judge_score')
        gpt_score_display = f"{gpt_score*100:.1f}%" if gpt_score is not None else "N/A"
        latency = best_mobile[1].get('avg_latency_ms')
        latency_display = f"{latency/1000:.1f}s" if latency is not None else "N/A"
        mobile_score = best_mobile[1].get('mobile_score')
        mobile_score_display = f"{mobile_score:.0f}/100" if mobile_score is not None else "N/A"

        golden_details = f"""
SZCZEGÓŁY ZŁOTEGO MODELU:
• Rodzina: [{model_metadata.get('architecture', 'unknown')}]
• Parametry: {params_str}
• Rozmiar: {size_display}
• Kwantyzacja: {best_mobile[1].get('quantization_level', 'N/A')}
• Wynik GPT Judge: {gpt_score_display}
• Latencja: {latency_display}
• Wynik Mobilny: {mobile_score_display}
• Liczba rund: {best_mobile[1].get('total_rounds', 0)}
        """

        fig.text(0.02, 0.02, golden_details, fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
        # Save and close
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plot_path = os.path.join(output_dir, f'{optimisation_type}_category_winners_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Persist golden model for later retrieval
        self._golden_model = best_mobile[0]
        return {'category_winners': plot_path, 'golden_model': self._golden_model}


    def generate_best_models_summary(self, session_locations, timestamp, use_polish=True):
        """Generates a summary of best performing models and uploads to Neptune."""
        print("\n Generating Best Models Summary...")
        log_file = session_locations["log_file"]
        all_models_log_file = log_file.replace(f"{self.agent_type}_evaluation_results", f"all_models_{self.agent_type}_evaluation_results")

        # Prefer the all_models log file if it exists and has content
        target_log_file = all_models_log_file if os.path.exists(all_models_log_file) else log_file

        # Filter only by agent_type, not by full parameters dict (which may not match exactly)
        models_data = self.get_last_sessions(
             key_dict={"agent_type": self.agent_type},
             log_file=target_log_file,
             group_by_keys=['model_name']
        )

        if not models_data:
             print("⚠️ No model data found for summary.")
             return

        summary_data = []
        for model_name, sessions in models_data.items():
            # Handle both list of sessions and single session dict
            if isinstance(sessions, list):
                # Use the last session
                session = sessions[-1]
            else:
                 session = sessions

            rounds = session.get('rounds', [])
            if not rounds:
                continue

            # Aggregate metrics
            latencies = []
            gpt_scores = []

            for r in rounds:
                metrics = r.get('metrics', {})
                breakdown = r.get('latency_breakdown', {})

                # GPT Score
                gpt_score = metrics.get('gpt_judge', {}).get('score', 0)
                gpt_scores.append(gpt_score)

                # Latency
                total_ms = breakdown.get('total_ms', 0)
                if total_ms:
                    latencies.append(total_ms)

            avg_score = sum(gpt_scores) / len(gpt_scores) if gpt_scores else 0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            model_info = self.all_model_metadata.get("model", {}).get(model_name.replace(":", "_"), {})
            model_size = model_info.get('model_size_gb', 0)

            summary_data.append({
                "Model": model_name,
                "Avg GPT Score (0-1)": round(avg_score, 3),
                "Avg Latency (ms)": round(avg_latency, 1),
                "Size (GB)": model_size
            })

        if not summary_data:
             print("⚠️ No valid metrics found for summary.")
             return

        # Sort by Score (descending)
        summary_data.sort(key=lambda x: x["Avg GPT Score (0-1)"], reverse=True)

        # Create Markdown Table
        import pandas as pd
        df = pd.DataFrame(summary_data)
        md_table = df.to_markdown(index=False)

        leaderboard_path = os.path.join(session_locations["all_models_run_folder"], f"leaderboard_{timestamp}.md")
        with open(leaderboard_path, "w") as f:
            f.write(f"#  Model Leaderboard - {timestamp}\n\n")
            f.write(md_table)

        print(f" Leaderboard saved to: {leaderboard_path}")

        # Upload to Neptune
        if self.neptune.run:
             self.neptune.upload_artifact(leaderboard_path, "leaderboard_summary")
