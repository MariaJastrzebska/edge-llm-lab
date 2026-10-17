"""Extracted evaluator methods; historical plotting contracts are preserved."""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt

class UnreferencedPlots:
    def per_patient_plots(self, session_locations, timestamp, list_of_models=None):
        """
        Tworzy wykresy dla każdego pacjenta dla wybranych optymalizacji.
        Analogiczne do per_round_plots w referenced.
        """
        def _create_patient_plots_for_single_model(model_name):
            model_comparison_dict = {
                "model_name": model_name,
                "parameters": self.MULTI_TURN_GLOBAL_CONFIG, 
                "tools": self.tools, 
                "cot_prompt": self.cot_prompt
            }
            group_by_keys = ['optimisation']

            valid_session_data = self.get_last_sessions(
                key_dict=model_comparison_dict,
                log_file=session_locations["log_file"], 
                group_by_keys=group_by_keys
            )

            for optimisation, session_data in valid_session_data.items():
                for patient_data in session_data.get('patients', []):
                    # Użyj oryginalnego wykresu per patient
                    self.plot_per_patient_summary(patient_data, output_dir, timestamp)
                    print(f"Patient plot saved for: {patient_data.get('patient_name', 'unknown')}")

        if list_of_models is None:
            output_dir = session_locations["model_output_directory"]
            metadata = self.current_model_metadata
            model_name = self.model_name
            _create_patient_plots_for_single_model(self.model_name)
        else:
            for model_name in list_of_models:
                # Create model-specific session for this model
                model_session_locations = self.create_session(model_name=model_name)
                output_dir = model_session_locations["model_run_folder"]
                metadata = self.all_model_metadata.get("model", {}).get(model_name, {}) if isinstance(self.all_model_metadata, dict) else {}
                _create_patient_plots_for_single_model(model_name)
                del model_session_locations  # Free memory


    def per_model_plots(self, session_locations, timestamp, list_of_models=None):
        """
        Tworzy wykresy agregowane dla modelu - jeden wykres per optymalizacja.
        Analogiczne do per_model_plots w referenced.
        """
        def _create_model_plots_for_single_model(model_name):
            model_comparison_dict = {
                "model_name": model_name,
                "parameters": self.MULTI_TURN_GLOBAL_CONFIG, 
                "tools": self.tools, 
                "cot_prompt": self.cot_prompt
            }
            group_by_keys = ['optimisation']
            valid_session_data = self.get_last_sessions(key_dict=model_comparison_dict, log_file=session_locations["log_file"], group_by_keys=group_by_keys)

            for optimisation, session_data in valid_session_data.items():
                # Użyj oryginalnego wykresu model summary
                self.plot_model_summary(session_data, output_dir, timestamp)
                print(f"Model summary plot saved for optimisation: {optimisation}")

        if list_of_models is None:
            output_dir = session_locations["model_output_directory"]
            metadata = self.current_model_metadata
            model_name = self.model_name
            _create_model_plots_for_single_model(self.model_name)
        else:
            for model_name in list_of_models:
                model_session_locations = self.create_session(model_name=model_name)
                output_dir = model_session_locations["model_run_folder"]
                metadata = self.all_model_metadata.get("model", {}).get(model_name, {}) if isinstance(self.all_model_metadata, dict) else {}
                _create_model_plots_for_single_model(model_name)
                del model_session_locations  # Free memory


    def all_models_plots(self, session_locations, timestamp):
        """
        Tworzy wykresy porównujące WSZYSTKIE modele dla każdej optymalizacji.
        Analogiczne do all_models_plots w referenced.
        """
        from collections import defaultdict
        all_models_comparison_dict = {
            "parameters": self.MULTI_TURN_GLOBAL_CONFIG, 
            "tools": self.tools, 
            "cot_prompt": self.cot_prompt
        }
        group_by_keys = ['optimisation', 'model_name']
        # Get ALL models for each optimization (not just latest session)
        valid_session_data = self.get_last_sessions(key_dict=all_models_comparison_dict, log_file=session_locations["log_file"], group_by_keys=group_by_keys)
        # filter optimisations
        output_dir = session_locations["all_models_output_directory"]
        # Generate a single timestamp for all files in this run
        metadata=self.all_model_metadata
        print("DEBUG:  valid_session_data")
        grouped_session_by_optim = defaultdict(list)
        for optim_model, sessiondata in valid_session_data.items():
            grouped_session_by_optim[optim_model[0]].append(sessiondata)

        for optimisation, session_data in grouped_session_by_optim.items():
            # Generate individual optimization plots

            # 1. Generate test pass rate comparison
            test_pass_chart_path = self.plot_test_pass_rates(
                session_data=session_data,
                optimisation_type=optimisation,
                agent_type=self.agent_type,
                plotting_session_timestamp=timestamp,
                metadata=metadata,
                output_dir=output_dir,
                output_file_name=f"{optimisation}_test_pass_rates_{timestamp}",
            )
            print(f"Test pass rates chart saved to: {test_pass_chart_path}")

            # 2. Generate patient completion analysis
            completion_chart_path = self.plot_patient_completion_analysis(
                session_data=session_data,
                optimisation_type=optimisation,
                agent_type=self.agent_type,
                plotting_session_timestamp=timestamp,
                metadata=metadata,
                output_dir=output_dir,
                output_file_name=f"{optimisation}_completion_analysis_{timestamp}",
            )
            print(f"Completion analysis chart saved to: {completion_chart_path}")

            # 3. Generate model comparison plot
            comparison_plot_path = self.plot_aggr_all_models_with_unreferenced(
                session_data=session_data,
                optimisation_type=optimisation,
                agent_type=self.agent_type,
                parameters=self.MULTI_TURN_GLOBAL_CONFIG,
                plotting_session_timestamp=timestamp,
                metadata=metadata,
                output_dir=output_dir,
                output_file_name=f"{optimisation}_models_comparison_{timestamp}",
            )
            print(f"All models comparison plot saved to: {comparison_plot_path}")


    def plot_per_patient_summary(self, patient_result: Dict, output_dir: str, timestamp: str):
        """Tworzy wykres podsumowujący dla pojedynczego pacjenta"""

        patient_name = patient_result["patient_name"]
        plot_path = os.path.join(output_dir, f"patient_{patient_name}_{timestamp}.png")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Patient: {patient_name} - {patient_result['test_result']}", fontsize=14, fontweight='bold')

        # 1. Conversation flow
        ax = axes[0, 0]
        rounds_data = patient_result.get("rounds_data", [])
        round_nums = [r["round"] for r in rounds_data]
        statuses = [r["status"] for r in rounds_data]

        ax.plot(round_nums, [1 if s == "COMPLETE" else 0 for s in statuses], 'o-', color='green')
        ax.set_title("Conversation Progress")
        ax.set_xlabel("Round")
        ax.set_ylabel("Status")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["INCOMPLETE", "COMPLETE"])
        ax.grid(True, alpha=0.3)

        # 2. Data collection progress
        ax = axes[0, 1]
        collected_fields = list(patient_result.get("collected_data", {}).keys())
        expected_fields = list(patient_result.get("expected_data", {}).get("expected_data", {}).keys())

        all_fields = sorted(set(collected_fields + expected_fields))
        collected_status = [1 if f in collected_fields else 0 for f in all_fields]

        colors = ['green' if s else 'red' for s in collected_status]
        ax.barh(all_fields, [1] * len(all_fields), color=colors, alpha=0.3)
        ax.barh(all_fields, collected_status, color=colors)
        ax.set_title("Field Collection Status")
        ax.set_xlabel("Collected")
        ax.set_xlim(0, 1)

        # 3. Latency per round
        ax = axes[1, 0]
        latencies = [r.get("latency_breakdown", {}).get("total_ms") for r in rounds_data]
        latencies = [value if isinstance(value, (int, float)) else float("nan") for value in latencies]
        if latencies:
            ax.bar(round_nums, latencies, color='blue', alpha=0.7)
            ax.set_title("Latency per Round")
            ax.set_xlabel("Round")
            ax.set_ylabel("Latency (ms)")
            ax.grid(True, alpha=0.3)

        # 4. Test result summary
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
        Patient Type: {patient_result['patient_type']}
        Test Result: {patient_result['test_result']}
        Status: {patient_result['status']}
        Rounds: {patient_result['rounds_count']}
        Duration: {patient_result['total_duration_sec']:.2f}s
        
        Collected Data:
        {json.dumps(patient_result['collected_data'], indent=2)[:300]}...
        """

        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"   Saved patient plot: {plot_path}")


    def plot_model_summary(self, session_data, output_dir: str, timestamp: str):
        """Tworzy wykres podsumowujący dla całego modelu"""

        plot_path = os.path.join(output_dir, f"model_summary_{timestamp}.png")
        metrics = session_data.get("aggregated_metrics", {})

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Model Summary: {session_data['model_name']}", fontsize=16, fontweight='bold')

        # 1. Test results distribution
        ax = axes[0, 0]
        successful = metrics.get("successful_patients", {})
        unsuccessful = metrics.get("unsuccessful_patients", {})

        categories = ['Perfect\nSuccess', 'Partial\nSuccess', 'False\nComplete', 'Timeout', 
                     'Correct\nRejection', 'False\nSuccess']
        values = [
            successful.get("perfect_success", 0),
            successful.get("partial_success", 0),
            successful.get("false_complete", 0),
            successful.get("timeout", 0),
            unsuccessful.get("correct_rejection", 0),
            unsuccessful.get("false_success", 0)
        ]

        colors = ['green', 'lightgreen', 'orange', 'red', 'blue', 'darkred']
        ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title("Test Results Distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)

        # 2. Field collection rates
        ax = axes[0, 1]
        field_rates = successful.get("field_collection_rates", {})
        if field_rates:
            fields = list(field_rates.keys())
            rates = list(field_rates.values())
            ax.bar(fields, rates, color='teal', alpha=0.7)
            ax.set_title("Field Collection Success Rates")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

            for i, v in enumerate(rates):
                ax.text(i, v + 0.02, f"{v:.0%}", ha='center', fontweight='bold')

        # 3. Success rates comparison
        ax = axes[0, 2]
        success_data = {
            'Successful\nPatients': successful.get("success_rate", 0),
            'Unsuccessful\nPatients': unsuccessful.get("success_rate", 0),
            'Overall\nTest Pass': metrics.get("overall", {}).get("test_pass_rate", 0)
        }

        ax.bar(success_data.keys(), success_data.values(), color=['green', 'blue', 'purple'], alpha=0.7)
        ax.set_title("Success Rates")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1)

        for i, (k, v) in enumerate(success_data.items()):
            ax.text(i, v + 0.02, f"{v:.0%}", ha='center', fontweight='bold')

        # 4. Latency distribution
        ax = axes[1, 0]
        overall = metrics.get("overall", {})
        latency_data = {
            'Avg': overall.get("avg_latency_per_round_ms", 0),
            'Min': overall.get("min_latency_ms", 0),
            'Max': overall.get("max_latency_ms", 0)
        }

        ax.bar(latency_data.keys(), latency_data.values(), color='coral', alpha=0.7)
        ax.set_title("Latency Statistics (ms)")
        ax.set_ylabel("Latency (ms)")

        # 5. Rounds to completion
        ax = axes[1, 1]
        patients = session_data.get("patients", [])
        rounds_data = [p["rounds_count"] for p in patients if p["patient_type"] == "successful"]

        if rounds_data:
            ax.hist(rounds_data, bins=range(1, max(rounds_data) + 2), color='skyblue', alpha=0.7, edgecolor='black')
            ax.set_title("Rounds to Completion Distribution")
            ax.set_xlabel("Number of Rounds")
            ax.set_ylabel("Count")
            ax.axvline(np.mean(rounds_data), color='red', linestyle='--', label=f'Mean: {np.mean(rounds_data):.1f}')
            ax.legend()

        # 6. Summary statistics
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = f"""
        Overall Test Pass Rate: {metrics.get('overall', {}).get('test_pass_rate', 0):.1%}
        
        Successful Patients ({successful.get('total', 0)}):
        - Success Rate: {successful.get('success_rate', 0):.1%}
        - Avg Rounds: {successful.get('avg_rounds_to_complete', 0):.1f}
        - Avg Time: {successful.get('avg_completion_time_sec', 0):.1f}s
        
        Unsuccessful Patients ({unsuccessful.get('total', 0)}):
        - Success Rate: {unsuccessful.get('success_rate', 0):.1%}
        - Avg Rounds: {unsuccessful.get('avg_rounds_to_reject', 0):.1f}
        
        Latency:
        - Avg: {overall.get('avg_latency_per_round_ms', 0):.0f}ms
        - Min: {overall.get('min_latency_ms', 0):.0f}ms
        - Max: {overall.get('max_latency_ms', 0):.0f}ms
        """

        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f" Saved model summary plot: {plot_path}")


    @staticmethod
    def plot_test_pass_rates(session_data, optimisation_type, agent_type, plotting_session_timestamp, metadata, output_dir, output_file_name):
        """
        Porównuje test pass rates między modelami.
        Analogiczne do plot_latency_performance w referenced.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Extract data for multiple models
        models = []
        pass_rates = []

        if isinstance(session_data, list):
            for model_session in session_data:
                model_name = model_session.get('model_name', 'unknown_model')
                models.append(model_name.replace(":", "_"))

                # Calculate pass rate
                patients = model_session.get('patients', [])
                if patients:
                    passed = len([p for p in patients if p.get('test_result') in ['perfect_success', 'partial_success', 'correct_rejection']])
                    pass_rate = passed / len(patients) * 100
                    pass_rates.append(pass_rate)
                else:
                    pass_rates.append(0)

        if not models:
            return None

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f'Test Pass Rates Comparison\nUnreferenced Evaluation - {plotting_session_timestamp}', 
                    fontsize=16, fontweight='bold')

        bars = ax.bar(models, pass_rates, color='green', alpha=0.7)

        # Add value labels
        for bar, rate in zip(bars, pass_rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Pass Rate (%)')
        ax.set_title('Test Pass Rate by Model')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save plot
        plot_filename = f"{output_file_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path


    @staticmethod
    def plot_patient_completion_analysis(session_data, optimisation_type, agent_type, plotting_session_timestamp, metadata, output_dir, output_file_name):
        """
        Analizuje completion rates i czas dla pacjentów.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Similar structure to plot_test_pass_rates but focused on completion metrics
        models = []
        avg_rounds = []
        completion_times = []

        if isinstance(session_data, list):
            for model_session in session_data:
                model_name = model_session.get('model_name', 'unknown_model')
                models.append(model_name.replace(":", "_"))

                patients = model_session.get('patients', [])
                completed_patients = [p for p in patients if p.get('completed_successfully')]

                if completed_patients:
                    avg_rounds.append(np.mean([p.get('rounds_count', 0) for p in completed_patients]))
                    completion_times.append(np.mean([p.get('total_duration_sec', 0) for p in completed_patients]))
                else:
                    avg_rounds.append(0)
                    completion_times.append(0)

        if not models:
            return None

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Patient Completion Analysis\nUnreferenced Evaluation - {plotting_session_timestamp}', 
                    fontsize=16, fontweight='bold')

        # Rounds to completion
        bars1 = ax1.bar(models, avg_rounds, color='blue', alpha=0.7)
        ax1.set_title('Average Rounds to Completion')
        ax1.set_ylabel('Rounds')
        ax1.tick_params(axis='x', rotation=45)

        for bar, rounds in zip(bars1, avg_rounds):
            if rounds > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{rounds:.1f}', ha='center', va='bottom', fontweight='bold')

        # Completion time
        bars2 = ax2.bar(models, completion_times, color='orange', alpha=0.7)
        ax2.set_title('Average Completion Time')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)

        for bar, time in zip(bars2, completion_times):
            if time > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(completion_times)*0.02,
                        f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Save plot
        plot_filename = f"{output_file_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path


    @staticmethod
    def plot_aggr_all_models_with_unreferenced(session_data, optimisation_type, agent_type, parameters, plotting_session_timestamp, metadata, output_dir, output_file_name):
        """
        Porównuje wszystkie modele w unreferenced evaluation.
        Analogiczne do plot_aggr_all_models_with_reference w referenced.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Extract models data
        models_data = {}

        if isinstance(session_data, list):
            for model_session in session_data:
                model_name = model_session.get('model_name', 'unknown_model')
                patients = model_session.get('patients', [])

                if not patients:
                    continue

                # Calculate metrics
                total_patients = len(patients)
                passed = len([p for p in patients if p.get('test_result') in ['perfect_success', 'partial_success', 'correct_rejection']])
                pass_rate = passed / total_patients * 100 if total_patients > 0 else 0

                # Average rounds and time
                completed = [p for p in patients if p.get('completed_successfully')]
                avg_rounds = np.mean([p.get('rounds_count', 0) for p in completed]) if completed else 0
                avg_time = np.mean([p.get('total_duration_sec', 0) for p in completed]) if completed else 0

                models_data[model_name] = {
                    'pass_rate': pass_rate,
                    'avg_rounds': avg_rounds,
                    'avg_time': avg_time,
                    'total_patients': total_patients
                }

        if not models_data:
            return None

        # Create comprehensive comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('All Models Comparison - Unreferenced Evaluation', fontsize=16, fontweight='bold')

        model_names = list(models_data.keys())

        # 1. Pass rates comparison
        pass_rates = [models_data[m]['pass_rate'] for m in model_names]
        bars1 = ax1.bar(model_names, pass_rates, color='green', alpha=0.7)
        ax1.set_title('Test Pass Rates')
        ax1.set_ylabel('Pass Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)

        for bar, rate in zip(bars1, pass_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 2. Average rounds
        avg_rounds_list = [models_data[m]['avg_rounds'] for m in model_names]
        ax2.bar(model_names, avg_rounds_list, color='blue', alpha=0.7)
        ax2.set_title('Average Rounds to Completion')
        ax2.set_ylabel('Rounds')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Average time
        avg_times = [models_data[m]['avg_time'] for m in model_names]
        ax3.bar(model_names, avg_times, color='orange', alpha=0.7)
        ax3.set_title('Average Completion Time')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Summary table
        ax4.axis('off')
        summary_text = "MODEL COMPARISON SUMMARY\n\n"
        for model_name in model_names:
            data = models_data[model_name]
            summary_text += f"{model_name}:\n"
            summary_text += f"  Pass Rate: {data['pass_rate']:.1f}%\n"
            summary_text += f"  Avg Rounds: {data['avg_rounds']:.1f}\n"
            summary_text += f"  Avg Time: {data['avg_time']:.1f}s\n"
            summary_text += f"  Total Patients: {data['total_patients']}\n\n"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        # Save plot
        plot_filename = f"{output_file_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path
