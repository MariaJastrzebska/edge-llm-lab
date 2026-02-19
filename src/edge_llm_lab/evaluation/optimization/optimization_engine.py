import optuna
from typing import Dict, Any, Callable, List
import yaml
import os

class OptimizationEngine:
    """Engine for running Optuna-based parameter optimization."""
    
    def __init__(self, study_name: str = "llm_optimization"):
        self.study_name = study_name
        self.study = optuna.create_study(study_name=study_name, direction="maximize")

    def run_optimization(self, objective_func: Callable[[optuna.Trial], float], n_trials: int = 20):
        """Runs the Optuna study."""
        print(f"🚀 Starting Optuna optimization: {self.study_name}")
        self.study.optimize(objective_func, n_trials=n_trials)
        print(" Optimization finished.")
        print(f"Best trial: {self.study.best_trial.params}")
        return self.study.best_trial.params

    @staticmethod
    def suggest_parameters(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggests parameters based on the provided search space."""
        params = {}
        for param_name, config in search_space.items():
            if "choices" in config:
                params[param_name] = trial.suggest_categorical(param_name, config["choices"])
            elif "low" in config and "high" in config:
                if isinstance(config["low"], float):
                    params[param_name] = trial.suggest_float(param_name, config["low"], config["high"])
                else:
                    params[param_name] = trial.suggest_int(param_name, config["low"], config["high"])
        return params
