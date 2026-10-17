"""Reproducible parameter search with explicit optimization direction and split."""
import math
from typing import Any, Callable, Dict

import optuna


class OptimizationEngine:
    def __init__(self, study_name='llm_optimization', *, direction='maximize',
                 storage=None, seed=42, load_if_exists=False, dataset_split='dev'):
        if dataset_split != 'dev':
            raise ValueError('Tune on dev data; reserve test data for final evaluation')
        if direction not in {'minimize', 'maximize'}:
            raise ValueError('direction must be minimize or maximize')
        self.study_name = study_name
        self.study = optuna.create_study(
            study_name=study_name, direction=direction, storage=storage,
            sampler=optuna.samplers.TPESampler(seed=seed), load_if_exists=load_if_exists)
        if self.study.direction.name.lower() != direction:
            raise ValueError('Stored study direction differs from requested direction')
        self.study.set_user_attr('dataset_split', dataset_split)

    def run_optimization(self, objective_func: Callable, n_trials=20):
        if type(n_trials) is not int or n_trials < 1:
            raise ValueError('n_trials must be a positive integer')
        def checked(trial):
            value = float(objective_func(trial))
            if not math.isfinite(value):
                raise ValueError('Objective must be finite; missing measurements are not zero')
            return value
        self.study.optimize(checked, n_trials=n_trials)
        return self.study.best_trial.params

    @staticmethod
    def suggest_parameters(trial, search_space: Dict[str, Any]):
        params = {}
        for name, spec in search_space.items():
            if 'choices' in spec:
                if not spec['choices']:
                    raise ValueError(f'Empty choices for {name}')
                params[name] = trial.suggest_categorical(name, spec['choices'])
            elif {'low', 'high'} <= spec.keys():
                kind = spec.get('type', 'float' if any(isinstance(spec[k], float) for k in ('low', 'high')) else 'int')
                if kind not in {'int', 'float'}:
                    raise ValueError(f'Unknown parameter type: {kind}')
                kwargs = {key: spec[key] for key in ('step', 'log') if key in spec}
                suggest = trial.suggest_float if kind == 'float' else trial.suggest_int
                params[name] = suggest(name, spec['low'], spec['high'], **kwargs)
            else:
                raise ValueError(f'Invalid search space for {name}')
        return params
