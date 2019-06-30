from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from keras.utils import Progbar
from sklearn.model_selection import ParameterGrid

from train_utils import batch_generator


class PbtOptimizer:

    def __init__(self, build_member: Callable,
                 population_size: int,
                 parameters: dict,
                 steps_ready: int):
        self.population_size = population_size
        parameters = ParameterGrid(parameters)
        self.population = [build_member(**parameters[p])
                           for p in np.random.choice(len(parameters), size=population_size, replace=False)]

        self.steps_ready = steps_ready

    def train(self, x_train, y_train, x_val, y_val, steps, eval_every,
              generator_fn=batch_generator):
        results = defaultdict(lambda: [])

        metrics, values = self._init_progress()
        progbar = Progbar(steps, stateful_metrics=metrics)
        progbar.update(0, values)
        train_gen = generator_fn(x_train, y_train)
        for step in range(1, steps + 1):
            x, y = next(train_gen)
            for member in self.population:
                member.step_on_batch(x, y)
                if step % eval_every == 0 or step == steps:
                    member.eval(x_val, y_val, generator_fn)
                    self.on_eval(member)

            for member in self.population:
                if self.ready(member):
                    exploited = self.exploit(member)
                    if exploited:
                        self.explore(member)
                        member.eval(x_val, y_val, generator_fn)
                        self.on_eval(member)

                if step % eval_every == 0 or step == steps:
                    results[str(member)].append(self._collect_result(member))

            if step % eval_every == 0 or step == steps:
                values = self._update_progress(results, metrics)
                progbar.update(step, values)

        results = pd.concat([pd.DataFrame(v).assign(member=k)
                             for k, v in results.items()])
        best_model = self.population[np.array([m.eval_metric_mean() for m in self.population]).argmax()].model

        return best_model, results

    def ready(self, member):
        return member.steps % self.steps_ready == 0

    def exploit(self, member):
        evals = np.array([m.eval_metric_mean() for m in self.population])
        threshold_worst, threshold_best = np.percentile(evals, (20, 80))
        if member.eval_metric_mean() < threshold_worst:
            top_performers = [m for m in self.population
                              if m.eval_metric_mean() > threshold_best]
            if top_performers:
                member.replace_with(np.random.choice(top_performers))
            return True
        else:
            return False

    def explore(self, member):
        for h in member.hyperparameters:
            h.set(h.get() * np.random.choice([0.8, 1.2]))

    def on_eval(self, member):
        pass

    def _collect_result(self, member):
        result = {
            'step': member.steps,
            'loss': member.loss
        }
        for metric, value in member.metrics.items():
            result[metric] = value
        for h, v in member.get_hyperparameter_config().items():
            result[h] = v
        return result

    def _init_progress(self):
        metrics = ['loss'] + list(self.population[0].metrics.keys())
        values = [(metric, np.nan) for metric in metrics]
        return metrics, values

    def _update_progress(self, results, metrics):
        last_metrics = defaultdict(lambda: [])
        for model, model_results in results.items():
            for metric in metrics:
                last_metrics[metric].append(model_results[-1][metric])
        return [(metric, np.mean(values)) for metric, values in last_metrics.items()]
