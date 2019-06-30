from collections import deque

import numpy as np

from hyperparameters import find_hyperparameters_model, FloatExpHyperparameter
from train_utils import batch_generator


class Member:

    def __init__(self, model, param_names, tune_lr=False, use_eval_metric='loss', custom_metrics={}):
        self.model = model
        self.steps = 0

        self.loss = np.Inf
        self.metrics = {metric: np.nan for metric in self.model.metrics + list(custom_metrics.keys())}

        self.use_eval_metric = use_eval_metric
        self.custom_metrics = custom_metrics

        self.recent_eval = deque(maxlen=10)

        self.hyperparameters = []
        if tune_lr:
            lr = FloatExpHyperparameter('lr', self.model.optimizer.lr)
            self.hyperparameters.append(lr)
        self.hyperparameters += find_hyperparameters_model(self.model)
        if not self.hyperparameters:
            raise ValueError('The model has no hyperparameters to tune')

        self.hyperparameter_config = {n: h for n, h in zip(param_names, self.hyperparameters)}

    def eval_metric(self):
        return self.metrics.get(self.use_eval_metric, self.loss)

    def eval_metric_mean(self):
        return np.mean(self.recent_eval)

    def step_on_batch(self, x, y):
        self.steps += 1

        scalars = self.model.train_on_batch(x, y)
        train_loss = scalars if not isinstance(scalars, list) else scalars[0]
        return train_loss

    def eval_on_batch(self, x, y):
        scalars = self.model.test_on_batch(x, y)
        eval_loss = scalars if not isinstance(scalars, list) else scalars[0]
        metric_values = [] if not isinstance(scalars, list) else scalars[1:]
        return eval_loss, metric_values

    def eval(self, x, y, generator_fn=batch_generator):
        values = [self.eval_on_batch(bx, by) for bx, by in generator_fn(x, y, shuffle=False, looping=False)]
        loss = np.mean([v[0] for v in values])
        metrics = np.mean([v[1] for v in values], axis=0).tolist()

        self.loss = loss

        for metric, value in zip(self.model.metrics, metrics):
            self.metrics[metric] = value

        for custom_metric, fn in self.custom_metrics.items():
            self.metrics[custom_metric] = fn(x, y, generator_fn)

        eval_metric = self.eval_metric()
        self.recent_eval.append(eval_metric)
        return eval_metric

    def replace_with(self, member):
        assert len(self.hyperparameters) == len(member.hyperparameters), \
            'Members do not belong to the same population!'
        self.model.set_weights(member.model.get_weights())
        for i, hyperparameter in enumerate(self.hyperparameters):
            hyperparameter.set(member.hyperparameters[i].get())

    def get_hyperparameter_config(self):
        return {n: h.get() for n, h in self.hyperparameter_config.items()}

    def __str__(self):
        return str(id(self))
