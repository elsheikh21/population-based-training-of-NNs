from abc import ABC, abstractmethod

from keras import backend as K
from keras.layers import Dropout
import numpy as np


class Hyperparameter(ABC):

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def set(self, value):
        pass

    @abstractmethod
    def get_config(self):
        pass


class FloatHyperparameter(Hyperparameter):

    def __init__(self, name, variable):
        self.name = name
        self.variable = variable

    def get(self):
        return K.get_value(self.variable)

    def set(self, value):
        K.set_value(self.variable, K.cast_to_floatx(value))

    def get_config(self):
        return {self.name: self.get()}


class FloatExpHyperparameter(FloatHyperparameter):

    def __init__(self, name, variable):
        super().__init__(name, variable)

    def get(self):
        return np.log10(K.get_value(self.variable))

    def set(self, value):
        K.set_value(self.variable, K.cast_to_floatx(np.power(10, value)))


class DropoutHP(Hyperparameter, Dropout):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)

    def get(self):
        return self.rate

    def set(self, rate):
        self.rate = min(1., max(0., rate))

    def get_config(self):
        return {'dr': self.rate}


def find_hyperparameters_model(keras_model):
    hyperparameters = []
    for layer in keras_model.layers:
        if isinstance(layer, Hyperparameter):
            hyperparameters.append(layer)
        else:
            hyperparameters.extend(find_hyperparameters_layer(layer))
    return hyperparameters


def find_hyperparameters_layer(keras_layer):
    hyperparameters_names = ['kernel_regularizer']
    hyperparameters = []
    for h_name in hyperparameters_names:
        if hasattr(keras_layer, h_name):
            h = getattr(keras_layer, h_name)
            if isinstance(h, Hyperparameter):
                hyperparameters.append(h)
    return hyperparameters
