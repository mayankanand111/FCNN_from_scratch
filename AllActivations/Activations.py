import abc
import numpy as np

class Activations(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self,x):
        return

    @abc.abstractmethod
    def backward(self, grad,original_input):
        return