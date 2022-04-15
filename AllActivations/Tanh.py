from AllActivations import Activations
import numpy as np

class Tanh(Activations.Activations):
    def forward(self, x):
        return ((np.exp(x)) - (np.exp(-x))) / ((np.exp(x)) + (np.exp(-x)))

    def backward(self, grad, original_input):
        x = original_input
        return (grad * (1 - (self.forward(x) ** 2)))

    def __call__(self, x, mode=None):
        return self.forward(x)