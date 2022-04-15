from AllActivations import Activations
import numpy as np

class Sigmoid(Activations.Activations):
    def forward(self, x):
        return 1 / (1 + (np.exp(-x)))

    def backward(self, grad, original_input):
        x = original_input
        return grad * (self.forward(x) * (1 - self.forward(x)))

    def __call__(self, x, mode=None):
        return self.forward(x)