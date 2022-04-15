import numpy as np
from AllActivations import Activations

class ReLU(Activations.Activations):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, grad, original_input):
        # this assumes that the original input to this layer has been saved somewhere else
        x = original_input
        return grad * (x > 0)