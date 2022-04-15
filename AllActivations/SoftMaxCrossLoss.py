from AllActivations import Activations
import numpy as np

class SoftMaxCrossLoss(Activations.Activations):
    def __init__(self):
        self.probs = None
        self.y = None

    def forward(self, x, y=None):
        # y being none should make this return simply a softmax activation: we are in testing mode
        # y is not none, you should return the cross-entropy loss using the softmax activations
        if y is None:
            exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
            self.probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            return self.probs
        else:
            self.y = y
            exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
            x_softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            loss = -np.sum(self.y * np.log(x_softmax)) / x.shape[0]
            return loss

    def backward(self, grad, original_input):
        return (grad * (original_input - self.y))

    def __call__(self, x, y=None, mode=None):
        return self.forward(x, y)