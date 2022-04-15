import numpy as np
from AllLayers import Layers


class Dropout(Layers.Layers):
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, x, mode):
        out = None
        self.mode = mode
        if mode == 'train':
            self.mask = (np.random.rand(*x.shape) < (1 - self.p)) / (1 - self.p)
            out = x * self.mask
        elif mode == 'test':
            out = x
        return out

    def backward(self, grad, original_input):
        grad_x = None
        grad_x = grad * self.mask
        # return gradient with respect to the original input to this layer
        return grad_x

    def __call__(self, x, mode="test"):
        return self.forward(x, mode)