from AllLayers import Layers
import numpy as np


class AffineLayer(Layers.Layers):
    def __init__(self, input_dim, hidden_units):
        ran = np.sqrt(1 / input_dim)

        # Initializing the weights and bias, in that order
        self.parameters = [np.random.uniform(-ran, ran, (hidden_units, input_dim)),
                           np.random.uniform(-ran, ran, hidden_units)]

        self.grads = [np.zeros_like(self.parameters[0]), np.zeros_like(self.parameters[1])]

        self.w_momentum_list = [0]
        self.b_momentum_list = [0]
        self.w_velocity = np.zeros_like(self.parameters[0])
        self.b_velocity = np.zeros_like(self.parameters[1])

        self.m_dw = np.zeros_like(self.parameters[0])
        self.m_db = np.zeros_like(self.parameters[1])
        self.v_dw = np.zeros_like(self.parameters[0])
        self.v_db = np.zeros_like(self.parameters[1])
        self.itr = 0

    def forward(self, x):
        out = np.dot(x, self.parameters[0].T) + self.parameters[1]
        return out

    def backward(self, grad, original_input):
        w = self.parameters[0]
        b = self.parameters[1]
        self.grads[0] = np.dot(grad.T, original_input)
        self.grads[1] = np.sum(grad, axis=0)
        grad_x = np.dot(grad, w)
        return grad_x

    def __call__(self, x, mode=None):
        return self.forward(x)
