import numpy as np
import OptmModule.Optimizer as Optimizer
import AllLayers.BatchNorm as BatchNorm
import AllLayers.AffineLayer as AffineLayer


class Adam(Optimizer.Optimizer):
    def __init__(self, learning_rate, beta1, beta2, epsilon):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def step(self, layer):
        if isinstance(layer, AffineLayer):
            layer.itr += 1
            m_dw = self.beta1 * layer.m_dw + (1 - self.beta1) * layer.grads[0]
            m_db = self.beta1 * layer.m_db + (1 - self.beta1) * layer.grads[1]

            v_dw = self.beta2 * layer.v_dw + (1 - self.beta2) * (layer.grads[0] * layer.grads[0])
            v_db = self.beta2 * layer.v_db + (1 - self.beta2) * (layer.grads[1] * layer.grads[1])

            m_dw_corrected = m_dw / (1 - (self.beta1 ** layer.itr))
            m_db_corrected = m_db / (1 - (self.beta1 ** layer.itr))
            v_dw_corrected = v_dw / (1 - (self.beta2 ** layer.itr))
            v_db_corrected = v_db / (1 - (self.beta2 ** layer.itr))

            layer.parameters[0] = layer.parameters[0] - self.learning_rate * m_dw_corrected / (
                        np.sqrt(v_dw_corrected) + self.epsilon)
            layer.parameters[1] = layer.parameters[1] - self.learning_rate * m_db_corrected / (
                        np.sqrt(v_db_corrected) + self.epsilon)
            layer.m_dw = m_dw
            layer.m_db = m_db
            layer.v_dw = v_dw
            layer.v_db = v_db
        elif isinstance(layer, BatchNorm):
            layer.parameters[0] -= self.learning_rate * layer.grads[0]
            layer.parameters[1] -= self.learning_rate * layer.grads[1]

    def __call__(self, layer):
        return self.step(layer)