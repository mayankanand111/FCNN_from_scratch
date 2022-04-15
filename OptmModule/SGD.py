import numpy as np
import OptmModule.Optimizer as Optimizer
import AllLayers.BatchNorm as BatchNorm
import AllLayers.AffineLayer as AffineLayer

class SGD(Optimizer.Optimizer):
    def __init__(self, momentum):
        self.learning_rate = 0.001
        self.momentum = momentum

    def step(self, layer):
      if isinstance(layer,AffineLayer.AffineLayer):
        w_velocity = self.momentum*layer.w_velocity - self.learning_rate*layer.grads[0]
        b_velocity = self.momentum*layer.b_velocity - self.learning_rate*layer.grads[1]
        layer.parameters[0] +=  w_velocity
        layer.parameters[1] +=  b_velocity
        layer.w_velocity = w_velocity
        layer.b_velocity = b_velocity
      elif isinstance(layer,BatchNorm.BatchNorm):
        layer.parameters[0] -= self.learning_rate*layer.grads[0]
        layer.parameters[1] -= self.learning_rate*layer.grads[1]

    def __call__(self,layer):
        return self.step(layer)