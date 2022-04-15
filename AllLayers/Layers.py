import abc

class Layers(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def backward(self, grad, original_input):
        return