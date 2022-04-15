import abc

class Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self,layer):
        return
