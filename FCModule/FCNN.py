from AllActivations import SoftMaxCrossLoss

class FCNN():
    def __init__(self, layers):
        self.layers = layers
        self.original_inputs = []
        self.loss_list = []
        self.accuracy_List = []

    def forward(self, x, y=None, mode='test'):
        # Start by adding the input to the original_inputs array that keeps the original_input to each
        # layer in order.
        self.original_inputs = [x.copy()]
        for index, layer in enumerate(self.layers):
            if isinstance(layer, SoftMaxCrossLoss.SoftMaxCrossLoss):
                out = layer(self.original_inputs[index], y, mode)
                if mode == 'train':
                    self.loss_list.append(out)
            else:
                out = layer(self.original_inputs[index], mode)
            self.original_inputs.append(out)
        return self.original_inputs[-1]

    def backward(self):
        # Pop loss value from the cache
        self.original_inputs.pop()
        grad = 1
        for index, layer in reversed(list(enumerate(self.layers))):
            grad = layer.backward(grad, self.original_inputs.pop(index))
        return grad

    def __call__(self, x, y=None, mode='test'):
        return self.forward(x, y, mode)
