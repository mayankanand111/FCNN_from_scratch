import numpy as np

class GlobalFunctions:

    def __init__(self):
        pass
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, 10))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def get_predictions(A2):
        return np.argmax(A2, 1)

    def get_accuracy(predictions, Y):
        return np.mean(GlobalFunctions.get_predictions(predictions) == GlobalFunctions.get_predictions(Y))