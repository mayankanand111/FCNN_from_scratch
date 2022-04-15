import numpy as np
import GlobalFunctions as gb


class LoadData:
    def __init__(self):
        self

    def Load_Data(self):
        train_data = np.load("train_data.npy")
        train_targets = np.load("train_targets.npy")
        test_data = np.load("test_data.npy")
        test_targets = np.load("test_targets.npy")
        train_targets = (gb.one_hot(train_targets)).T
        test_targets = (gb.one_hot(test_targets)).T
        image_vector_size = 28 * 28
        train_data = train_data.reshape(train_data.shape[0], image_vector_size)
        test_data = test_data.reshape(test_data.shape[0], image_vector_size)
        return train_data, train_targets, test_data, test_targets

    def __call__(self, ):
        return self.Load_Data()
