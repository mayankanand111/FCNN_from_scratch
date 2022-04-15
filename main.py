# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from FCModule.FCNN import FCNN
from AllLayers.AffineLayer import AffineLayer
from AllLayers.BatchNorm import BatchNorm
from AllLayers.Dropout import Dropout
from AllActivations.Sigmoid import Sigmoid
from AllActivations.ReLU import ReLU
from AllActivations.SoftMaxCrossLoss import SoftMaxCrossLoss
from AllActivations.Tanh import  Tanh

from TrainingLoop import TrainingLoop
import numpy as np
from GlobalFunctions import GlobalFunctions
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data = np.load("train_data.npy")
    train_targets = np.load("train_targets.npy")
    test_data = np.load("test_data.npy")
    test_targets = np.load("test_targets.npy")
    train_targets = (GlobalFunctions.one_hot(train_targets)).T
    test_targets = (GlobalFunctions.one_hot(test_targets)).T
    image_vector_size = 28 * 28
    train_data = train_data.reshape(train_data.shape[0], image_vector_size)
    test_data = test_data.reshape(test_data.shape[0], image_vector_size)
    Best_Model_Sigmoid = FCNN([AffineLayer(784, 128), BatchNorm(128, 1e-5, 0.9), Dropout(0.5), Sigmoid(),
                               AffineLayer(128, 64), BatchNorm(64, 1e-5, 0.9), Dropout(0.3), Sigmoid(),
                               AffineLayer(64, 10),
                               SoftMaxCrossLoss()])
    TrainingLoop.MiniBatchSDG(TrainingLoop(),1, 100, train_data, train_targets, Best_Model_Sigmoid, 'SGD', test_data, test_targets)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
