import numpy as np
from GlobalFunctions import GlobalFunctions
from OptmModule import SGD
from OptmModule import Adam
from AllLayers import AffineLayer
from AllLayers import BatchNorm
import matplotlib.pyplot as plt

class TrainingLoop:

    def MiniBatchSDG(self,epochs, size, train_data, train_targets, net, optimizer,test_data,test_targets):
        iteractions = train_data.shape[0] / size
        index = np.linspace(train_data.shape[0], size)
        batch_list = np.linspace(0, train_data.shape[0], num=int(train_data.shape[0] / size), dtype=int)
        batches_data = []
        batches_target = []
        optimizer = SGD.SGD(0.9) if (optimizer == 'SGD') else Adam.Adam(learning_rate=0.001, beta1=0.9, beta2=0.9, epsilon=1e-8)
        for x in batch_list:
            if x == 0:
                previous_x = x
                continue
            else:
                batches_data.append(train_data[previous_x:x])
                batches_target.append(train_targets[previous_x:x])
                previous_x = x
        for epoch in range(epochs):
            counter = 0
            for Btrain_data, Btrain_targets in zip(batches_data, batches_target):
                counter += 1
                net(Btrain_data, Btrain_targets, 'train')
                net.backward()
                for layer in net.layers:
                    if isinstance(layer, AffineLayer.AffineLayer) or isinstance(layer, BatchNorm.BatchNorm):
                        optimizer(layer)
            batch_test = net(train_data)
            print("************************", type(optimizer).__name__,
                  "****************************Accuracy************************", type(optimizer).__name__,
                  "****************************")
            net.accuracy_List.append(GlobalFunctions.get_accuracy(GlobalFunctions.one_hot(GlobalFunctions.get_predictions(batch_test)).T, train_targets))
            print("Accuaracy after", epoch + 1, " epochs : ", GlobalFunctions.get_accuracy(train_targets, batch_test) * 100)
        print("----------------------------------------Running Test Set---------------------------------------------")
        test_pred = net(test_data)
        print("Test Set Accuaracy is : ", GlobalFunctions.get_accuracy(test_targets, test_pred) * 100)
        print("----------------------------------------Running Test Set---------------------------------------------")
        plt.plot(range(epochs * len(batches_data)), net.loss_list, 'blue')
        plt.title('Loss decay')
        plt.xlabel('number of training batches seen')
        plt.ylabel('Loss')
        plt.show()

    def __call__(self):
        return self.MiniBatchSDG()