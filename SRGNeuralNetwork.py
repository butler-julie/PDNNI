from NeuralNetwork import Trainer
from NeuralNetwork import Restore
# FOR DEBUGGING
# Eigenvalue, eigenvector solver
from scipy.linalg import eigvalsh 
import numpy as np
from math import sqrt
from Support import print_dims, get_dims


class SRGNeuralNetwork (Restore):
    def __init__(self, weights_file, biases_file):
        super().__init__(weights_file, biases_file)

    def MSE (self,true, test):
        return np.sum((true-test)**2)/len(true)

    def get_diags (self, lst):
        dim = int(sqrt(len(lst)))
        #print ('&&&&&&&', dim)
        return np.sort(np.diag(np.reshape(lst, (dim, dim))))

    def compare(self, prediction_value, true_value):
        predict = self.predict(prediction_value)
        return self.MSE(true_value, predict)

    def compare_eigenvalues(self, prediction_values, true_value):
        diff = []
        for i in prediction_values:
            print(i)
            predict = self.predict(i)
            #print ('^^^^^^^^^^^^^^^')
            #print (predict)
            #print ('$$$$$$$$$$$$$$$$$$$$$$$$$')
            eigen_predict = self.get_diags(predict)
            diff.append(self.MSE(true_value, eigen_predict))
        return diff


def main():
    print("Initializing Neural Network")
    srg_train = Trainer(1, 375, 0.1, 1, 36, 'input_pm_ds_1.npy', 'SRG_PM/SRG_PM_4_DS_1.npy')
    print ("Training and Saving Weights and Biases")
    srg_train.run(3000, 'results_pm/weights_4_1', 'results_pm/biases_4_1')
    w = srg_train.get_weights()
    print(len(w))
    print(srg_train.get_loss())
    H0 = np.load('SRG_PM/SRG_PM_4_DS_1e-1.npy')[0]
    H0 = np.reshape(H0, (6, 6))
    true = eigvalsh(H0)
    print ("Restoring Neural Network")
    srg = SRGNeuralNetwork('results_pm/weights_4_1.npy', 'results_pm/biases_4_1.npy')
    print(srg.compare_eigenvalues([[1.01], [5.01], [10.01], [100]], true))


if __name__=='__main__':
    main()
