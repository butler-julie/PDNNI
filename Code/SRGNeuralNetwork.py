##################################################
# SRG Neural Network
# Julie Hartley
# Version 1.0.0
# August 13, 2019
#
# A child class of Restore, useful when a neural network
# is trained on SRG matrices.  
# Currently only compares the diagonals of the SRG matrices
# to the true eigenvalues.
##################################################

##################################################
##################################################

##############################
# IMPORTS
##############################
# LOCAL IMPORTS
from NeuralNetwork import Restore

##############################
# SRGNEURALNETWORK
##############################
class SRGNeuralNetwork (Restore):
    def __init__(self, weights_file, biases_file):
        super().__init__(weights_file, biases_file)

    def MSE (self,true, test):
        return np.sum((true-test)**2)/len(true)

    def get_diags (self, lst):
        dim = int(sqrt(len(lst)))
        return np.sort(np.diag(np.reshape(lst, (dim, dim))))

    def compare(self, prediction_value, true_value):
        predict = self.predict(prediction_value)
        return self.MSE(true_value, predict)

    def compare_eigenvalues(self, prediction_values, true_value):
        diff = []
        for i in prediction_values:
            print(i)
            predict = self.predict(i)
            eigen_predict = self.get_diags(predict)
            diff.append(self.MSE(true_value, eigen_predict))
        return diff



