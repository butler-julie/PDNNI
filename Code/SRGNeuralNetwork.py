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
# OUTLINE:
# SRGNeuralNetwork (Restore): A child class of restore that provides 
# additional functions for analysis of SRG matrices created with the 
# neural network.
# 
#    __init__(self, weights_file, biases_file): Imports the trained weights 
#    and biases using the initializer of Restore.
#
#   get_diags (self, lst): Returns the diagonals of a matrix sorted by size.
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
    """
        A child class of restore that provides additional functions for analysis of
        SRG matrices created with the neural network.
    """
    # __INIT__
    def __init__(self, weights_file, biases_file):
        """
            Inputs:
                weights_file (a str): the location of the numpy file that holds the trained 
                    weights                
                biases_file (a str): the location of the numpy file that holds the trained 
                    biases
            Imports the trained weights and biases using the initializer of Restore.
        """
        # Sends the arguments to the initializer of Restore (the parent class)
        super().__init__(weights_file, biases_file)

    # GET_DIAGS
    def get_diags (self, lst):
        """
            Inputs:
                lst (a list or numpy array): a flattened version of the matrix to get the 
                    diagonals of
            Returns:
                Unnamed (a numpy array): the diagonals of the matrix, sorted by size
            Returns the diagonals of a matrix sorted by size.
        """
        # Gets the dimension of the matrix (the flattened list is a dim x dim matrix)
        dim = int(sqrt(len(lst)))
        # Reshapes the list into a dim x dim matrix, gets the diagonal elements, and then sorts
        # them by size
        return np.sort(np.diag(np.reshape(lst, (dim, dim))))

    # COMPARE
    def compare(self, prediction_value, true_value):
        """
            Inputs:
                
        """
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



