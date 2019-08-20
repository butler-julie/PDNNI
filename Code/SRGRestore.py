##################################################
# SRG Restore
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
# SRGRestore (Restore): A child class of restore that provides 
# additional functions for analysis of SRG matrices created with the 
# neural network.
# 
#    __init__(self, weights_file, biases_file): Imports the trained weights 
#    and biases using the initializer of Restore.
#
#   get_diags (self, lst): Returns the diagonals of a matrix sorted by size.
#
#   compare_eigenvalues(self, prediction_values, true_value): Compares the 
#   diagonals from predicted SRG matrices to the exact eigenvalues of the starting
#   matrix.
#
#   graph_compare_eigenvalues(self, prediction_values, true_value):Plots the MSE error
#   between the diagonals of the predicted SRG matrices and the true eigenvalues
#   of the starting matrix.
##################################################

##############################
# IMPORTS
##############################
# LOCAL IMPORTS
from NeuralNetwork import Restore

##############################
# SRGRestore
##############################
class SRGRRestore (Restore):
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
    
    # COMPARE_EIGENVALUES
    def compare_eigenvalues(self, prediction_values, true_value):
        """
            Inputs:
                prediction_values (a 2D list or array): the input values at which predictions
                    are to be made with the neural network.  Must be 2D even if the input is 
                    a single dim (ex: [[2.0]] is acceptable but [2.0] or 2.0 are not)
                true_value: the true eigenvalues of the matrix
            Returns:
                diff (a list): a list of the MSE difference of the predicted value and the true
                    value for each inputted prediction value.
            Compares the diagonals from predicted SRG matrices to the exact eigenvalues of the starting
            matrix.
        """
        diff = []
        for i in prediction_values:
            # Get the predicted SRG matrix at the current input
            predict = self.predict(i)
            # Get diagonals of predicted matrix
            diags_predict = self.get_diags(predict)
            # Compare diagonals to eigenvalues, add MSE to diff
            diff.append(self.L2(true_value, diags_predict))
        return diff
    
    def graph_compare_eigenvalues(self, prediction_values, true_value, filename):
        """
            Inputs:
                prediction_values (a 2D list or array): the input values at which predictions
                    are to be made with the neural network.  Must be 2D even if the input is 
                    a single dim (ex: [[2.0]] is acceptable but [2.0] or 2.0 are not)
                true_value (a list or numpy array): the true eigenvalues of the matrix
                filename (a str): the location to save the plot
            Plots the MSE error between the diagonals of the predicted SRG matrices and the true eigenvalues
            of the starting matrix.
        """
        # Get the MSE error
        diff = self.compare_eigenvalues (prediction_values, true_value)
        # Plot the MSE error
        plt.plot(prediction_values, diff, label='MSE Error', 'go', linewidth=4.0)
        # Save the plot
        plt.save(filename)



