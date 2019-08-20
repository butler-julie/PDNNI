##################################################
# Main
# Julie Hartley
# Version 1.0.0
# August 14, 2019
#
# A collection of function used to run Trainer,
# Restore, and SRGNeuralNetwork
##################################################

##################################################
# OUTLINE
# train_and_predict_eigen_srg(nn_specs, save_names, prediction_values):
# Trains a neural network based on specifications and then used the 
# trained neural network to predict the eigenvalues of the SRG
# matrices at particular s-values.
#
# predict_eigen_srg (training_file, save_names, prediction_values):
# Uses a trained neural network to predict the eigenvalues of 
# the SRG matrices at particular s-values.
##################################################

##############################
# IMPORTS
##############################
# LOCAL IMPORTS
from NeuralNetork import Trainer
from SRGNeuralNetwork import SRGNeuralNetwork

# TRAIN_AND_PREDICT
def train_and_predict(nn_specs, save_names, prediction_values):
    print("Initializing Neural Network")
    train = Trainer(nn_specs[0], nn_secs[1], nn_specs[2], nn_specs[3], nn_specs[4], nn_specs[5], nn_specs[6])
    
    print ("Training and Saving Weights and Biases")
    strain.train_and_save(nn_specs[7], save_names[0], save_names[1])
    
    print('\nLoss: ', train.get_loss(), '\n')
    
    
    print ("Restoring Neural Network")
    restore = Restore(save_names[0], save_names[1])
    for i in prediction_values:
        predict = restore.predict(i)
        print()
        print('Input: ', i)
        print('Predicted Output: ', predict)
        print()
         
    
    
# PREDICT
def predict(save_names, prediction_values):
    print ("Restoring Neural Network")
    restore = Restore(save_names[0], save_names[1])
    for i in prediction_values:
        predict = restore.predict(i)
        print()
        print('Input: ', i)
        print('Predicted Output: ', predict)
        print()    
    
# TRAIN_AND_ERROR_ANALYSIS
def train_and_error_analysis (nn_specs, save_names, prediction_values, true_values, save_prefix):
    print("Initializing Neural Network")
    train = Trainer(nn_specs[0], nn_secs[1], nn_specs[2], nn_specs[3], nn_specs[4], nn_specs[5], nn_specs[6])
    
    print ("Training and Saving Weights and Biases")
    strain.train_and_save(nn_specs[7], save_names[0], save_names[1])
    
    print('\nLoss: ', train.get_loss(), '\n')
    
    print ("Restoring Neural Network")
    restore = Restore(save_names[0], save_names[1])
    
    mae, mse = restore.average_mae_and_mse (prediction_values, true_values)
    print ()
    print ("Average Mean Absolute Error: ", mae)
    print ("Average Mean Squared Error: ", mse)
    print()
    
    mae_and_mse_filename = save_prefix + 'mae_and_mse.png'
    mae_filename = save_prefix + 'mae.png'
    mse_filename = save_prefix + 'mse.png'
    
    print ("Making and Saving Graph of MAE and MSE")
    restore.graph_mae_and_mse (prediction_values, true_values, mae_and_mse_filename)
    
    print("Making and Saving Graph of MAE")
    restore.graph_mae (prediction_values, true_values, mae_filename)
    
    print("Making and Saving Graph of MSE")
    restore.graph_mse (prediction_values, true_values, mse_filename)
    
    
# ERROR_ANALYSIS
def error_analysis (save_names, prediction_values, save_prefix):
    print ("Restoring Neural Network")
    restore = Restore(save_names[0], save_names[1])
    
    mae, mse = restore.average_mae_and_mse (prediction_values, true_values)
    print ()
    print ("Average Mean Absolute Error: ", mae)
    print ("Average Mean Squared Error: ", mse)
    print()
    
    mae_and_mse_filename = save_prefix + 'mae_and_mse.png'
    mae_filename = save_prefix + 'mae.png'
    mse_filename = save_prefix + 'mse.png'
    
    print ("Making and Saving Graph of MAE and MSE")
    restore.graph_mae_and_mse (prediction_values, true_values, mae_and_mse_filename)
    
    print("Making and Saving Graph of MAE")
    restore.graph_mae (prediction_values, true_values, mae_filename)
    
    print("Making and Saving Graph of MSE")
    restore.graph_mse (prediction_values, true_values, mse_filename)    

def train_and_predict_eigen_srg(nn_specs, save_names, prediction_values):
    """
        Inputs:
            nn_specs (a list of length 8): Contains the following specifications for the neural network in order:
                number of hidden layers, number of neurons per hiddne layer, learning rate, input data dimension,
                output data dimension, file location of input data, file location of training data, number of 
                training iterations
            save_names (a list of length 2): Contains the following names in order: the location to save/load
                the trained weights, the location to save/load the trained biases
            prediction_values (a 2D array): s values at which values of the diagonals are wanted.  Each individual 
                s-value must be its own separate array (ex. input: [[2], [3], [4]])
        Returns:
            Unnamed (an array): the error of the SRG matrices diagonals at inputted s values from the true
                eigenvalues of the neural network
        
        Trains a neural network based on specifications and then used the trained neural network to predict 
        the eigenvalues of the SRG matrices at particular s-values.
    """
    print("Initializing Neural Network")
    srg_train = Trainer(nn_specs[0], nn_secs[1], nn_specs[2], nn_specs[3], nn_specs[4], nn_specs[5], nn_specs[6])
    
    print ("Training and Saving Weights and Biases")
    srg_train.train_and_save(nn_specs[7], save_names[0], save_names[1])
    
    print('\nLoss: ', srg_train.get_loss(), '\n')
    
    H0 = np.load(nn_specs[6])[0]
    H0 = np.reshape(H0, (6, 6))
    true = eigvalsh(H0)
    
    print ("Restoring Neural Network")
    srg = SRGNeuralNetwork(save_names[0], save_names[1])
    
    return srg.compare_eigenvalues(prediction_values, true)
    
def predict_eigen_srg (training_file, save_names, prediction_values):
    """
        Inputs:
            training_file (a str): the location of the training data.  Needed to get the true eigenvalues.
            save_names (a list of length 2): Contains the following names in order: the location to save/load
                the trained weights, the location to save/load the trained biases
            prediction_values (a 2D array): s values at which values of the diagonals are wanted.  Each individual 
                s-value must be its own separate array (ex. input: [[2], [3], [4]])
        Returns:
            Unnamed (an array): the error of the SRG matrices diagonals at inputted s values from the true
                eigenvalues of the neural network
        
        Uses a trained neural network to predict the eigenvalues of the SRG matrices at particular s-values.
    """
    H0 = np.load(training_file)[0]
    H0 = np.reshape(H0, (6, 6))
    true = eigvalsh(H0)
    
    print ("Restoring Neural Network")
    srg = SRGNeuralNetwork(save_names[0], save_names[1])
    
    print(srg.compare_eigenvalues(prediction_values, true))

def main():
    print('Main')

if __name__=='__main__':
    main()
