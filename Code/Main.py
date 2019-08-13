##################################################
##################################################

##################################################
##################################################

##############################
# IMPORTS
##############################
# LOCAL IMPORTS
from NeuralNetork import Trainer
from SRGNeuralNetwork import SRGNeuralNetwork

def train_and_predict_eigen_srg(nn_specs, save_names, prediction_values):
    """
        Inputs:
            nn_specs (a list of length 8): Contains the following specifications for the neural network in order:
                number of hidden layers, number of neurons per hiddne layer, learning rate, input data dimension,
                output data dimension, file location of input data, file location of training data, number of 
                training iterations
            save_names (a list of length 2): Contains the following names in order: the location to save/load
                the trained weights, the location to save/load the trained biases
            
    """
    print("Initializing Neural Network")
    srg_train = Trainer(nn_specs[0], nn_secs[1], nn_specs[2], nn_specs[3], nn_specs[4], nn_specs[5], nn_specs[6])
    
    print ("Training and Saving Weights and Biases")
    srg_train.run(nn_specs[7], save_names[0], save_names[1])
    
    print('\nLoss: ', srg_train.get_loss(), '\n')
    
    H0 = np.load(nn_specs[6])[0]
    H0 = np.reshape(H0, (6, 6))
    true = eigvalsh(H0)
    
    print ("Restoring Neural Network")
    srg = SRGNeuralNetwork(save_names[0], save_names[1])
    
    print(srg.compare_eigenvalues(prediction_values, true))
    
def predict_eigen_srg (training_file, save_names, prediction_values):
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
