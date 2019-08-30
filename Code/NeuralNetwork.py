#################################################
# Neural Network
# Julie Hartley
# Version 3.0.0
# August 8, 2019
#
# Basic code for a neural network that allows the
# trained weights and biases to be saved and 
# used later.
#################################################

#################################################
# OUTLINE
# 
# Trainer: Trains a neural network based on spefications given in the initialization.  Allows for
# weights and biases of the trained neural network to be viewed and saved.
#
#   __init__ (self, hidden_layers, hidden_neurons, learning_rate, input_dim, output_dim,
#   input_file, training_file):  Initializes the neural network with given specifications
#
#   get_tensor_numeric (self, name): Gets the numeric value of a specific Tensorflow tensor, 
#   specified by name.
#
#   get_weights(self): Returns the trained weights of the neural network (if it has been trained).
#
#   save_weights(self, filename): Save the trained weights to a specified location.
#
#   get_biases(self): Returns the trained biases of the neural network (if it has been trained).
#
#   save_biases(self, filename): Save the trained biases to a specified location.
#
#   get_loss(self): Returns the final loss of the trained neural network.
#
#   get_losses(self): Returns the losses for every training iteration.
#
#   save_losses(self, filename): Saves the losses for every training iteration to a specified location.
#
#   get_dims(self):  Returns the architecture of the neural network.
#
#   save_dims(self, filename): Saves the architecture of the neural network (number of hidden 
#   layers, input dimension, number of hidden neurons, and output dimension).
#
#   train (self,iterations): Trains the neural network with specified number of training iterations, 
#   save the weights, biases, and final loss.
#
#   train_and_save (self, iterations, weights_file, biases_file): Trains the neural network and then 
#   saves the weights and biases.
#
# Restore: Restores a trained neural network and uses it to make predictions and perform error analysis 
# on the neural network's results.
#
#   __init__ (self,weights_file, biases_file): Retrieves the trained weights and biases from file.
#
#   relu (self, x): Returns f(x) = max(0, x) (the rectified linear function).
#
#   restore_NN (self, input_vector): Uses trained weights and biases to predict the value of 
#   the neural network at a given input.
#
#   predict (self, prediction_value): Uses the restored neural network to preduct the output value 
#   for the given input.
#
#   mse (self,A, B): Finds the MSE error between two lists/arrays.
#
#   mae (self,A, B): Finds the MAE error between two lists/arrays.
#
#   compare_to_true (self, prediction_values, true_results): Compares results of the neural network 
#   to the true results using two metrics (MAE and MSE) at different input values.
#
#   average_mae_and_mse (self, prediction_values, true_results): Finds the average MAE and MSE errors of 
#   the neural network results from true results from a given set of input values.
#
#   graph_mae_and_mse (self, prediction_values, true_results, filename): Produces a graph of 
#   the MAE and MSE errors from compare_to_true.
#
#   graph_mae (self, prediction_values, true_results, filename): Produces a graph of the MAE error 
#   from compare_to_true.
#
#   graph_mse (self, prediction_values, true_results, filename): Produces a graph of the MSE error 
#   from compare_to_true.
#################################################
#############################
# IMPORTS
#############################
# THIRD-PARTY IMPORTS
# For machine learning
import tensorflow as tf
# For calculations
import numpy as np
from math import sqrt
# for graphing
import matplotlib.pyplot as plt
# LOCAL IMPORTS 
# The neural network for Trainer
from NeuralNetworkSupport import neural_network as ua

#############################
# TRAINER
#############################
class Train:
    """
        Trains a neural network based on spefications given in the initialization.  Allows for
        weights and biases of the trained neural network to be viewed and saved.
    """
    # __INIT__
    def __init__ (self, hidden_layers, hidden_neurons, input_dim, output_dim,
        input_file, training_file):
        """
            Inputs:
                hidden_layers (an int): the number of hidden layers in the neural network
                hidden_neurons (an int): the number of neurons per hidden layer
                learning_rate (a float): the learning rate for the optimizer (typically less
                    than one)
                input_dim (an int): the dimension of the input data
                output_dim (an int): the dimension of the training data
                input_file (a str): the location where the input data is saved (must be .npy extension)
                training_file (a str): the location where the training data is saved (must be .npy extension)
            Initializes the neural network with given specifications.
        """
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_data = np.load(input_file)
        self.training_data = np.load(training_file)
        # Initialize placeholders for the trained weights, biases, and final loss
        self.weights = []
        self.biases = []
        self.loss = 'NaN'
        # Changed to True after training
        self.isTrained = False
        # Holds losses for every training iteration is isSavedLosses=True
        self.losses = []
    
    # GET_TENSOR_NUMERIC
    def get_tensor_numeric (self, name):
        """
            Inputs:
                name (a str): The name of the tensor
            Returns:
                Unnamed: The numeric value of the tensor (either a single number
                    or a numpy array
            Gets the numeric value of a specific Tensorflow tensor, specified by name.
        """
        return tf.get_default_graph().get_tensor_by_name(name).eval()

    # GET_WEIGHTS
    def get_weights(self):
        """"
            Returns:
                self.weights (a numpy array): the trained weights
            Returns the trained weights of the neural network (if it has been trained).
        """
        if self.isTrained:
            return self.weights
        else:
            print ("Neural Network is not yet trained.")

    # SAVE_WEIGHTS
    def save_weights(self, filename):
        """
            Inputs:
                filename (a str): the location to save the trained weights
            Save the trained weights to a specified location.
        """
        if self.isTrained:
            np.save(filename, self.weights)
        else:
            print ("Neural Network is not yet trained.")            

    #GET_BIASES
    def get_biases(self):
        """"
            Returns:
                self.biases (a numpy array): the trained biases
            Returns the trained biases of the neural network (if it has been trained).
        """
        if self.isTrained:
            return self.biases
        else:
            print ("Neural Network is not yet trained.")        

    # SAVE_BIASES
    def save_biases(self, filename):
        """
            Inputs:
                filename (a str): the location to save the trained biases
            Save the trained biases to a specified location.
        """        
        if self.isTrained:
            np.save(filename, self.biases)
        else:
            print ("Neural Network is not yet trained.")            

    # GET_LOSS
    def get_loss(self):
        """
            Returns:
                self.loss (a float): the final loss of the trained neural network
            Returns the final loss of the trained neural network.
        """
        if self.isTrained:
            return self.loss
        else:
            print ("Neural Network is not yet trained.")
            
    # GET_LOSSES        
    def get_losses(self):
        """
            Returns:
                get_losses (a list): the loss for every training iterations
            Returns the losses for every training iteration.
        """
        if self.isTrained:
            return self.losses
        else:
            print("Neural network is not trained, or losses were not saved by input")
         
    # SAVE_LOSSES    
    def save_losses(self, filename):
        """
            Input:
                filename (a str): the location to save the losses
            Saves the losses for every training iteration to a specified location.
        """
        if self.isTrained:
            np.save(filename, self.losses)
        else:
           print("Neural network is not trained, or losses were not saved by input")
          
    # GRAPH_LOSSES                  
    def graph_losses (self, filename):
        N = len(self.losses)
        x = np.arange(0, N, 1)
        plt.plot (x, self.losses, 'go', linewidth=4.0)
        plt.savefig(filename)
                  
    # GET_DIMS
    def get_dims(self):
         """
            Returns:
                self.hidden_layers (an int): the number of hidden layers in the neural network
                Unnamed (a list): contains three elements: the dimension of the input layer, the 
                    number of neurons for each hidden layer, and the dimension of the output layer
            Returns the architecture of the neural network.
         """
         return self.hidden_layers, [self.input_dim, self.hidden_neurons, self.output_dim]
                  
    # SAVE_DIMS
    def save_dims(self, filename):
        """
            Input:
                filename (a str): the location to save the architecture of the neural network
            Saves the architecture of the neural network (number of hidden layers, input dimension, 
            number of hidden neurons, and output dimension).
        """
        lst = [self.hidden_layers, [self.input_dim, self.hidden_neurons, self.output_dim]]
        np.save (filename, lst)

    # TRAIN
    def train (self,iterations, learning_rate):
        """"
            Input:
                iterations (an int): the number of training iterations
            Trains the neural network with specified number of training iterations, save the weights, biases, and
            final loss.
        """
        # All Tensorflow calculations must take place inside a graph
        with tf.variable_scope ('Graph'):
            # Placeholder for the values of the input
            # Given values when the Tensorflow session runs
            input_values = tf.placeholder (tf.float32, shape=[None, self.input_dim], 
                name='input_values')

            # Placeholder for the training data
            # Given values when the Tensorflow session runs
            output_values = tf.placeholder (tf.float32, shape=[None, self.output_dim], 
                name='output_values')

            # The true values are the training data
            y_true = output_values
            # Output values produced by the neural network
            y_approximate = ua (input_values, self.input_dim, self.hidden_neurons, 
                self.output_dim, self.hidden_layers)

            # Function used to train the neural network
            with tf.variable_scope ('Loss'):
                # Cost function (currently MSE)
                loss=tf.reduce_mean (tf.square (y_approximate-y_true))
                loss_summary_t = tf.summary.scalar ('loss', loss)

            # Optimizer, uses an Adam optimizer
            adam = tf.train.AdamOptimizer (learning_rate = learning_rate)
            # Minimize the cost function using the Adam optimizer
            train_optimizer = adam.minimize (loss)

        # Tensorflow Session (what acutally runs the neural network)
        with tf.Session() as sess:          
            # Training the neural network
            # Start the Tensorflow Session
            sess.run (tf.global_variables_initializer ())
            for i in range (iterations):
                # The actual values that will be put into the placeholder input_vector
                self.input_data = self.input_data.reshape (len(self.input_data), 
                    1) 
                # Runs the Tensorflow session
                current_loss, loss_summary, _ = sess.run ([loss, loss_summary_t, 
                    train_optimizer], feed_dict = { input_values:self.input_data,
                     output_values:self.training_data})
                # Save current loss
                self.losses.append(current_loss)
            
            # Neural network is trained
            self.isTrained = True
                  
            self.loss = current_loss
                  
            # Get the numeric values of the weights and biases of the hidden layers
            for i in range(1, self.hidden_layers+1):
                name = "Graph/weights_" + str(i) + ":0"
                self.weights.append(self.get_tensor_numeric(name))
                name = "Graph/biases_" + str(i) + ":0"
                self.biases.append(self.get_tensor_numeric(name))

            # Get the numeric values of the weigts of the output layer
            self.weights.append(self.get_tensor_numeric("Graph/weights_output_layer:0"))

    # TRAIN_AND_SAVE
    def train_and_save (self, iterations, learning_rate, weights_file, biases_file):
        """
            Inputs:
                iterations (an int): the number of training iterations
                weights_file (a str): the location to save the trained weights
                biases_file (a str): the location to save the trained biases
            Trains the neural network and then saves the weights and biases.
        """
        # Train the neural network
        self.train(iterations, learning_rate)
                  
        # Save the weigths and biases
        self.save_weights(weights_file)
        self.save_biases(biases_file)



#############################
# RESTORE                  
#############################
class Restore:
    """
        Restores a trained neural network and uses it to make predictions and perform error analysis 
        on the neural network's results.
    """
    # __INIT__
    def __init__ (self,weights_file, biases_file):
        """
            Inputs:
                weights_file (a str): the location where the trained weights are saved
                biases_file (a str): the location where the trained biases are saved
            Retrieves the trained weights and biases from file.
        """
        self.weights = np.load(weights_file, allow_pickle=True)
        self.biases = np.load(biases_file, allow_pickle=True)

    # RELU
    def relu (self, x):
        """
            Inputs:
                x (a float)
            Returns f(x) = max(0, x) (the rectified linear function).
        """
        return x * (x > 0)

    # RESTORE_NN    
    def restore_NN (self, input_vector):
        """
            Input:
                input_vector (a list or numpy array): the input to the neural network
            Returns:
                final (a numpy array): the output of the neural network
            Uses trained weights and biases to predict the value of the neural network
            at a given input.
        """
        # The number of biases (the number of hidden layers)
        N = len(self.biases) 
        # The index of the last entry in the weights array
        n = len(self.weights) - 1
                  
        # First hidden layer
        z = np.matmul(input_vector, self.weights[0]) + self.biases[0]
        a = self.relu(z)
        
        # Finds the output of the hidden layers of the neural network
        #a = input_vector
        for i in range (1, N):
            z = np.matmul(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
        # The result of the output layer of the neural network
        # No biases or activation function by design
        # See neural_network in NeuralNetworkSupport.py
        final =  np.matmul(a, self.weights[n])
        return final
        
    # PREDICT
    def predict (self, prediction_value):
        """
            Input:
                prediction_value (a list or numpy array): the input value for the neural network
            Returns:
                Unnamed (a numpy array): the output of the neural network from the given input value
            Uses the restored neural network to preduct the output value for the given input.
        """
        return self.restore_NN(prediction_value)
                  
    # BATCH_PREDICT
    def batch_predict(self, prediction_values, verbose=True):
        if verbose:
            for i in prediction_values:
                predict = self.predict(i)
                print()
                print('Input: ', i)
                print('Predicted Output: ', predict)
                print()      
        else:
            predicted = []
            for i in prediction_values:    
                predict = self.predict(i)
                predicted.append(predict)

    # MSE
    def mse (self,A, B):
        """
            Inputs:
                A, B (lists or numpy arrays)
            Returns:
                Unnamed (a float): the MSE error between A and B
            Finds the mean squared error between two lists/arrays.
        """
        return np.square(np.subtract(A, B)).mean()
    
    # MAE
    def mae (self, A, B):
        """
            Inputs:
                A, B (lists or numpy arrays)
            Returns:
                Unnamed (a float): the MAE error between A and B
            Finds the mean absolute error between two lists/arrays.
        """                  
        return np.absolute(np.subtract(A, B)).mean()

    # COMPARE_TO_TRUE
    def compare_to_true (self, prediction_values, true_results):
        """
            Inputs:
                prediction_values (a 2D list or numpy array): the values to generate outputs of the neural
                    network.  Must be a 2D list or array even if the input is one dimension.  (ex: [[2.0]]
                    is fine but [2.0] or 2.0 are not.
                true_results (a list or array): the true results that correspond to each value in 
                    prediction_values
            Returns:
                mae_tot (a list): the MAE error for each value predicted by the neural network and the true
                    result
                mse_tot (a list): the MSE error for each value predicted by the neural network and the true
                    result   
            Compares results of the neural network to the true results using two metrics (MAE and MSE) at
            different input values.
        """
        mae_tot = []
        mse_tot = []
        # cycle through prediction_values
        for i in range (0, len(true_results)):
            # Get the value predicted by the neural network
            predict = self.predict(prediction_values[i])
            # Get the MAE and MSE errors betwwn the predicted and true resutls
            mae = self.mae(true_results[i], predict)
            mse = self.mse(true_results[i], predict)
            mae_tot.append(mae)
            mse_tot.append(mse)
        return mae_tot, mse_tot
                 
    # AVERAGE_MAE_AND_MSE              
    def average_mae_and_mse (self, prediction_values, true_results, verbose=False):
        """
            Inputs:
                prediction_values (a 2D list or numpy array): the values to generate outputs of the neural
                    network.  Must be a 2D list or array even if the input is one dimension.  (ex: [[2.0]]
                    is fine but [2.0] or 2.0 are not.
                true_results (a list or array): the true results that correspond to each value in 
                    prediction_values
            Returns:
               mae_avg (a float): the average MAE error from all the MAE errors produced by compare_to_true
               mse_avg (a float): the average MAE error from all the MSE errors produced by compare_to_true
           Finds the average MAE and MSE of the neural network results from true results from a given set
           of input values.
        """
        mae, mse = self.compare_to_true (prediction_values, true_results)            
        mae_avg = np.average(mae)
        mse_avg = np.average(mse)
        if verbose:
            print ('Average MAE Error: ', mae_avg)
            print ('Average MSE Error: ', mse_avg)
        else:
            return mae_avg, mse_avg
    
    # GRAPH_MAE_AND_MSE              
    def  graph_mae_and_mse (self, prediction_values, true_results, filename):
        """
            Inputs: 
                prediction_values (a 2D list or numpy array): the values to generate outputs of the neural
                    network.  Must be a 2D list or array even if the input is one dimension.  (ex: [[2.0]]
                    is fine but [2.0] or 2.0 are not.
                true_results (a list or array): the true results that correspond to each value in 
                    prediction_values
                filename (a str): the location to save the graph
            Produces a graph of the MAE and MSE from compare_to_true.
        """
        # Get the MAE and MSE errors
        mae, mse = self.compare_to_true (prediction_values, true_results)                  
        # Plot the MAE and MSE errors
        prediction_values = prediction_values.flatten()
        plt.plot (prediction_values, mae, 'r^', label='MAE', linewidth=4.0)
        plt.plot (prediction_values, mse, 'bo', label='MSE', linewidth=4.0)
        # Add the legend
        plt.legend()
        # Save the graph
        plt.savefig(filename)

    # GRAPH_MAE              
    def  graph_mae (self, prediction_values, true_results, filename):
        """
            Inputs: 
                prediction_values (a 2D list or numpy array): the values to generate outputs of the neural
                    network.  Must be a 2D list or array even if the input is one dimension.  (ex: [[2.0]]
                    is fine but [2.0] or 2.0 are not.
                true_results (a list or array): the true results that correspond to each value in 
                    prediction_values
                filename (a str): the location to save the graph
            Produces a graph of the MAE error from compare_to_true.
        """
        # Get the MAE and MSE errors
        mae, mse = self.compare_to_true (prediction_values, true_results)                  
        # Plot the MAE error
        prediction_values = prediction_values.flatten()
        plt.plot (prediction_values, mae, 'r^', label='MAE Error', linewidth=4.0)
        # Save the graph
        plt.savefig(filename)
          
    # GRAPH_MSE              
    def  graph_mse (self, prediction_values, true_results, filename):
        """
            Inputs: 
                prediction_values (a 2D list or numpy array): the values to generate outputs of the neural
                    network.  Must be a 2D list or array even if the input is one dimension.  (ex: [[2.0]]
                    is fine but [2.0] or 2.0 are not.
                true_results (a list or array): the true results that correspond to each value in 
                    prediction_values
                filename (a str): the location to save the graph
            Produces a graph of the MSE error from compare_to_true.
        """
        # Get the MAE and MSE errors
        mae, mse = self.compare_to_true (prediction_values, true_results)                  
        # Plot the MSE error
        prediction_values = prediction_values.flatten()
        plt.plot (prediction_values, mse, 'bo', label='MSE Error', linewidth=4.0)
        # Save the graph
        plt.savefig(filename)  
                  
    # ERROR_ANALYSIS
    def error_analysis (self, prediction_values, true_values, save_prefix):

        mae, mse = self.average_mae_and_mse (prediction_values, true_values)
        print ()
        print ("Average Mean Absolute Error: ", mae)
        print ("Average Mean Squared Error: ", mse)
        print()

        mae_and_mse_filename = save_prefix + 'mae_and_mse.png'
        mae_filename = save_prefix + 'mae.png'
        mse_filename = save_prefix + 'mse.png'

        print ("Making and Saving Graph of MAE and MSE")
        self.graph_mae_and_mse (prediction_values, true_values, mae_and_mse_filename)

        print("Making and Saving Graph of MAE")
        self.graph_mae (prediction_values, true_values, mae_filename)

        print("Making and Saving Graph of MSE")
        self.graph_mse (prediction_values, true_values, mse_filename)                      
                  
class TrainAndRestore (Train, Restore):
# __INIT__
    def __init__ (self, hidden_layers, hidden_neurons, input_dim, output_dim,
        input_file, training_file):
        Train.__init__(hidden_layers, hidden_neurons, input_dim, output_dim,
        input_file, training_file)
                  
    # TRAIN_AND_BATCH_PREDICT
    def train_and_batch_predict(iterations, learning_rate, save_names, prediction_values):
        print ("Training and Saving Weights and Biases")
        self.train_and_save(iterations, learning_rate, save_names[0], save_names[1])

        print('\nLoss: ', train.get_loss(), '\n')


        print ("Restoring Neural Network")
        restore = Restore(save_names[0], save_names[1])
        self.batch_predict(prediction_values)
                  
    # TRAIN_AND_ERROR_ANALYSIS
    def train_and_error_analysis (iterations, learning_rate, save_names, prediction_values, true_values, save_prefix):
        print ("Training and Saving Weights and Biases")
        strain.train_and_save(iterations, learning_rate, save_names[0], save_names[1])

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


                  
                
                  
