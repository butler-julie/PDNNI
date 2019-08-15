#################################################
# Neural Network
# Julie Hartley
# Version 0.0.5
# August 8, 2019
#
# Basic code for a neural network that allows the
# trained weights and biases to be saved and 
# used later.
#################################################

#################################################
# OUTLINE
# 
# Trainer
#
# Restore
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
# LOCAL IMPORTS 
# The neural network for Trainer
from NeuralNetworkSupport import neural_network as ua

class Trainer:
    """
        Trains a neural network based on spefications given in the initialization.  Allows for
        weights and biases of the trained neural network to be viewed and saved.
    """
    # __INIT__
    def __init__ (self, hidden_layers, hidden_neurons, learning_rate, input_dim, output_dim,
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
        """
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_data = np.load(input_file)
        self.training_data = np.load(training_file)
        self.weights = []
        self.biases = []
        self.loss = 'NaN'
        self.isTrained = False
    
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
        if isTrianed:
            return self.weights
        else:
            print ("Neural Network is not yet trained.")

    # SAVE_WEIGHTS
    def save_weights(self, filename):
        np.save(filename, self.weights)

    #GET_BIASES
    def get_biases(self):
        """"
            Returns:
                self.biases (a numpy array): the trained biases
            Returns the trained biases of the neural network (if it has been trained).
        """
        return self.biases

    # SAVE_BIASES
    def save_biases(self, filename):
        np.save(filename, self.biases)

    # GET_LOSS
    def get_loss(self):
        return self.loss

    # GET_DIMS
    def get_dims(self):
        return self.hidden_layers, [self.input_dim, self.hidden_neurons, self.output_dim]

    # TRAIN
    def train (self,iterations):
        with tf.variable_scope ('Graph'):
            # Placeholder for the values of s at which SRG matrices will be calculated
            # Given values when the Tensorflow session runs
            input_values = tf.placeholder (tf.float32, shape=[None, self.input_dim], 
                name='input_values')

            # Placeholder for the SRG matrices
            # Given values when the Tensorflow session runs
            output_values = tf.placeholder (tf.float32, shape=[None, self.output_dim], 
                name='output_values')

            # The SRG matrices produced from the odeint solver
            y_true = output_values
            # The values of the SRG matrices approximated by the neural network
            y_approximate = ua (input_values, self.input_dim, self.hidden_neurons, 
                self.output_dim, self.hidden_layers)

            # Function used to train the neural network
            with tf.variable_scope ('Loss'):
                # Cost function
                loss=tf.reduce_mean (tf.square (y_approximate-y_true))
                loss_summary_t = tf.summary.scalar ('loss', loss)

            # Optimizer, uses an Adam optimizer
            adam = tf.train.AdamOptimizer (learning_rate = self.learning_rate)
            # Minimize the cost function using the Adam optimizer
            train_optimizer = adam.minimize (loss)

        # Saves a training session
        saver = tf.train.Saver()
        # Tensorflow Session (what acutally runs the neural network)
        with tf.Session() as sess:          
            # Training the neural network
            print ('Training Universal Approximator:')
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
                if i%100 == 0:
                    print('Current Iteration: ', i)
            
            self.isTrained = True
            self.loss = current_loss
            for i in range(1, self.hidden_layers+1):
                name = "Graph/weights_" + str(i) + ":0"
                self.weights.append(self.get_tensor_numeric(name))
                name = "Graph/biases_" + str(i) + ":0"
                self.biases.append(self.get_tensor_numeric(name))

            self.weights.append(self.get_tensor_numeric("Graph/weights_output_layer:0"))

    # TRAIN_AND_SAVE
    def train_and_save (self, iterations, weights_file, biases_file):
        self.train(iterations)
        self.save_weights(weights_file)
        self.save_biases(biases_file)



class Restore:
    # __INIT__
    def __init__ (self,weights_file, biases_file):
        self.weights = np.load(weights_file, allow_pickle=True)
        self.biases = np.load(biases_file, allow_pickle=True)

    def relu (self, x):
        return x * (x > 0)

    # RESTORE_NN    
    def restore_NN (self, input_vector):
        N = len(self.biases) 
        n = len(self.weights) - 1

        #print(self.weights[0])

        # First hidden layer
        z = np.matmul(input_vector, self.weights[0]) + self.biases[0]
        #print('*******', z)
        #z = input_vector*self.weights[0] + self.biases[0]
        a = self.relu(z)

        for i in range (1, N):
            z = np.matmul(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
            #print(len(a))

        final =  np.matmul(a, self.weights[n])
        #print(len(final))
        return final
        
    # PREDICT
    def predict (self, prediction_value):
        return self.restore_NN(prediction_value)

    def L2 (self,A, B):
        return np.square(np.subtract(A, B)).mean()

    def L1 (self, A, B):
        return (np.subtract(A, B)).mean()


    def compare_to_true (self, prediction_values, true_results):
        L1_tot = []
        L2_tot = []
        for i in range (0, len(prediction_values)):
            predict = self.predict(prediction_values[i])
            l1 = self.L1(true_results[i], predict)
            l2 = self.L2(true_results[i], predict)
            L1_tot.append(l1)
            L2_tot.append(l2)
        return L1_tot, L2_tot
