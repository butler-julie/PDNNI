
#############################
# IMPORTS
#############################
# SYSTEM IMPORTS
import time, os, sys
dir = os.path.dirname (os.path.realpath (__file__))
# For timing
import timeit

# THIRD-PARTY IMPORTS
# For machine learning
import tensorflow as tf

# For calculations
import numpy as np

# For graphing
import matplotlib.pyplot as plt
from pylab import *

# For SRG and matrix manipulation
# Dor product and matrix/array manipulation
from numpy import array, dot, diag, reshape
# Eigenvalue, eigenvector solver
from scipy.linalg import eigvalsh
# ODE solver
from scipy.integrate import ode

# Creating random numbers (for file naming)
import random

# LOCAL IMPORTS 
# Local imports are stored in the common code repository
sys.path.append(os.path.abspath("../CommonCode"))
# The neural network
from NeuralNetworkFunctions import universal_function_approximator_N_hidden_layers as ua
# Diagonalizes a matrix with sorted eigenvalues
from LinearAlgebra import diagonalize_matrix as diag_mat
# Commutator
from LinearAlgebra import commutator
# Used to compare results
from MatrixComparisons import compare_square_matrices_by_element as diff_element
from MatrixComparisons import compare_square_matrices as diff
from MatrixComparisons import compare_square_matrices_diagonals as diff_diag
# The SRG generator (using the White generator)
from Generators import white as generator

def class UniversalApproximator:
    def __init__ (self, hidden_layers, hidden_neurons, learning_rate, input_dim, output_dim,
        input_file, training_file):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_data = np.load(input_file)
        self.training_data = np.load(training_file)
    def function_to_approximate (self, input_values, output_values):
        """
            Inputs:
                flow_parameters (a 2D-array): the flow parameters at which SRG matrices are
                    to be calculated.  Each inner array is simply a number, a single flow
                    parameter.
                SRG_matrices (a 2D-array): SRG matrices for the 
                    given flow parameters. Each inner array is 36 numbers long; they are
                    flattened SRG matrices.
                times (an int): the number of times the training data is to be fed through 
                    the neural network in a given iteration

            Returns:
                SRG_matrices or return_SRG_matrices (a 2D-array): SRG matrices for the 
                    given flow parameters. Each inner array is 36 numbers long; they are
                    flattened SRG matrices. If times = 1 then it is just the SRG matrices.
                    If times is greater than 1 then it is the SRG matrices concatentated the
                    given number of times.
            Returns the SRG matrices to train the neural network for the given flow parameter
            values.  The order and length of the flow parameters array must match the order
            and length of the SRG_matrices array.
        """        
        return output_values

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
            y_true = function_to_approximate (input_values, output_values)
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
                output_values = self.output_data
                # Runs the Tensorflow session
                current_loss, loss_summary, _ = sess.run ([loss, loss_summary_t, 
                    train_optimizer], feed_dict = { input_values:self.input_data,
                     output_values:self.output_data})
    def predict (self, prediction_values):
        prediction_values = prediction_values.reshape (len(prediction_values), 1)

        # Dummy variable for prediction algorithm        
        zeros = np.zeros([len(self.output_data[0])**2])
        dummy_output = []
        dummy_output.append (zeros)
        
        # Use the trained neural network to make the predictions
        y_true_results, y_approximate_results = sess.run ([y_true, y_approximate], 
            feed_dict={flow_params:prediction_values, SRG_matrices:dummy_output})
    
        return y_true_results, y_prediction_results

