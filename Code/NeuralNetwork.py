#################################################
# Universal Approximator Matrices
# Julie Hartley
# Version 0.0.5
# August 8, 2019
#
# A universal approximator for data produced by the SRG
# and the IMSRG
#################################################

#################################################
# OUTLINE
# 
#################################################
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
#import matplotlib.pyplot as plt
#from pylab import *

# For SRG and matrix manipulation
# Dor product and matrix/array manipulation
from numpy import array, dot, diag, reshape
# Eigenvalue, eigenvector solver
from scipy.linalg import eigvalsh
# ODE solver
from scipy.integrate import ode

# Creating random numbers (for file naming)
import random
from math import sqrt

# LOCAL IMPORTS 
# Local imports are stored in the common code repository
# The neural network
from NeuralNetworkSupport import neural_network as ua

class Trainer:
    # __INIT__
    def __init__ (self, hidden_layers, hidden_neurons, learning_rate, input_dim, output_dim,
        input_file, training_file):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim_sqrt = sqrt(self.output_dim)
        self.input_data = np.load(input_file)
        self.training_data = np.load(training_file)
        self.weights = []
        self.biases = []
        self.loss = 'NaN'
    
    # GET_TENSOR_NUMERIC
    def get_tensor_numeric (self, name):
        return tf.get_default_graph().get_tensor_by_name(name).eval()

    def get_weights(self):
        return self.weights

    def save_weights(self, filename):
        np.save(filename, self.weights)

    def get_biases(self):
        return self.biases

    def save_biases(self, filename):
        np.save(filename, self.biases)

    def get_loss(self):
        return self.loss

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
            
            self.loss = current_loss
            for i in range(1, self.hidden_layers+1):
                name = "Graph/weights_" + str(i) + ":0"
                self.weights.append(self.get_tensor_numeric(name))
                name = "Graph/biases_" + str(i) + ":0"
                self.biases.append(self.get_tensor_numeric(name))

            self.weights.append(self.get_tensor_numeric("Graph/weights_output_layer:0"))

    def run (self, iterations, weights_file, biases_file):
        self.train(iterations)
        self.save_weights(weights_file)
        self.save_biases(biases_file)



class Restore:
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
        

    def predict (self, prediction_value):
        return self.restore_NN(prediction_value)

        
     





"""



# Using the neural network to predict values of the SRG flow equation

# The values of s to calculate matrices at
prediction_values = np.arange(0, 10.2, 0.01)
prediction_values = prediction_values.reshape (len(prediction_values), 1)

# Dummy variable for prediction algorithm        
zeros = np.zeros([10**2])
dummy_output = []
dummy_output.append (zeros)

# Use the trained neural network to make the predictions
y_true_results, y_approximate_results = sess.run ([y_true, y_approximate], 
    feed_dict={input_values:prediction_values, output_values:dummy_output})

true = eigvalsh(reshape(self.training_data[0], (10,10)))
predict_ml = np.sort(diag(reshape(y_approximate_results[-1], (10,10))))
N = len(true)

ml_error = np.sum((true-predict_ml)**2)/N

print ('ML Error ', ml_error, '\n')
"""

"""       
)


    def prediction_error_analysis(self, prediction_values):
        true_result = eigvalsh(reshape(self.H0, (self.output_dim_sqrt,self.output_dim_sqrt)))
        y_true_results, y_approximate_results = predict(prediction_values)

        predict_ml = get_eigenvalues(y_approximate_result)
        predict_ode = get_eigenvalues(y_true_results)

        N = len(true)

        ml_error = MSE(true_result, predict_ml)
        ode_error = MSE(true_result, predict_ode)
    
        return ml_error, ode_error
"""
