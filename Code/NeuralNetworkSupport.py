########################################################################################
# NeuralNetworkFunctions.py
# Julie Butler
# February 20, 2019
# Version 0.2
#
# A collection of functions for setting up and running neural networks.  Many are 
# related to the use of neural networks as universal function approximators.

# FUNCTIONS:
# initilize_neural_network:  Creates the weights and biases of a neural network, for any
#   number of hidden layers given in an inputted array of seed values for the weights and
#   biases.
# xavier_initialization: initializes the weights of a neural network using the Xavier
#   initialization method
# universal_function_approximator: approximates a function of any input and output 
#   dimensions using a one hidden layer neural network with a specified number of 
#   neurons.
########################################################################################

#############################
#                           #
# IMPORTS                   #
#                           #
#############################
# THIRD-PARTY IMPORTS
# For machine learning
import tensorflow as tf
# For matrix calculations
import numpy as np

def neural_network (input_vector, input_dim, hidden_dim, output_dim, num_hidden_layers):
    # First Hidden Layer
    weights_first_hidden_layer = tf.get_variable (name='weights_1', 
        shape=[input_dim, hidden_dim],
        initializer=tf.random_normal_initializer(stddev=0.1))
        
    biases_first_hidden_layer = tf.get_variable (name='biases_1', 
        shape=[hidden_dim],
        initializer=tf.constant_initializer(0.0))
        
    z = tf.matmul (input_vector, weights_first_hidden_layer) + biases_first_hidden_layer
    activated_function = tf.nn.relu (z)

    for i in range (2, num_hidden_layers):
        # Interior Hidden Layers
        weight_name = 'weights_' + str(i)
        bias_name = 'biases_' + str(i)

        weights_hidden_layer_1 = tf.get_variable (name=weight_name, 
            shape=[hidden_dim, hidden_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
        
        biases_hidden_layer_1 = tf.get_variable (name=bias_name, 
            shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0))
        
        z = tf.matmul (activated_function, weights_hidden_layer_1) + biases_hidden_layer_1
        activated_function = tf.nn.relu (z)

    #Output Layer
    weights_output_layer = tf.get_variable (name='weights_output_layer', 
        shape=[hidden_dim, output_dim],
        initializer=tf.random_normal_initializer(stddev=0.1))
            
    # See notebook #1 page 22
    z = tf.matmul (activated_function, weights_output_layer)
        
    return z
"""
PANIC VERSION -- WORKS WITH 2 HIDDEN LAYERS
# UNIVERSAL_FUNCTION_APPROXIMATOR_N_HIDDEN_LAYERS
def universal_function_approximator_N_hidden_layers (input_vector, input_dim, hidden_dim, output_dim, num_hidden_layers, lower_bound, upper_bound):
        # First Hidden Layer
        weights_hidden_layer = tf.get_variable (name='weights_hidden_layer', 
            shape=[input_dim, hidden_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
        
        biases_hidden_layer = tf.get_variable (name='biases_hidden_layer', 
            shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0))
        
        z = tf.matmul (input_vector, weights_hidden_layer) + biases_hidden_layer
        activated_function = tf.nn.relu (z)

        #Second Hidden Layer
        weights_hidden_layer_1 = tf.get_variable (name='weights_hidden_layer_1', 
            shape=[hidden_dim, hidden_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
        
        biases_hidden_layer_1 = tf.get_variable (name='biases_hidden_layer_1', 
            shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0))
        
        z = tf.matmul (activated_function, weights_hidden_layer_1) + biases_hidden_layer_1
        activated_function = tf.nn.relu (z)

        #Output Layer
        weights_output_layer = tf.get_variable (name='weights_output_layer', 
            shape=[hidden_dim, output_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
            
        # See notebook #1 page 22
        z = tf.matmul (activated_function, weights_output_layer)
        
        return z
"""   

 
    
