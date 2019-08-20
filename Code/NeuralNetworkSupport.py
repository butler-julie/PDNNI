########################################################################################
# NeuralNetworkFunctions.py
# Julie Butler
# February 20, 2019
# Version 3.0.0
#
# A neural network created in Tensorflow.  It can have any number of hidden layers but 
# each hidden layer must have the same number of neurons
########################################################################################

########################################################################################
# OUTLINE:
# neural_network (input_vector, input_dim, hidden_dim, output_dim, num_hidden_layers):
# Initializes a neural network using Tensorflow and then is used for the training process.
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

# NEURAL_NETWORK
def neural_network (input_vector, input_dim, hidden_dim, output_dim, num_hidden_layers):
    """
        Inputs:
            input_vector (a Tensor): the input value of the neural netowrk
            input_dim (an int): the dimension of the input
            hidden_dim (an int): the number of neurons per hidden layer (must be the same per 
                hidden layer)
            output_dim (an int): the dimension of the output of the neural network
            num_hidden_layers (an int): the number of hidden layers
        Returns:
            z (an Tensor): the output of the neural network
        Initializes a neural network using Tensorflow and then is used for the training process.
    """
    # First Hidden Layer
    # weights of first hidden layer (initialized to random numbers at the beginning)
    # Names of the weights and biases are important for lookup later
    weights_first_hidden_layer = tf.get_variable (name='weights_1', 
        shape=[input_dim, hidden_dim],
        initializer=tf.random_normal_initializer(stddev=0.1))
    # biases of the first hidden layer (initialized to zero at the beginning)
    biases_first_hidden_layer = tf.get_variable (name='biases_1', 
        shape=[hidden_dim],
        initializer=tf.constant_initializer(0.0))
        
    # multiply input vector by the weights and add the biases
    # matmul = Matrix Multiplication
    z = tf.matmul (input_vector, weights_first_hidden_layer) + biases_first_hidden_layer
    # apply the activation function (relu(x) = max(0,x)
    activated_function = tf.nn.relu (z)

    # Do the same thing for each of the other hidden layers
    for i in range (1, num_hidden_layers):
        # Interior Hidden Layers
        weight_name = 'weights_' + str(i+1)
        bias_name = 'biases_' + str(i+1)

        weights_hidden_layer_1 = tf.get_variable (name=weight_name, 
            shape=[hidden_dim, hidden_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
        
        biases_hidden_layer_1 = tf.get_variable (name=bias_name, 
            shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0))
        
        z = tf.matmul (activated_function, weights_hidden_layer_1) + biases_hidden_layer_1
        activated_function = tf.nn.relu (z)

    # Output Layer
    # Output layer does not have biases or activation function by design
    weights_output_layer = tf.get_variable (name='weights_output_layer', 
        shape=[hidden_dim, output_dim],
        initializer=tf.random_normal_initializer(stddev=0.1))
            
    # See notebook #1 page 22
    z = tf.matmul (activated_function, weights_output_layer)
        
    return z
