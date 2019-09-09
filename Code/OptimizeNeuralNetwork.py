#################################################
# Optimize Neural Network
# Julie Hartley
# September 9, 2019
# Version 0.0.0
#
# Finds the optimal parameters of the neural network for a new set fo trianing data.
#################################################


#################################################
# OUTLINE
#
# Optimize
#################################################

#############################
# IMPORTS
#############################
# Local Imports
from NeuralNetwork import Train, Restore

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class Optimize:

    def __init__(self, hidden_layer_range, hidden_neuron_range, learning_rate_range, 
        iteration_range, input_dim, output_dim, input_data, output_data):
        self.hidden_layer_range = hidden_layer_range
        self.hidden_neuron_range = hidden_neuron_range
        self.learning_rate_range = learning_rate_range
        self.iteration_range = iteration_range

    def optimize (self, subset_percentage, isGraph=True):
        for hl in self.hidden_layer_range:
            for hn in self.hidden_neuron_range:
                for lr in self.learning_rate_range:
                    for i in iteration_range:
                        tf.reset_default_graph()
                        train = Train(hl, hn, 
	
