##################################################
# Main Interactive
# Julie Hartley
# Version 0.0.0
# August 13, 2019
#
# An interactive way to use PDNNI
##################################################

import tkinter as tk
from NeuralNetwork import Trainer, Restore

def command_line():
  input_file = str(input("Where is the input data located?"))
  input_dim = int(input("What is the dimension of the input data?"))
  
  training_file = str(input("Where is the training data located?"))
  output_dim = str(input("What is the dimension of the training data?"))
  
  hidden_layers = int(input("How many hidden layers?"))
  hidden_neurons = int(input("How many neurons per hidden layer?"))
  learning_rate = float(input("What is the learning rate for the training?"))
  
  isSavedLosses = bool(input("Save all training losses?"))
  
  print ("Setting up the neural network.")
  
  train = Trainer (hidden_layers, hidden_neurons, learning_rate, input_dim, output_dim, input_file, training_file, isSavedLosses)
  
  iterations = int(input("Number of training iterations?"))
  
  print ("Training the neural network.")
  
  train.train(iterations)
  
  loss = train.get_loss()
  
  print ("Neural network saved with final loss of ", loss)
  
  
  
  
  
def gui ():  
  print ("GUI")
