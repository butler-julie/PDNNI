############################################################
# Yanni
# Julie and Yanni
# Version 0.0.0
# August 14, 2019
#
# 
############################################################

############################################################
############################################################

##############################
# IMPORTS
##############################
# THIRD PARTY
import timeit
import numpy as np
import matplotlib.pyplot as plt
# LOCAL
from NeuralNetwork import Trainer, Restore
from SRGNeuralNetwork import SRGNeuralNetwork as SRG



def train ():
    srg_train = Trainer(2, 500, 0.025, 1, 2229, 'yanni_flow.npy', 'yanni_matrix.npy')
    start = timeit.default_timer()
    srg_train.train_and_save(3000, 'yanni_weights_13', 'yanni_biases_13')
    print('Time: ', timeit.default_timer() - start)
    print('Loss: ', srg_train.get_loss())

def predict ():
    srg_predict = Restore ('yanni_13/yanni_weights_13.npy', 'yanni_13/yanni_biases_13.npy')
    true = np.load('yanni_matrix_1e-2.npy')
    true1 = np.load('yanni_matrix.npy')
    prediction_values = np.arange(0, 6.5, 0.01)
    prediction_values = np.reshape (prediction_values, (len(prediction_values), 1))
    predict = srg_predict.predict([0.1])
    diff = np.subtract(true1[0], true[0])
    for i in diff:
        print(i)    

    #l1, l2 = srg_predict.compare_to_true (prediction_values, true)
    #print ("L1 Average: ", np.average(l1))
    #print ("L2 Average: ", np.average(l2))


if __name__ == '__main__':
    predict()
