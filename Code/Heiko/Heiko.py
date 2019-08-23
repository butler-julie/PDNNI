
import numpy as np
import timeit
import matplotlib.pyplot as plt
from NeuralNetwork import Trainer, Restore

def train():
    train = Trainer(1, 250, 0.01, 1, 4161, 'Heiko/heiko_train_s_1.npy', 'Heiko/heiko_train_1.npy')
    start = timeit.default_timer()
    train.train_and_save(3000, 'heiko_weights_2', 'heiko_biases_2')
    print('Time: ', timeit.default_timer() - start)
    print('Loss:', train.get_loss())

def predict():
    srg_predict = Restore ('heiko_weights_2.npy', 'heiko_biases_2.npy')
    true = np.load('Heiko/imsrg_heiko.npy')
    prediction_values = np.load('Heiko/imsrg_heiko_s.npy')
    prediction_values = np.reshape (prediction_values, (len(prediction_values), 1))
    l1, l2 = srg_predict.compare_to_true (prediction_values, true)
    print ("L1 Average: ", np.average(l1))
    print ("L2 Average: ", np.average(l2))


if __name__=='__main__':
   predict()
