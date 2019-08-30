import numpy as np
from timeit import default_timer as timer
from NeuralNetwork import Train, Restore
import tensorflow as tf

train = Train(2, 500, 1, 2229, 'yanni_1e-3_s_subset4.npy', 'yanni_1e-3_subset4.npy')


i = 3000

lr = 0.05 

time = timer()
train.train (i, lr)
time = timer() - time

print('Learning Rate; ', lr)
print ('Iterations: ', i)
print('Training Time: ', time)
print ('Final Loss; ', train.get_loss())
#print()
#print()	

weights_name = 'yanni_1e-3_weights4.npy'
biases_name = 'yanni_1e-3_biases4.npy'

train.save_weights(weights_name)
train.save_biases(biases_name)

#train.graph_losses('imsrg_pm_set3_losses.png')

restore = Restore(weights_name, biases_name)

s = np.load('Yanni/yanni_flow_1e-3.npy')
ys = np.load('Yanni/yanni_matrix_1e-3.npy')

s = np.reshape (s, (len(s), 1))

time = timer()
restore.batch_predict(s, False)
time = timer() - time

print ('Prediction Time: ', time)

mae, mse = restore.average_mae_and_mse (s, ys)

print ('Average MSE: ', mse)

#restore.graph_mse(s, ys, 'imsrg_pm_mse.png')
