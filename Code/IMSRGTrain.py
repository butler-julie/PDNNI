import numpy as np
from timeit import default_timer as timer
from NeuralNetwork import Train, Restore
import tensorflow as tf

#train = Train(2, 375, 1, 4161, 'data/imsrg_pm_s_set3.npy', 'data/imsrg_pm_set3.npy')


#i = 1000

#lr = 0.1 

#time = timer()
#train.train (i, lr)
#time = timer() - time

#print('Learning Rate; ', lr)
#print ('Iterations: ', i)
#print('Training Time: ', time)
#print ('Final Loss; ', train.get_loss())
#print()
#print()	

weights_name = 'IMSRGWB/imsrg_pm_set4_weights.npy'
biases_name = 'IMSRGWB/imsrg_pm_set4_biases.npy'

#train.save_weights(weights_name)
#train.save_biases(biases_name)

#train.graph_losses('imsrg_pm_set3_losses.png')

restore = Restore(weights_name, biases_name)

s = np.load('data/imsrg_heiko_s_new3.npy')
ys = np.load('data/imsrg_heiko_new3.npy')

s = np.reshape (s, (len(s), 1))

#time = timer()
#restore.batch_predict(s, False)
#time = timer() - time

#print ('Prediction Time: ', time)

restore.graph_mse(s, ys, 'imsrg_pm_mse.png')
