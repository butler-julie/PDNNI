from NeuralNetwork import Train
from timeit import default_timer as timer


train = Train (4, 250, 1, 400, 'ds1e-1.npy', 'SRGHO/HO_20_ds_1e-1.npy')

time = timer()
train.train (3000, 1.0 )
time = timer() - time

train.save_weights('HO_20_weights_1.npy')
train.save_biases('HO_20_biases_1.npy')
train.graph_losses('HO_20_losses_graph_1.png')

print ('Time: ', time)
print ('Final Loss: ', train.get_loss())


