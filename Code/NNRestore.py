from NeuralNetwork import Restore
from timeit import default_timer as timer
import numpy as np

restore = Restore('HOWB/HO_252_weights_1.npy', 'HOWB/HO_252_biases_1.npy')

prediction_values = np.load('ds1e-2.npy')
prediction_values = np.reshape (prediction_values, (len(prediction_values), 1))

time = timer()
restore.batch_predict (prediction_values, False)
time = timer() - time
print ('Time: ', time)

true_results = np.load('SRGHO/HO_252_ds_1e-2.npy')

restore.error_analysis(prediction_values, true_results, 'HO_252_')
