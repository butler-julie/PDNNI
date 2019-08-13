from NeuralNetork import Trainer, Restore
from SRGNeuralNetwork import SRGNeuralNetwork

def main():
    print("Initializing Neural Network")
    srg_train = Trainer(1, 375, 0.1, 1, 36, 'input_pm_ds_1.npy', 'SRG_PM/SRG_PM_4_DS_1.npy')
    print ("Training and Saving Weights and Biases")
    srg_train.run(3000, 'results_pm/weights_4_1', 'results_pm/biases_4_1')
    w = srg_train.get_weights()
    print(len(w))
    print(srg_train.get_loss())
    H0 = np.load('SRG_PM/SRG_PM_4_DS_1e-1.npy')[0]
    H0 = np.reshape(H0, (6, 6))
    true = eigvalsh(H0)
    print ("Restoring Neural Network")
    srg = SRGNeuralNetwork('results_pm/weights_4_1.npy', 'results_pm/biases_4_1.npy')
    print(srg.compare_eigenvalues([[1.01], [5.01], [10.01], [100]], true))


if __name__=='__main__':
    main()
