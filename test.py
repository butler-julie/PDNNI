from UniversalApproximatorMatrices import UniversalTrainer as UA
from scipy.linalg import eigvalsh
import numpy as np

ua = UA(4, 250, 1.0, 1, 100, 'input.npy', 'test.npy')
ua.train(3000)
H0 = ua.get_first()
H0 = np.reshape(H0, (10, 10))
H0 = eigvalsh(H0)
print(ua.compare(9.9, H0))


