import numpy as np
import common as com

from QuasiNewton import QuasiNewton
from QuasiNewton import AlgorithmType

if __name__ == '__main__':
    x = np.array([0, 0]).reshape(2, 1)
    q = np.array([4, 2, 2, 2]).reshape(2, 2)
    b = np.array([-1, 1]).reshape(2, 1)
    threshold = np.zeros(x.shape)
    bfgs = QuasiNewton(x, com.create_unit_matrix(x.shape[0]), q, b, threshold, AlgorithmType.BFGS)
    print(bfgs.minimize())