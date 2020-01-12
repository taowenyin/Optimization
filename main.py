import numpy as np
import common as com

from QuasiNewton import QuasiNewton
from QuasiNewton import AlgorithmType

if __name__ == '__main__':
    x_1 = np.array([0, 0]).reshape(2, 1)
    q_1 = np.array([5, -3, -3, 2]).reshape(2, 2)
    b_1 = np.array([0, 1]).reshape(2, 1)
    threshold_1 = np.zeros(x_1.shape)
    # 秩一法解决例11.4
    rankOne = QuasiNewton(x_1, com.create_unit_matrix(x_1.shape[0]), q_1, b_1, threshold_1, AlgorithmType.RankOne)

    x_2 = np.array([0, 0]).reshape(2, 1)
    q_2 = np.array([4, 2, 2, 2]).reshape(2, 2)
    b_2 = np.array([-1, 1]).reshape(2, 1)
    threshold_2 = np.zeros(x_2.shape)
    # DFP解决例11.3
    dfp = QuasiNewton(x_2, com.create_unit_matrix(x_2.shape[0]), q_2, b_2, threshold_2, AlgorithmType.DFP)
    # BFGS解决例11.3
    bfgs = QuasiNewton(x_2, com.create_unit_matrix(x_2.shape[0]), q_2, b_2, threshold_2, AlgorithmType.BFGS)

    print("Rank One算法结果：")
    print(rankOne.minimize().T)
    print("DFP算法结果：")
    print(dfp.minimize().T)
    print("BFGS算法结果：")
    print(bfgs.minimize().T)
