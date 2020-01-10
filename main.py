import numpy as np
import common as com

from QuasiNewtonRankOne import RankOne

if __name__ == '__main__':
    q = np.array([2, 0, 0, 1]).reshape(2, 2)
    x = np.array([1, 2]).reshape(2, 1)
    threshold = np.zeros(x.shape)

    rankOne = RankOne(x, com.create_unit_matrix(x.shape[0]), q, threshold)
    print(rankOne.minimize())