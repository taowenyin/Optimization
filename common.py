import numpy as np

# 创建单位矩阵
def create_unit_matrix(size):
    return np.mat(np.identity(size))


# 计算梯度
def rank_one_gradient(q, x):
    return np.dot(q, x)


# 计算方向
def rank_one_direction(h, g):
    return -1 * np.dot(h, g)


# 计算步长
def rank_one_step(g, d, q):
    step = -1 * (np.dot(g.T, d) / np.dot(np.dot(d.T, q), d))
    return step[0, 0]


# 计算x
def rank_one_x(x, step, d):
    return x + step * d