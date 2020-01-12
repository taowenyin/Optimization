import numpy as np
from enum import IntEnum


# 算法类型
class AlgorithmType(IntEnum):
    RankOne = 0
    DFP = 1
    BFGS = 2


# 拟牛顿计算
class QuasiNewton:
    # x的值
    __x = 0
    # 黑塞矩阵的近似矩阵
    __h = 0
    # 二次型的黑塞矩阵Q
    __q = 0
    # 二次型的一次参数
    __b = 0
    # 阈值
    __threshold = 0
    # 算法类型
    __type = 0

    def __init__(self, x0, h0, q, b, threshold, type):
        self.__x = x0
        self.__h = h0
        self.__q = q
        self.__b = b
        self.__threshold = threshold
        self.__type = type

    # 得到目标函数的极小点
    def minimize(self):
        # 初始化迭代次数
        iteration_count = 1
        # S1：求得初始化时的梯度
        gradient = np.dot(self.__q, self.__x) - self.__b
        # S2：计算方向向量
        direction = -1 * np.dot(self.__h, gradient)
        # S3：计算步长
        alpha = -1 * (np.dot(gradient.T, direction) / np.dot(np.dot(direction.T, self.__q), direction))
        alpha = alpha.item((0, 0))
        # S4：计算新的x
        self.__x = self.__x + (alpha * direction)

        while iteration_count < self.__x.shape[0]:
            # S5：计算x的差值
            delta_x = alpha * direction
            # S6：计算新梯度
            old_gradient = gradient
            gradient = np.dot(self.__q, self.__x) - self.__b
            # S7：计算梯度差值
            delta_gradient = gradient - old_gradient
            # S8：计算H的更新值
            if self.__type == AlgorithmType.RankOne: # RankOne的H更新算法
                u = np.dot((delta_x - np.dot(self.__h, delta_gradient)),
                           (delta_x - np.dot(self.__h, delta_gradient)).T) / \
                    np.dot(delta_gradient.T, (delta_x - np.dot(self.__h, delta_gradient)))
            elif self.__type == AlgorithmType.DFP: # DFP的H更新算法
                u = np.dot(delta_x, delta_x.T) / np.dot(delta_x.T, delta_gradient) - \
                    np.dot(np.dot(self.__h, delta_gradient), (np.dot(self.__h, delta_gradient)).T) / \
                    np.dot(np.dot(delta_gradient.T, self.__h), delta_gradient)
            else: # BFGS的H更新算法
                u = np.dot(
                    (1 + np.dot(np.dot(delta_gradient.T, self.__h), delta_gradient) / np.dot(delta_gradient.T, delta_x)).item((0, 0)),
                    np.dot(delta_x, delta_x.T) / np.dot(delta_x.T, delta_gradient)) - \
                    ((np.dot(np.dot(delta_x, delta_gradient.T), self.__h) + np.dot(np.dot(self.__h, delta_gradient), delta_x.T)) /
                     np.dot(delta_gradient.T, delta_x))

            # S9：计算新的H
            self.__h = self.__h + u
            # S10：计算新的方向向量
            direction = -1 * np.dot(self.__h, gradient)
            # S11：计算步长
            alpha = -1 * (np.dot(gradient.T, direction) / np.dot(np.dot(direction.T, self.__q), direction))
            alpha = alpha.item((0, 0))
            # S12：计算新的x
            self.__x = self.__x + (alpha * direction)
            # 迭代次数+1
            iteration_count += 1

        return self.__x
