import common as com
import numpy as np

# 秩一法计算
class RankOne:
    # x的值
    __x = 0
    # 黑塞矩阵的近似矩阵
    __h = 0
    # 二次型的黑塞矩阵Q
    __q = 0
    __threshold = 0

    def __init__(self, x0, h0, q, threshold):
        self.__x = x0
        self.__h = h0
        self.__q = q
        self.__threshold = threshold

    # 得到目标函数的极小点
    def minimize(self):
        # S1：求得初始化时的梯度
        gradient = np.dot(self.__q, self.__x)
        # S2：计算方向向量
        direction = -1 * np.dot(self.__h, gradient)
        # S3：计算步长
        alpha = -1 * (np.dot(gradient.T, direction) / np.dot(np.dot(direction.T, self.__q), direction))
        alpha = alpha.item((0, 0))
        # S4：计算新的x
        self.__x = self.__x + (alpha * direction)

        while (gradient > self.__threshold).all():
            # S5：计算x的差值
            delta_x = alpha * direction
            # S6：计算新梯度
            old_gradient = gradient
            gradient = np.dot(self.__q, self.__x)
            # S7：计算梯度差值
            delta_gradient = gradient - old_gradient
            # S8：计算H的更新值
            u = np.dot((delta_x - np.dot(self.__h, delta_gradient)), (delta_x - np.dot(self.__h, delta_gradient)).T) / \
                np.dot(delta_gradient.T, (delta_x - np.dot(self.__h, delta_gradient)))
            # S9：计算新的H
            self.__h = self.__h + u
            # S10：计算新的方向向量
            direction = -1 * np.dot(self.__h, gradient)
            # S11：计算步长
            alpha = -1 * (np.dot(gradient.T, direction) / np.dot(np.dot(direction.T, self.__q), direction))
            alpha = alpha.item((0, 0))
            # S12：计算新的x
            self.__x = self.__x + (alpha * direction)

        return self.__x
