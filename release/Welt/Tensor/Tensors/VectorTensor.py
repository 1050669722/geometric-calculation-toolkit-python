# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/22 下午4:12
# @Function:
# @Refer:

import numpy as np

from typing import Tuple

from Welt.Utils import Utils
from Welt.Tensor.Tensors.LineSegTensor import LineSegTensor


class VectorTensor(LineSegTensor):
    """
    向量张量
    存储向量信息的张量
    """
    def __init__(self, lineSegs: np.ndarray):
        """
        向量张量初始化
        Args:
            lineSegs: 容器 存储 向量
        """
        # 超类初始化方法 初始化
        super(VectorTensor, self).__init__(lineSegs)

        # 更新数据
        VectorTensor.update(self)


    def update(self) -> None:
        """
        更新数据
        Returns: None
        """
        # 定义属性张量 横坐标 纵坐标 张量
        Xs = self.endXs - self.startXs
        Ys = self.endYs - self.startYs

        # 在 axis=2 方向上 扩展维度
        Xs = np.expand_dims(Xs, axis=2)
        Ys = np.expand_dims(Ys, axis=2)

        # 在 axis=2 方向上 拼接 张量 #定义属性张量vectors
        self.vectors = np.concatenate((Xs, Ys), axis=2)

        # 定义属性张量Xs, Ys
        self.Xs = self.vectors[:, :, 0]
        self.Ys = self.vectors[:, :, 1]

        # 获取 中点（形心）
        self.__getCentroid()

        # 获取 模长
        self.__getNorm()

        # 获取 两个单位垂直向量
        self.__getUnitVerticalVecs()


    def __add__(self, vectors: np.ndarray) -> np.ndarray:
        """
        计算 向量张量的vectors 经过 相加运算 后的 vectors
        Args:
            vectors: 待加vectors
        Returns: 向量张量的vectors 经过 相加运算 后的 vectors
        """
        # 断言 形状相同
        assert vectors.shape == self.vectors.shape

        # 返回 相加结果
        return vectors + self.vectors


    def scalarValue(self, scaleFactorTensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 向量张量的vectors 经过 数乘运算 后的 vectors
        Args:
            scaleFactorTensor: 数乘因子张量
        Returns: 向量张量的vectors 经过 数乘运算 后的 vectors
        """
        # 断言 形状相同
        assert scaleFactorTensor.shape == self.vectors.shape

        # 返回 数乘结果
        return scaleFactorTensor * self.vectors


    def dot(self, vectorTensor) -> np.ndarray:
        """
        计算 点积
        Args:
            vectorTensor: 另一个向量张量
        Returns: 点积 张量
        """
        # x1 * x2 + y1 * y2
        return self.Xs * vectorTensor.Xs + self.Ys * vectorTensor.Ys


    def crossValue(self, vectorTensor) -> np.ndarray:
        """
        计算 叉积
        Args:
            vectorTensor: 另一个向量张量
        Returns: 叉积 张量
        """
        # x1 * y2 - x2 * y1
        return self.Xs * vectorTensor.Ys - vectorTensor.Xs * self.Ys


    @staticmethod
    def generateVectorTensorFromStartsAndEnds(startXs: np.ndarray, startYs: np.ndarray, endXs: np.ndarray, endYs: np.ndarray):
        """
        根据 起点 终点 横坐标 纵坐标 张量 产生 向量张量
        Args:
            startXs: 起点 横坐标 张量
            startYs: 起点 纵坐标 张量
            endXs: 终点 横坐标 张量
            endYs: 终点 纵坐标 张量
        Returns: 产生的 向量张量
        """
        # 限制数据形状
        assert len(startXs.shape) == 2 and len(startYs.shape) == 2 and len(endXs.shape) == 2 and len(endYs.shape) == 2
        assert startXs.shape == startYs.shape == endXs.shape == endYs.shape

        # 起点 终点 横坐标 纵坐标 张量 在 axis=2 方向上 扩展维度
        startXs = np.expand_dims(startXs, axis=2)
        startYs = np.expand_dims(startYs, axis=2)
        endXs = np.expand_dims(endXs, axis=2)
        endYs = np.expand_dims(endYs, axis=2)

        # 返回
        return VectorTensor(np.concatenate((startXs, startYs, endXs, endYs), axis=2))


    def __getCentroid(self) -> np.ndarray:
        """
        计算 形心张量
        Returns: 形心张量
        """
        # 形心
        centroidXs = 0.5 * (self.startXs + self.endXs)
        centroidYs = 0.5 * (self.startYs + self.endYs)

        # 拼接
        self.centroid = np.concatenate((centroidXs, centroidYs), axis=1)

        # 返回
        return self.centroid


    def __getNorm(self) -> np.ndarray:
        """
        计算 模长
        Returns: 模长张量
        """
        # 计算 模长
        self.norm = self._calcLengths()

        # 返回
        return self.norm


    def __getUnitVerticalVecs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取单位垂直向量
        Returns: 元组 (单位垂直向量0, 单位垂直向量1)
        """
        # 垂直向量张量0
        # self.unitVerticalVecs0 = np.concatenate((self.Ys, -self.Xs), axis=1)
        self.unitVerticalVecs0 = np.concatenate((np.expand_dims(self.Ys, axis=2), np.expand_dims(-self.Xs, axis=2)), axis=2)

        # 模长矩阵0
        # normMat0 = np.linalg.norm(self.unitVerticalVecs0, axis=1, keepdims=True)
        # normMat0[normMat0 < 1e-4 * Utils.Threshold] = 1e-4 * Utils.Threshold
        normMat0 = np.linalg.norm(self.unitVerticalVecs0, axis=2, keepdims=True)
        normMat0[normMat0 < 1e-4 * Utils.Threshold] = 1e-4 * Utils.Threshold

        # 单位化0
        self.unitVerticalVecs0 = np.divide(self.unitVerticalVecs0, normMat0)

        # 垂直向量张量1
        # self.unitVerticalVecs1 = np.concatenate((-self.Ys, self.Xs), axis=1)
        self.unitVerticalVecs1 = np.concatenate((np.expand_dims(-self.Ys, axis=2), np.expand_dims(self.Xs, axis=2)), axis=2)

        # 模长矩阵1
        # normMat1 = np.linalg.norm(self.unitVerticalVecs1, axis=1, keepdims=True)
        # normMat1[normMat1 < 1e-4 * Utils.Threshold] = 1e-4 * Utils.Threshold
        normMat1 = np.linalg.norm(self.unitVerticalVecs1, axis=2, keepdims=True)
        normMat1[normMat1 < 1e-4 * Utils.Threshold] = 1e-4 * Utils.Threshold

        # 单位化1
        self.unitVerticalVecs1 = np.divide(self.unitVerticalVecs1, normMat1)

        # 返回
        return self.unitVerticalVecs0, self.unitVerticalVecs1

    pass




if __name__ == '__main__':
    from Welt.Tensor.Converters.LineSegTensorConverter import LineSegTensorConverter

    # 向量
    vectors = [
        [[0, 0], [0, 1]],
        [[1, 1], [2, 2]],
        [[-1, -1], [-2, -3]],
    ]

    # 转换
    vectors = LineSegTensorConverter.convert(vectors)

    # 模拟tile过程
    vectors = np.reshape(vectors, (-1, 1, 4))

    # 实例化 向量张量
    vectorTensor = VectorTensor(vectors)

    # 打印
    print(vectors)

    # ### 绘图 #########################################
    from matplotlib import pyplot as plt

    # 原向量
    for vector in vectors:
        xs = [vector[0][0], vector[0][2]]
        ys = [vector[0][1], vector[0][3]]
        plt.plot(xs, ys, color='b')
        plt.annotate(
            text="",
            xy=(xs[1], ys[1]),
            xytext=(xs[0], ys[0]),
            arrowprops=dict(arrowstyle="->", color='b')
        )

    # 垂直向量
    for vector, unitVerticalVec0 in zip(vectors, vectorTensor.unitVerticalVecs0):
        # xs = [0.0 + vector[0][0], unitVerticalVec0[0] + vector[0][0]]
        # ys = [0.0 + vector[0][1], unitVerticalVec0[1] + vector[0][1]]
        xs = [0.0 + vector[0][0], unitVerticalVec0[0][0] + vector[0][0]]
        ys = [0.0 + vector[0][1], unitVerticalVec0[0][1] + vector[0][1]]
        plt.plot(xs, ys, color='r')
        plt.annotate(
            text="",
            xy=(xs[1], ys[1]),
            xytext=(xs[0], ys[0]),
            arrowprops=dict(arrowstyle="->", color='r')
        )
    for vector, unitVerticalVec1 in zip(vectors, vectorTensor.unitVerticalVecs1):
        # xs = [0.0 + vector[0][0], unitVerticalVec1[0] + vector[0][0]]
        # ys = [0.0 + vector[0][1], unitVerticalVec1[1] + vector[0][1]]
        xs = [0.0 + vector[0][0], unitVerticalVec1[0][0] + vector[0][0]]
        ys = [0.0 + vector[0][1], unitVerticalVec1[0][1] + vector[0][1]]
        plt.plot(xs, ys, color='r')
        plt.annotate(
            text="",
            xy=(xs[1], ys[1]),
            xytext=(xs[0], ys[0]),
            arrowprops=dict(arrowstyle="->", color='r')
        )

    # 展示
    plt.axis("equal")
    plt.show()
    # #################################################
