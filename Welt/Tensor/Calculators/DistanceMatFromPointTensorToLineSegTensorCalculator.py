# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/19 9:21
# @Function：
# @Refer：

import numpy as np

from typing import Tuple

from Welt.Utils import Utils
from Welt.Tensor.Executors.Calculator import Calculator
from Welt.Tensor.Tensors.VectorTensor import VectorTensor
from Welt.Tensor.Generators.SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator import SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator
from Welt.Tensor.Converters.PointTensorConverter import PointTensorConverter
from Welt.Tensor.Converters.LineSegTensorConverter import LineSegTensorConverter


class DistanceMatFromPointTensorToLineSegTensorCalculator(Calculator):
    """
    点张量 中的 点 与 线段张量 中的 线段所在直线的 欧式距离 的 计算器
    """
    def __init__(self, points: np.ndarray, lineSegs: np.ndarray):
        """
        初始化 计算器
        Args:
            points: 点数组 | (m, 1, 2)
            lineSegs: 线段数组 | (n, 2, 2)
        """
        # 获取 起点与点向量张量 线段向量张量 #(m, n, 4), (m, n, 4)
        self.spVectorTensor, self.lineSegVectorTensor = self.__convertRawData(points, lineSegs)


    def __convertRawData(self, points: np.ndarray, lineSegs: np.ndarray) -> Tuple[VectorTensor, VectorTensor]:
        """
        获取 起点与点向量张量 线段向量张量
        Args:
            points: 点数组 | (m, 1, 2)
            lineSegs: 线段数组 | (n, 2, 2)
        Returns: 元组 (起点与点向量张量, 线段向量张量)
        """
        # 实例化 起点与点向量张量-线段向量张量生成器
        spVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator = SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator(points, lineSegs)

        # 生成张量
        spVectorTensor, lineSegVectorTensor = spVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator.generateSpVectorTensorAndLineSegVectorTensor()

        # 返回
        return spVectorTensor, lineSegVectorTensor


    def calcDistanceMatFromPointTensorToLineSegTensor(self) -> np.ndarray:
        """
        计算 距离矩阵 从 点张量 到 线段张量
        Returns: 距离矩阵
        """
        # 叉积数值矩阵 #(m, n)
        crossValueMat = self.spVectorTensor.crossValue(self.lineSegVectorTensor)

        # 线段向量张量 的 模长矩阵
        normMat = self.lineSegVectorTensor.norm

        # 处理模长很小的情况
        threshold = 1e-4 * Utils.Threshold #1e-6 #
        normMat[normMat < threshold] = threshold

        # 点 到 线段所在直线的 距离矩阵
        distanceMat = np.divide(crossValueMat, normMat)

        # 距离 应取 绝对值
        distanceMat = np.abs(distanceMat)

        # 返回
        return distanceMat

    pass




if __name__ == '__main__':
    # 点
    points = [
        [0, 0],
        [1, 1],
        [2, 1],
        [2, 0],
    ]

    # 线段
    lineSegs = [
        [[-1, 0], [0, 1]],
        [[-2, 0], [2, 2]],
        [[-3, -1], [-4, -4]],
    ]

    # 转换
    arrayPoints = PointTensorConverter.convert(points)
    arrayLineSegs = LineSegTensorConverter.convert(lineSegs)

    # 实例化
    distanceMatFromPointTensorToLineSegTensorCalculator = DistanceMatFromPointTensorToLineSegTensorCalculator(arrayPoints, arrayLineSegs)

    # 计算
    distanceMat = distanceMatFromPointTensorToLineSegTensorCalculator.calcDistanceMatFromPointTensorToLineSegTensor()
    print(distanceMat, end='\n\n')


    # ### 验证逻辑 #################################################
    from Welt.Structs.StructsDoubleParticles.LineSeg.LineSeg import LineSeg
    from Welt.Geometry.GeneralAlgos import GeneralAlgos

    distances = []
    for point in points:
        for lineSeg in lineSegs:
            distance = GeneralAlgos.calcDistanceFromPointToLine(point, LineSeg(lineSeg))
            distances.append(distance)

    print(distances)
    # ############################################################


    # ### 绘图逻辑 #################################################
    from matplotlib import pyplot as plt

    # 点
    xs, ys = [], []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
    plt.scatter(xs, ys, color='r')

    # 线段
    for lineSeg in lineSegs:
        xs = [lineSeg[0][0], lineSeg[1][0]]
        ys = [lineSeg[0][1], lineSeg[1][1]]
        plt.plot(xs, ys, color='b')

    # 展示
    plt.axis("equal")
    plt.grid()
    plt.show()
    # ############################################################
