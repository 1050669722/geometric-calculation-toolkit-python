# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/15 11:37
# @Function：
# @Refer：

import numpy as np

from typing import Tuple

from Welt.Utils import Utils
from Welt.Tensor.Tensors.Tensor import Tensor
from Welt.Tensor.Tensors.PointTensor import PointTensor
from Welt.Tensor.Tensors.VectorTensor import VectorTensor
from Welt.Tensor.Executors.Calculator import Calculator
from Welt.Tensor.Tilers.PointTensorPointTensorTiler import PointTensorPointTensorTiler
from Welt.Tensor.Tilers.LineSegTensorLineSegTensorTiler import LineSegTensorLineSegTensorTiler
from Welt.Tensor.Converters.PointTensorConverter import PointTensorConverter


class DotProdMatFromThreePointTensorsCalculator(Calculator):
    """
    计算 三堆点 形成的 点积矩阵
    其中，一堆点作为起点集，另外两堆点作为终点集

    pointsA: 点集A，含有n个点 | endPoints
    pointsB: 点集B，含有m个点 | startPoints
    pointsP: 点集P，含有p个点 | markPoints

    转换形状
    (m * n, 1, 4)
    (m * p, 1, 4)

    再形成 两个 向量 张量
    (m * n, 2, 2)
    (m * p, 2, 2)

    最终获得 点积 张量
    (m * n, m * p)
    """
    def __init__(self, pointsA: np.ndarray, pointsB: np.ndarray, pointsP: np.ndarray):
        """
        计算器实例化
        Args:
            pointsA: 点数组A
            pointsB: 点数组B
            pointsP: 点数组P
        """
        # 属性赋值
        self.pointsA = pointsA
        self.pointsB = pointsB
        self.pointsP = pointsP

        pass


    def __convertRawData(self, pointsM: np.ndarray, pointsN: np.ndarray) -> Tuple[Tensor, Tensor]:
        """
        点数组 转换为 点张量
        (n, 1, 1), (m, 1, 1) -> (n, m, 1), (n, m, 1)
        Args:
            pointsM: 点数组M
            pointsN: 点数组N
        Returns: 元组 (点行张量, 点列张量)
        """
        # 实例化 PointTensorPointTensorTiler
        pointTensorPointTensorTiler = PointTensorPointTensorTiler(pointsM, pointsN)

        # 返回
        return pointTensorPointTensorTiler.pointTensorRow, pointTensorPointTensorTiler.pointTensorColumn


    def __getVectorTensorEntitiesFromTwoPointTensors(self, pointsM: np.ndarray, pointsN: np.ndarray) -> np.ndarray:
        """
        获取 向量张量实体 从 两个点张量中
        Args:
            pointsM: 点数组M
            pointsN: 点数组N
        Returns: 线段张量实体
        """
        # 点行张量，点列张量
        pointsMTensorRow, pointsNTensorColumn = self.__convertRawData(pointsM, pointsN)

        # 断言 它们都是点张量
        assert isinstance(pointsMTensorRow, PointTensor) and isinstance(pointsNTensorColumn, PointTensor)

        # 两个点张量 两两之间 产生的 向量张量 | 形状：(M, N, 4)
        vectorTensorMN = VectorTensor.generateVectorTensorFromStartsAndEnds(pointsMTensorRow.Xs, pointsMTensorRow.Ys, pointsNTensorColumn.Xs, pointsNTensorColumn.Ys)

        # 向量张量实体 改变形状至 (M * N, 2, 2)
        reshapedEntitiesMN = vectorTensorMN.entities.reshape((-1, 1, 4))
        reshapedEntitiesMN = reshapedEntitiesMN.reshape((-1, 2, 2))

        # 返回
        return reshapedEntitiesMN

    def calcDotProdMatFromThreePointTensors(self, whetherToNormalize: bool=False) -> np.ndarray:
        """
        计算 点积矩阵 来自于 三个点张量
        whetherToNormalize: 是否 按照向量张量BP的模长 对 结果 进行归一化
        Returns: 点积矩阵
        """
        # 向量张量实体BA | (m * n, 2, 2)
        lineSegTensorEntitiesBA = self.__getVectorTensorEntitiesFromTwoPointTensors(self.pointsB, self.pointsA)

        # 向量张量实体BP | (m * p, 2, 2)
        lineSegTensorEntitiesBP = self.__getVectorTensorEntitiesFromTwoPointTensors(self.pointsB, self.pointsP)

        # 采用线段张量铺开器 将 向量张量 铺开 | (m * n, m * p, 4), (m * n, m * p, 4)
        lineSegTensorLineSegTensorTiler = LineSegTensorLineSegTensorTiler(lineSegTensorEntitiesBA,
                                                                          lineSegTensorEntitiesBP)
        LTBA, LTBP = lineSegTensorLineSegTensorTiler.lineSegTensorRow, lineSegTensorLineSegTensorTiler.lineSegTensorColumn

        # 线段张量 转换为 向量张量 | (m * n, m * p, 4), (m * n, m * p, 4)
        vecTBA = VectorTensor(LTBA.entities)
        vecTBP = VectorTensor(LTBP.entities)

        # 计算 张量积矩阵 | (m * n, m * p)
        dotProd = vecTBA.dot(vecTBP)

        # 如果 对结果 进行归一化
        if whetherToNormalize:
            norm = vecTBP.norm
            norm[norm < 1e-4 * Utils.Threshold] = 1e-4 * Utils.Threshold
            dotProd = np.divide(dotProd, norm)

        # 返回
        return dotProd

    pass




if __name__ == '__main__':
    # 点
    pointsA = [
        [-1.3, 2],
        [-0.6, 0.5],
        [0.7, 1.5],
        [0.5, 2.5],
        [-1, -1],
        [-1.5, -0.6],
    ]
    pointsB = [
        [2, -1],
        [2, -2]
    ]
    pointsP = [
        [4, 3],
        [4, 4],
        [4, 5]
    ]

    # 转换
    arrayPointsA = PointTensorConverter.convert(pointsA)
    arrayPointsB = PointTensorConverter.convert(pointsB)
    arrayPointsP = PointTensorConverter.convert(pointsP)

    # 实例化
    dotProdTensorFromThreePointTensorsCalculator = DotProdMatFromThreePointTensorsCalculator(arrayPointsA, arrayPointsB, arrayPointsP)

    # 计算
    dotProdMat = dotProdTensorFromThreePointTensorsCalculator.calcDotProdMatFromThreePointTensors()
    print(dotProdMat, end='\n\n')


    # ### 数量积（验证） ################################
    vecsBA, vecsBP = [], []

    for pntB in pointsB:
        for pntA in pointsA:
            vecBA = np.array([pntA[0] - pntB[0], pntA[1] - pntB[1]])
            vecsBA.append(vecBA)

    for pntB in pointsB:
        for pntP in pointsP:
            vecBP = np.array([pntP[0] - pntB[0], pntP[1] - pntB[1]])
            vecsBP.append(vecBP)

    for vecBA in vecsBA:
        for vecBP in vecsBP:
            print(np.dot(vecBA, vecBP))
        print()
    # #################################################


    # ### 绘图逻辑 #####################################
    from matplotlib import pyplot as plt

    xs, ys = [], []
    for pnt in pointsA:
        xs.append(pnt[0])
        ys.append(pnt[1])
    plt.scatter(xs, ys, color='b')

    xs, ys = [], []
    for pnt in pointsB:
        xs.append(pnt[0])
        ys.append(pnt[1])
    plt.scatter(xs, ys, color='k')

    xs, ys = [], []
    for pnt in pointsP:
        xs.append(pnt[0])
        ys.append(pnt[1])
    plt.scatter(xs, ys, color='r')

    # 设置轴 与 展示
    plt.axis("equal")
    plt.show()
    # #################################################

