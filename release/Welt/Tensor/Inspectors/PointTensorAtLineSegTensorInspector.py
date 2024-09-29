# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/31 上午10:12
# @Function:
# @Refer:

import numpy as np

# from typing import Tuple

from Welt.Utils import Utils
# from Welt.Tensor.Tensors.PointTensor import PointTensor
# from Welt.Tensor.Tensors.LineSegTensor import LineSegTensor
from Welt.Tensor.Tensors.VectorTensor import VectorTensor
from Welt.Tensor.Tilers.PointTensorLineSegTensorTiler import PointTensorLineSegTensorTiler


class PointTensorAtLineSegTensorInspector(PointTensorLineSegTensorTiler):
    """
    点张量 中的 每一个点 是否位于 线段张量 中的 每一个线段 的 判定器
    """
    def __init__(self, points: np.ndarray, lineSegs: np.ndarray):
        """
        判定器初始化
        Args:
            points: 容器 存储 点
            lineSegs: 容器 存储 线段
        """
        # 采用父类初始化方法 初始化
        super(PointTensorAtLineSegTensorInspector, self).__init__(points, lineSegs)


    def __calcIsPointTensorIncludedInLineSegTensor(self, isLoose: bool, threshold: float=1e2 * Utils.Threshold) -> bool:
        """
        计算 是否重叠矩阵 这一矩阵记录着 点张量中的点是否位于以线段张量中的线段为对角线的矩形内
        Returns: 是否存在重叠部分的矩阵
        """
        # 矩阵 存储 判断结果 点是否位于以线段为对角线的矩形内
        if isLoose:
            includedMat = ((self.pointTensorRow.Xs > self.lineSegTensorColumn.xmins) + (np.abs(self.pointTensorRow.Xs - self.lineSegTensorColumn.xmins) <= threshold)) * \
                          ((self.pointTensorRow.Xs < self.lineSegTensorColumn.xmaxs) + (np.abs(self.pointTensorRow.Xs - self.lineSegTensorColumn.xmaxs) <= threshold)) * \
                          ((self.pointTensorRow.Ys > self.lineSegTensorColumn.ymins) + (np.abs(self.pointTensorRow.Ys - self.lineSegTensorColumn.ymins) <= threshold)) * \
                          ((self.pointTensorRow.Ys < self.lineSegTensorColumn.ymaxs) + (np.abs(self.pointTensorRow.Ys - self.lineSegTensorColumn.ymaxs) <= threshold))
        else:
            includedMat = (self.pointTensorRow.Xs >= self.lineSegTensorColumn.xmins) * \
                          (self.pointTensorRow.Xs <= self.lineSegTensorColumn.xmaxs) * \
                          (self.pointTensorRow.Ys >= self.lineSegTensorColumn.ymins) * \
                          (self.pointTensorRow.Ys <= self.lineSegTensorColumn.ymaxs)

        # 返回
        return includedMat


    def __calcIsPointTensorAtLineSegTensorLocatedLine(self, isLoose: bool, threshold: float=1e2 * Utils.Threshold) -> bool:
        """
        计算 是否位于直线矩阵 这一矩阵记录着 点张量中的点是否位于线段张量中的线段所在的直线上
        Args:
            isLoose: 如真，则在一定阈值内认为相等
        Returns: 是否位于直线矩阵
        """
        # 线段张量 形成的 向量张量
        vectorTensorLSSstartToLSEnd = VectorTensor.generateVectorTensorFromStartsAndEnds(
            self.lineSegTensorColumn.startXs,
            self.lineSegTensorColumn.startYs,
            self.lineSegTensorColumn.endXs,
            self.lineSegTensorColumn.endYs
        )

        # 线段张量的起点 与 点张量的点 形成的 向量张量
        vectorTensorLSStartToPoint = VectorTensor.generateVectorTensorFromStartsAndEnds(
            self.lineSegTensorColumn.startXs,
            self.lineSegTensorColumn.startYs,
            self.pointTensorRow.Xs,
            self.pointTensorRow.Ys
        )

        # 以上两个向量张量 的 乘积张量
        crossValue = vectorTensorLSSstartToLSEnd.crossValue(vectorTensorLSStartToPoint)

        # 矩阵 存储 判断结果 点是否位于线段所在直线上
        if isLoose:
            isAtLineMat = (np.abs(crossValue - 0.0) <= threshold)
        else:
            isAtLineMat = (crossValue == 0.0)

        # 返回
        return isAtLineMat


    def calcIsIncludedMatForPTAndLT(self, isLoose: bool=True, threshold: float=1e2 * Utils.Threshold) -> bool:
        """
        计算 点张量 中的 每一个点 是否在 线段张量 中的 每一个线段 的 矩阵
        Args:
            isLoose: 如真，则在一定阈值内认为相等
        Returns: 这一矩阵
        """
        # 是否重叠矩阵
        includedMat = self.__calcIsPointTensorIncludedInLineSegTensor(isLoose, threshold=threshold)

        # 是否位于矩阵
        isAtLineMat = self.__calcIsPointTensorAtLineSegTensorLocatedLine(isLoose, threshold=threshold)

        # 是否位于线段矩阵
        isAtLineSegMat = includedMat * isAtLineMat

        # 返回
        return isAtLineSegMat

    pass




if __name__ == '__main__':
    points = np.array([
        [[10.0, 10.0]],
        [[0.7, 0.7]],
        [[-0.5, 0.5]],
        [[2.0, 2.5]]
    ])

    lineSegs = np.array([
        [[0.0, 0.0], [1.0, 1.0]],
        [[-1.0, 0.5], [2.0, 0.5]],
        [[1.0, 3.0], [2.0, 2.0]],
        [[1.5, 2.7], [2.0, 3.0]],
    ])

    pointTensorAtLineSegTensorInspector = PointTensorAtLineSegTensorInspector(points, lineSegs)
    isAtLineSegMat = pointTensorAtLineSegTensorInspector.calcIsIncludedMatForPTAndLT()

    pass
