# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/31 下午4:30
# @Function:
# @Refer:

import numpy as np

from Welt.Utils import Utils
from Welt.Tensor.Tilers.LineSegTensorLineSegTensorTiler import LineSegTensorLineSegTensorTiler


class LineSegTensorCoincideWithLineSegTensorInspector(LineSegTensorLineSegTensorTiler):
    """
    线段张量 中的 线段 与 线段张量 中的 线段 是否重合 的 判定器
    """
    def __init__(self, lineSegsA: np.ndarray, lineSegsB: np.ndarray):
        """
        判定器初始化
        Args:
            lineSegsA: 容器A 存储 线段
            lineSegsB: 容器B 存储 线段
        """
        # 采用父类初始化方法 初始化
        super(LineSegTensorCoincideWithLineSegTensorInspector, self).__init__(lineSegsA, lineSegsB)


    def calcIsCoincidedMatForTwoLineSegTensors(self, isLoose: bool=True, threshold: float=1e2 * Utils.Threshold) -> np.ndarray:
        """
        计算 两个线段张量 中的 线段 两两之间 是否 重合 的 矩阵
        Args:
            isLoose: 如真，则在一定阈值内认为相等
        Returns: 这一矩阵
        """
        # 矩阵 存储 两两判断的结果 线段是否重合
        if isLoose:
            isCoincidedMat = (np.abs(self.lineSegTensorRow.startXs - self.lineSegTensorColumn.startXs) <= threshold) * \
                             (np.abs(self.lineSegTensorRow.startYs - self.lineSegTensorColumn.startYs) <= threshold) * \
                             (np.abs(self.lineSegTensorRow.endXs - self.lineSegTensorColumn.endXs) <= threshold) * \
                             (np.abs(self.lineSegTensorRow.endYs - self.lineSegTensorColumn.endYs) <= threshold) \
                             + \
                             (np.abs(self.lineSegTensorRow.startXs - self.lineSegTensorColumn.endXs) <= threshold) * \
                             (np.abs(self.lineSegTensorRow.startYs - self.lineSegTensorColumn.endYs) <= threshold) * \
                             (np.abs(self.lineSegTensorRow.endXs - self.lineSegTensorColumn.startXs) <= threshold) * \
                             (np.abs(self.lineSegTensorRow.endYs - self.lineSegTensorColumn.startYs) <= threshold)
        else:
            isCoincidedMat = (self.lineSegTensorRow.startXs == self.lineSegTensorColumn.startXs) * \
                             (self.lineSegTensorRow.startYs == self.lineSegTensorColumn.startYs) * \
                             (self.lineSegTensorRow.endXs == self.lineSegTensorColumn.endXs) * \
                             (self.lineSegTensorRow.endYs == self.lineSegTensorColumn.endYs) \
                              + \
                             (self.lineSegTensorRow.startXs == self.lineSegTensorColumn.endXs) * \
                             (self.lineSegTensorRow.startYs == self.lineSegTensorColumn.endYs) * \
                             (self.lineSegTensorRow.endXs == self.lineSegTensorColumn.startXs) * \
                             (self.lineSegTensorRow.endYs == self.lineSegTensorColumn.startYs)

        # 返回
        return isCoincidedMat

    pass




if __name__ == '__main__':
    lineSegsA = np.array([
        [[0.0, 0.0], [1.0, 1.0]],
        [[-1.0, 0.5], [2.0, 0.5]],
        [[1.0, 3.0], [2.0, 2.0]],
        [[1.5, 2.7], [2.0, 3.0]],
    ])

    lineSegsB = np.array([
        [[0.0, 0.0], [1.0, 1.0]],
        [[-1.0, 0.5], [2.0, 0.5]],
        [[1.0, 3.0], [2.0, 2.0]],
        [[1.5, 2.7], [2.0, 3.0]],
    ])

    lineSegTensorCoincideWithLineSegTensorInspector = LineSegTensorCoincideWithLineSegTensorInspector(lineSegsA, lineSegsB)
    isCoincidedMat = lineSegTensorCoincideWithLineSegTensorInspector.calcIsCoincidedMatForTwoLineSegTensors(False)

    pass
