# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/31 上午10:48
# @Function:
# @Refer:

import numpy as np

# from typing import Tuple

from Welt.Utils import Utils
# from Welt.Tensor.Tensors.PointTensor import PointTensor
from Welt.Tensor.Tilers.PointTensorPointTensorTiler import PointTensorPointTensorTiler


class PointTensorCoincideWithPointTensorInspector(PointTensorPointTensorTiler):
    """
    点张量 中的 点 与 点张量 中的 点 是否重合 的 判定器
    """
    def __init__(self, pointsA: np.ndarray, pointsB: np.ndarray):
        """
        判定器初始化
        Args:
            pointsA: 容器A 存储 点
            pointsB: 容器B 存储 点
        """
        # 采用父类初始化方法 初始化
        super(PointTensorCoincideWithPointTensorInspector, self).__init__(pointsA, pointsB)


    def calcIsCoincidedMatForTwoPointTensors(self, isLoose: bool=True, threshold=Utils.Threshold) -> np.ndarray:
        """
        计算 两个点张量 中的 点 两两之间 是否 重合 的 矩阵
        Args:
            isLoose: 如真，则在一定阈值内认为相等
        Returns: 这一矩阵
        """
        # 如果放宽判定阈值
        if isLoose: #1e2 * Utils.Threshold
            # 矩阵 存储 两两判断的结果 点是否重合
            isCoincidedMat = (np.abs(self.pointTensorRow.Xs - self.pointTensorColumn.Xs) <= threshold) * \
                             (np.abs(self.pointTensorRow.Ys - self.pointTensorColumn.Ys) <= threshold)
        # 否则
        else:
            # 矩阵 存储 两两判断的结果 点是否重合
            isCoincidedMat = (self.pointTensorRow.Xs == self.pointTensorColumn.Xs) * \
                             (self.pointTensorRow.Ys == self.pointTensorColumn.Ys)
    
        # 返回
        return isCoincidedMat
    
    pass




if __name__ == '__main__':
    pointsA = np.array([
        [[10.0, 10.0]],
        [[0.7, 0.7]],
        [[-0.5, 0.5]],
        [[2.0, 2.5]]
    ])

    pointsB = np.array([
        [[10.0, 10.0]],
        [[0.7, 0.7]],
        [[-0.5, 0.5]],
        [[2.0, 2.5]]
    ])

    pointTensorCoincideWithPointTensorInspector = PointTensorCoincideWithPointTensorInspector(pointsA, pointsB)
    isCoincidedMat = pointTensorCoincideWithPointTensorInspector.calcIsCoincidedMatForTwoPointTensors(False)

    pass
