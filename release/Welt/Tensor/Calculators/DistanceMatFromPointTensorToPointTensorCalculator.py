# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/05 10:43
# @Function：
# @Refer：

import numpy as np

from Welt.Tensor.Tilers.PointTensorPointTensorTiler import PointTensorPointTensorTiler


class DistanceMatFromPointTensorToPointTensorCalculator(PointTensorPointTensorTiler):
    """
    点张量 中的 点 与 点张量 中的 点 欧氏距离 的 计算器
    """
    def __init__(self, pointsA: np.ndarray, pointsB: np.ndarray):
        """
        计算器初始化
        Args:
            pointsA: 容器A 存储 点
            pointsB: 容器B 存储 点
        """
        # 采用父类初始化方法 初始化
        super(DistanceMatFromPointTensorToPointTensorCalculator, self).__init__(pointsA, pointsB)


    def calcDistanceMatForTwoPointTensors(self) -> np.ndarray:
        """
        计算 两个点张量 中的 点 两两之间 欧氏距离 的 矩阵
        Returns: 这一矩阵
        """
        # # 计算 距离矩阵 #尽管采用了更大容量的数据类型
        # distanceMat = np.sqrt(np.power(self.pointTensorRow.Xs - self.pointTensorColumn.Xs, 2).astype(np.float64) + np.power(self.pointTensorRow.Ys - self.pointTensorColumn.Ys, 2).astype(np.float64))

        # 计算 距离矩阵 #更加安全的实现
        vecMat = np.concatenate(
            (
                np.expand_dims(self.pointTensorRow.Xs - self.pointTensorColumn.Xs, axis=2),
                np.expand_dims(self.pointTensorRow.Ys - self.pointTensorColumn.Ys, axis=2)
            ),
            axis=2
        )
        distanceMat = np.linalg.norm(vecMat, axis=2, ord=2)

        # 返回
        return distanceMat

    pass




if __name__ == '__main__':
    from Welt.Tensor.Converters.PointTensorConverter import PointTensorConverter

    # 点
    pointsA = [
        [0, 0],
        [1.0, 1.0],
        [2, 0],
    ]
    pointsB = [
        [0, 0],
        [1.0, 1.0],
        [2, 0],
    ]

    # 转换
    pointsA = PointTensorConverter.convert(pointsA)
    pointsB = PointTensorConverter.convert(pointsB)

    # 实例化
    distanceFromPointTensorToPointTensorCalculator = DistanceMatFromPointTensorToPointTensorCalculator(pointsA, pointsB)

    # 计算
    distanceMat = distanceFromPointTensorToPointTensorCalculator.calcDistanceMatForTwoPointTensors()

    # 打印
    print(distanceMat)
    pass
