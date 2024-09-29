# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/05 11:06
# @Function：
# @Refer：

import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.PointTensor import PointTensor
# from Welt.Tensor.Executors.Inspector import Inspector
from Welt.Tensor.Executors.Executor import Executor


class PointTensorPointTensorTiler(Executor):
    """
    点张量 与 点张量 的 平铺器（向量平铺为矩阵）
    """
    def __init__(self, pointsA: np.ndarray, pointsB: np.ndarray):
        """
        判定器初始化
        Args:
            pointsA: 容器A 存储 点
            pointsB: 容器B 存储 点
        """
        # 获取 点张量行 点张量列
        self.pointTensorRow, self.pointTensorColumn = self.__tileRawData(pointsA, pointsB)


    def __tileRawData(self, pointsA: np.ndarray, pointsB: np.ndarray) -> Tuple[PointTensor, PointTensor]:
        """
        转换 原始数据
        Args:
            pointsA:容器A 存储 点
            pointsB:容器B 存储 点
        Returns: 元组 (点张量行, 点张量列)
        """
        # 限制数据形状
        assert len(pointsA.shape) == 3 and len(pointsB.shape) == 3
        assert pointsA.shape[1] == 1 and pointsB.shape[1] == 1
        assert pointsA.shape[2] == 2 and pointsB.shape[2] == 2

        # 获取点数量
        pointsNumA = pointsA.shape[0]
        pointsNumB = pointsB.shape[0]

        # 点张量行 点张量列
        pointTensorRow = PointTensor(pointsA.reshape((pointsNumA, 1, 2)))
        pointTensorColumn = PointTensor(pointsB.reshape((1, pointsNumB, 2)))

        # 按列扩展 按行扩展
        pointTensorRow.entities = np.tile(pointTensorRow.entities, (1, pointsNumB, 1))
        pointTensorColumn.entities = np.tile(pointTensorColumn.entities, (pointsNumA, 1, 1))

        # 更新属性
        pointTensorRow.update()
        pointTensorColumn.update()

        # 返回
        return pointTensorRow, pointTensorColumn

    pass




if __name__ == '__main__':
    pointTensorPointTensorTiler = PointTensorPointTensorTiler()
