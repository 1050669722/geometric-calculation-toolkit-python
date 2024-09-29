# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/05 11:28
# @Function：
# @Refer：

import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.PointTensor import PointTensor
from Welt.Tensor.Tensors.LineSegTensor import LineSegTensor
# from Welt.Tensor.Executors.Inspector import Inspector
from Welt.Tensor.Executors.Executor import Executor


class PointTensorLineSegTensorTiler(Executor):
    """
    点张量 与 线段张量 的 平铺器（向量平铺为矩阵）
    """
    def __init__(self, points: np.ndarray, lineSegs: np.ndarray):
        """
        判定器初始化
        Args:
            points: 容器 存储 点
            lineSegs: 容器 存储 线段
        """
        # 获取 点张量行 线段张量列
        self.pointTensorRow, self.lineSegTensorColumn = self.__tileRawData(points, lineSegs)


    def __tileRawData(self, points: np.ndarray, lineSegs: np.ndarray) -> Tuple[PointTensor, LineSegTensor]:
        """
        转换 原始数据
        Args:
            points: 容器 存储 点
            lineSegs: 容器 存储 线段
        Returns: 元组 (点张量行, 线段张量列)
        """
        # 限制数据形状
        assert len(points.shape) == 3 and len(lineSegs.shape) == 3
        assert points.shape[1] == 1 and lineSegs.shape[1] == 2
        assert points.shape[2] == 2 and lineSegs.shape[2] == 2

        # 获取 点数量 和 线段数量
        pointsNum = points.shape[0]
        lineSegsNum = lineSegs.shape[0]

        # 点张量行 线段数量列
        pointTensorRow = PointTensor(points.reshape((pointsNum, 1, 2)))
        lineSegTensorColumn = LineSegTensor(lineSegs.reshape((1, lineSegsNum, 4)))

        # 点张量行按列扩展 线段张量列按行扩展
        pointTensorRow.entities = np.tile(pointTensorRow.entities, (1, lineSegsNum, 1))
        lineSegTensorColumn.entities = np.tile(lineSegTensorColumn.entities, (pointsNum, 1, 1))

        # 更新属性
        pointTensorRow.update()
        lineSegTensorColumn.update()

        # 返回
        return pointTensorRow, lineSegTensorColumn

    pass




if __name__ == '__main__':
    pointTensorLineSegTensorTiler = PointTensorLineSegTensorTiler()
