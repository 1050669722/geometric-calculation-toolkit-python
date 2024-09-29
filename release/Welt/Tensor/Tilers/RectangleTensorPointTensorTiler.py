# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/05 14:26
# @Function：
# @Refer：

import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.PointTensor import PointTensor
from Welt.Tensor.Tensors.RectangleTensor import RectangleTensor
# from Welt.Tensor.Executors.Inspector import Inspector
from Welt.Tensor.Executors.Executor import Executor


class RectangleTensorPointTensorTiler(Executor):
    """
    矩形张量 与 点张量 的 平铺器（向量平铺为矩阵）
    """
    def __init__(self, rectangles: np.ndarray, points: np.ndarray):
        """
        判定器初始化
        Args:
            rectangles: 容器 存储 矩形
            points: 容器 存储 点
        """
        # 获取 矩形张量行 点张量列
        self.rectangleTensorRow, self.pointTensorColumn = self.__tileRawData(rectangles.astype(np.float64), points.astype(np.float64))


    # TODO: 似乎 可以与 LineSegTensorIntersectionInspector.py中的 __tileRawData()方法 合并
    def __tileRawData(self, rectangles: np.ndarray, points: np.ndarray) -> Tuple[RectangleTensor, PointTensor]:
        """
        转换 原始数据
        Args:
            rectangles: 原始数据中的矩形容器
            points: 原始数据中的点容器
        Returns: 元组 (矩形张量行, 点张量列)
        """
        # 限制数据形状
        assert len(rectangles.shape) == 3 and len(points.shape) == 3
        assert rectangles.shape[1] == 4 and points.shape[1] == 1
        assert rectangles.shape[2] == 2 and points.shape[2] == 2

        # 获取 矩形数量 和 点数量
        rectanglesNum = rectangles.shape[0]
        pointsNum = points.shape[0]

        # 矩形张量行 点张量列
        rectangleTensorRow = RectangleTensor(rectangles.reshape((rectanglesNum, 1, 8)))
        pointTensorColumn = PointTensor(points.reshape((1, pointsNum, 2)))

        # 矩形张量行按列扩展 点张量列按行扩展
        rectangleTensorRow.entities = np.tile(rectangleTensorRow.entities, (1, pointsNum, 1))
        pointTensorColumn.entities = np.tile(pointTensorColumn.entities, (rectanglesNum, 1, 1))

        # 更新属性
        rectangleTensorRow.update()
        pointTensorColumn.update()

        # 返回
        return rectangleTensorRow, pointTensorColumn




if __name__ == '__main__':
    rectangleTensorPointTensorTiler = RectangleTensorPointTensorTiler()