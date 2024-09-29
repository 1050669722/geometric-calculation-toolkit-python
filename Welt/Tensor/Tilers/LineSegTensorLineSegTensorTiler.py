# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/31 下午4:42
# @Function:
# @Refer:

import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.LineSegTensor import LineSegTensor
# from Welt.Tensor.Executors.Inspector import Inspector
from Welt.Tensor.Executors.Executor import Executor


class LineSegTensorLineSegTensorTiler(Executor):
    """
    线段张量 与 线段张量 的 平铺器（向量平铺为矩阵）
    """
    def __init__(self, lineSegsA: np.ndarray, lineSegsB: np.ndarray):
        """
        判定器初始化
        Args:
            lineSegsA: 容器A 存储 线段
            lineSegsB: 容器B 存储 线段
        """
        # 获取 线段张量行 线段张量列
        self.lineSegTensorRow, self.lineSegTensorColumn = self.__tileRawData(lineSegsA, lineSegsB)


    def __tileRawData(self, lineSegsA: np.ndarray, lineSegsB: np.ndarray) -> Tuple[LineSegTensor, LineSegTensor]:
        """
        转换 原始数据
        Args:
            lineSegsA: 原始数据中的线段容器A
            lineSegsB: 原始数据中的线段容器B
        Returns: 元组 (线段张量行, 线段张量列)
        """
        # 限制数据形状
        assert len(lineSegsA.shape) == 3 and len(lineSegsB.shape) == 3
        assert lineSegsA.shape[1] == 2 and lineSegsB.shape[1] == 2
        assert lineSegsA.shape[2] == 2 and lineSegsB.shape[2] == 2

        # 获取线段数量
        lineSegsNumA = lineSegsA.shape[0]
        lineSegsNumB = lineSegsB.shape[0]

        # 线段张量行 线段张量列
        lineSegTensorRow = LineSegTensor(lineSegsA.reshape((lineSegsNumA, 1, 4)))
        lineSegTensorColumn = LineSegTensor(lineSegsB.reshape((1, lineSegsNumB, 4)))

        # 按列扩展 按行扩展
        lineSegTensorRow.entities = np.tile(lineSegTensorRow.entities, (1, lineSegsNumB, 1))
        lineSegTensorColumn.entities = np.tile(lineSegTensorColumn.entities, (lineSegsNumA, 1, 1))

        # 更新属性
        lineSegTensorRow.update()
        lineSegTensorColumn.update()

        # 返回
        return lineSegTensorRow, lineSegTensorColumn

    pass




if __name__ == '__main__':
    lineSegTensorLineSegTensorInspector = LineSegTensorLineSegTensorTiler()
