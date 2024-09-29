# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/23 上午9:03
# @Function:
# @Refer:

import numpy as np

from typing import List
from typing import Dict


class Graph(object):
    """
    图 类
    图 的 基本类型 | 保存一些关于图的基本方法
    """
    def __init__(self):
        pass


    @staticmethod
    def cvtAdjacencyMatToAdjacencyList(adjacencyMat: np.ndarray, allowRowEqsCol: bool=False) -> Dict[int, List[int]]:
        """
        将 邻接矩阵 转换为 邻接表
        Args:
            adjacencyMat: 邻接矩阵
            allowRowEqsCol: 是否允许 行号 等于 列号
        Returns: 邻接表
        """
        # 断言 邻接矩阵的维度等于2
        assert len(adjacencyMat.shape) == 2
        # assert adjacencyMat.shape[0] == adjacencyMat.shape[1]

        # 邻接表
        adjacencyList = dict()

        # 从 邻接矩阵 中 获取 值为真的 行索引 列索引
        rows, cols = np.where(adjacencyMat == True)

        # 以 行索引 为 key，以 空列表 为 value，建立邻接表（字典）
        for row in rows:
            adjacencyList[row] = list()

        # 联合遍历 行索引 列索引
        for row, col in zip(rows, cols):
            # 如果 不允许 行索引 与 列索引 相等
            if not allowRowEqsCol:
                # 则 当遇到 行索引 与 列索引 相等 时
                if row == col:
                    # 跳过
                    continue
            # 为 邻接表 的 一行 添加一个列索引
            adjacencyList[row].append(col)

        # 返回 邻接表
        return adjacencyList

    pass




if __name__ == '__main__':
    from Welt.Tensor.Inspectors.LineSegTensorIntersectWithLineSegTensorInspector import LineSegTensorIntersectWithLineSegTensorInspector

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

    algos = LineSegTensorIntersectWithLineSegTensorInspector(lineSegsA, lineSegsB)
    isIntersectMat = algos.calcIntersectingMatForTwoLineSegTensors()

    adjacencyList = Graph.cvtAdjacencyMatToAdjacencyList(isIntersectMat)

    pass
