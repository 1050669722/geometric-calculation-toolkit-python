# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/23 上午9:03
# @Function:
# @Refer:

import sys
import numpy as np

from typing import List

from Welt.Graph.Graph import Graph
from Welt.Constants import Constants

sys.setrecursionlimit(Constants.MaxRecursionLimit) #设置调用栈最大深度，解释器默认为1000


class UndirectedGraph(Graph):
    """
    无向图 类
    """
    def __init__(self, adjacencyMat: np.ndarray):
        """
        初始化 无向图
        Args:
            adjacencyMat: 邻接矩阵 | 以邻接矩阵初始化一个无向图
        """
        # 采用 父类Graph 的 初始化方法 进行 初始化
        super(UndirectedGraph, self).__init__()

        # 邻接矩阵 转换为 邻接表
        self.adjacencyList = self.cvtAdjacencyMatToAdjacencyList(adjacencyMat)

        # 已经经历过的 vertexes 集合 | 图 的 基本组成单元 vertex, edge
        self.experiencedVertexIdxes = set()

        # 分类结果 列表
        self.classificationRes = list()


    def __search(self, currClsContainer: List[int], vertexIdx: int) -> None: #intersectLineSegIdxes: List[int]
        """
        以 currClsContainer 为类别容器，以 vertexIdx 为起始，搜索currClsContainer中应该添加的vertexIdx，并添加到currClsContainer中
        Args:
            currClsContainer: 容器 存储 应该属于本类 的 vertexIdx
            vertexIdx: 起始vertexIdx
        Returns: None
        """
        # 如果 vertexIdx 不是 已经经历过的 vertexIdx #TODO: 这个if判断语句似乎可以删除，因为进入此函数的vertexIdx 应该已经被判定过 是没有经历过的
        if vertexIdx not in self.experiencedVertexIdxes:
            # 则 向currClsContainer中 添加 vertexIdx
            currClsContainer.append(vertexIdx)
            # 将 此vertexIdx 记录为 已经经历过的
            self.experiencedVertexIdxes.add(vertexIdx)

        # 在 邻接表 中 遍历 与此vertexIdx 邻接的 adjacencyVertexIdx
        for adjacencyVertexIdx in self.adjacencyList[vertexIdx]:
            # 如果 本轮adjacencyVertexIdx 是 已经经历过的
            if adjacencyVertexIdx in self.experiencedVertexIdxes:
                # 则 跳过
                continue
            # 否则 以 currClsContainer 为类别容器，以 本轮adjacencyVertexIdx 为起始，搜索currClsContainer中应该添加的vertexIdx，并添加到currClsContainer中
            self.__search(currClsContainer, adjacencyVertexIdx)


    def classify(self) -> List[List[int]]:
        """
        将 无向图 中的 vertexes 按照 邻接关系 分类
        Returns: 分类结果列表 | 每一个类具有一个单独列表，其中存储着本类的vertex索引
        """
        # 遍历 邻接表 中的 vertex索引
        for vertexIdx in self.adjacencyList.keys():
            # 如果 本轮vertex索引 已经经历过
            if vertexIdx in self.experiencedVertexIdxes:
                # 则 跳过
                continue
            # 否则 分类结果列表 新增一类
            self.classificationRes.append([])
            # 以 新增这一类的容器 为类别容器，以 此vertex索引 为起始，搜索本类中应该添加的vertex索引，并添加到本类中
            self.__search(self.classificationRes[-1], vertexIdx)

        # 返回 分类结果列表
        return self.classificationRes

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

    # isIntersectMat = np.array([
    #     [True, True, True, True, True, False, False, False, False],
    #     [True, False, False, False, False, False, False, False, False],
    #     [True, False, False, False, False, False, False, False, False],
    #     [True, False, False, False, False, False, False, False, False],
    #     [True, False, False, False, False, False, False, False, True],
    #     [False, False, False, False, False, False, True, True, False],
    #     [False, False, False, False, False, True, False, False, False],
    #     [False, False, False, False, False, True, False, False, False],
    #     [False, False, False, False, True, False, False, False, False],
    # ])

    # isIntersectMat = np.array([
    #     [False, True, False, True],
    #     [True, False, True, False],
    #     [False, True, False, True],
    #     [True, False, True, False],
    # ])

    undirectedGraph = UndirectedGraph(isIntersectMat)
    classificationRes = undirectedGraph.classify()

    pass
