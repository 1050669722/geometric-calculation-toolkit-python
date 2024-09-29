# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/06/22 下午7:46
# @Function:
# @Refer:

# TODO: 不要试图运行此脚本，可能会宕机

import numpy as np
import warnings
warnings.simplefilter(action="default")

from typing import List
from warnings import warn

from Welt.Utils import Utils
from Welt.Structs.StructsMultiParticles.Surface.Polygon.Polygon import Polygon
from Welt.Tensor.Executors.Inspector import Inspector
from Welt.Tensor.Inspectors.PolygonIncludePointTensorInspector import PolygonIncludePointTensorInspector


class PolygonIncludePointTensorInspectorSplit(Inspector):
    """
    一个多边形 是否 包含 点张量 中的 点 的 判定器 | 将 点集合 分开
    """
    def __init__(self, polygon: Polygon, points: np.ndarray):
        self.polygon = polygon
        self.points = points.astype(np.float64)


    def __splitPoints(self, pointsList: List[np.ndarray], groupNum: int) -> None:
        """
        将 点张量 沿着 列(维度1) 分解
        Args:
            pointsList: 容器 存储 点张量
            groupNum: 分组数量
        Returns: None
        """
        # 列数
        colNum = self.points.shape[0]

        # 计算 分组索引
        startInds, endInds = Utils.calcIndsATTotalNumGroupNum(colNum, groupNum)

        # 联合遍历 startInds, endInds
        for startInd, endInd in zip(startInds, endInds):
            pointsList.append(self.points[:, startInd:endInd, :])


    def calcIncludingMatForPolygonAndPointTensorSplit(self) -> np.ndarray:
        """
        计算 多边形 是否包含 点张量中的 点 的 矩阵
        Returns: 这一矩阵
        """
        # 初始分组数目
        groupNum = 1

        # 主循环
        while True:
            try:
                # 分组数据
                pointsList = []
                self.__splitPoints(pointsList, groupNum)

                # 分组计算
                # 容器 存储 结果矩阵
                includingMatList = []
                for points in pointsList:
                    # 实例化 判定器
                    polygonIncludePointTensorInspector = PolygonIncludePointTensorInspector(self.polygon, points)
                    # 判定
                    includingMat = polygonIncludePointTensorInspector.calcIncludingMatForPolygonAndPointTensor()
                    # 添加 判定结果 至 容器
                    includingMatList.append(np.expand_dims(includingMat, axis=0))

                # 合并结果
                includingMatTotal = np.concatenate(includingMatList, axis=1)

                # 返回
                return includingMatTotal.squeeze(axis=0)

            except:
                groupNum *= 2
                warn("[WARNING] The points in too much, they have been split into {} groups".format(groupNum))

    pass




if __name__ == '__main__':
    # # 实例化 多边形 方法1 <点>
    # polygon = Polygon([
    #     [0.0, 1.0],
    #     [1.0, -1.0],
    #     [0.0, 0.0],
    #     [-1.0, -1.0]
    # ])
    #
    # # 点
    # points = np.array([
    #     [[0.0, 0.0]],
    #     [[0.0, 0.5]],
    #     [[0.1, 0.2]],
    #     [[-1.0, -1.0]],
    #     [[-0.2, 0.0]],
    #     [[1.0, -2.0]],
    #     [[0.5, -0.5]],
    #     [[-0.5, -0.5]],
    # ])

    # 读取数据
    max_contour = Polygon(Utils.load("../../../data/max_contour.pkl"))
    seed_points = Utils.load("../../../data/seed_points.pkl")
    # seed_points = np.concatenate([seed_points] * 10, axis=0)

    # 实例化 判定器
    # polygonIncludePointTensorInspectorSplit = PolygonIncludePointTensorInspectorSplit(polygon, points)
    polygonIncludePointTensorInspectorSplit = PolygonIncludePointTensorInspectorSplit(max_contour, seed_points)

    # 判定
    res = polygonIncludePointTensorInspectorSplit.calcIncludingMatForPolygonAndPointTensorSplit()

    # 打印
    print(res)