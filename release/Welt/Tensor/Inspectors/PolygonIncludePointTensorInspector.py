# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/06/02 下午2:51
# @Function:
# @Refer:

import numpy as np

# from typing import List

# from Welt.Utils import Utils
from Welt.Constants import Constants
from Welt.Structs.StructsMultiParticles.Surface.Polygon.Polygon import Polygon
# from Welt.Tensor.Executors.Inspector import Inspector
from Welt.Tensor.Tilers.PolygonPointTensorTiler import PolygonPointTensorTiler
from Welt.Tensor.Inspectors.LineSegTensorIntersectWithLineSegTensorInspector import LineSegTensorIntersectWithLineSegTensorInspector
from Welt.Tensor.Inspectors.PointTensorAtLineSegTensorInspector import PointTensorAtLineSegTensorInspector


class PolygonIncludePointTensorInspector(PolygonPointTensorTiler):
    """
    一个多边形 是否 包含 点张量 中的 点 的 判定器
    """
    def __init__(self, polygon: Polygon, points: np.ndarray):
        """
        判定器初始化
        Args:
            polygon: 任意多边形
            points: 容器 存储 点
        """
        self.points = points.astype(np.float64)
        # self.polygonEdgesNdarray = self.__convertPolygonToPolygonEdgesNdarray(polygon)
        # self.pointLineSegsNdarray = self.__convertPointsToPointLineSegsNdarray(polygon, points)
        super(PolygonIncludePointTensorInspector, self).__init__(polygon, points)


    # def __convertPolygonToPolygonEdgesNdarray(self, polygon: Polygon) -> np.ndarray:
    #     """
    #     将多边形 转换为 多边形边数组
    #     Args:
    #         polygon: 待转换多边形
    #     Returns: 多边形边数组
    #     """
    #     # 将要转换为的多边形数组
    #     polygonEdgesNdarray = None #np.array([0, 0]).reshape((1, 1, 2))
    #
    #     # 遍历 多边形 的 每一条边
    #     for edgeNdarray in polygon.edges:
    #         # 数组化 起点 终点
    #         startPointNdarray = np.array(edgeNdarray.startPoint).reshape((1, 1, 2))
    #         endPointNdarray = np.array(edgeNdarray.endPoint).reshape((1, 1, 2))
    #         # 数组化 边
    #         edgeNdarray = np.concatenate((startPointNdarray, endPointNdarray), axis=1)
    #         # edgeNdarray 拼接到 polygonNdarray
    #         polygonEdgesNdarray = edgeNdarray if polygonEdgesNdarray is None else np.concatenate((polygonEdgesNdarray, edgeNdarray), axis=0)
    #
    #     # 返回
    #     return polygonEdgesNdarray


    # def __convertPointsToPointLineSegsNdarray(self, polygon: Polygon, points: np.ndarray) -> np.ndarray:
    #     """
    #     将点 转换为 线段数组
    #     Args:
    #         polygon: 待判断包含逻辑的多边形
    #         points: 待转换点集合
    #     Returns: 从这些点出发的线段，这些线段的终点一定位于多边形外部
    #     """
    #     # 限制数据形状
    #     assert len(points.shape) == 3
    #     assert points.shape[1] == 1
    #     assert points.shape[2] == 2
    #
    #     # 将要转换为的线段数组
    #     pointLineSegsNdarray = None
    #
    #     # 包围盒坐标
    #     xmin, ymin, xmax, ymax = polygon.getBoundCoord()
    #
    #     # 包围盒外一点
    #     exteriorPoint = np.array([xmax + Constants.boundingBoxExteriorLength, ymax + Constants.boundingBoxExteriorLength]).reshape((1, 1, 2))
    #
    #     # 遍历 points 中的每一个点
    #     for point in points:
    #         # 这一点 形成 数组
    #         point = point.reshape((1, 1, 2))
    #         # 拼接 这一点 与 外部点 形成 一个 线段数组
    #         pointLineSegNdarray = np.concatenate((point, exteriorPoint), axis=1)
    #         # pointLineSegNdarray 拼接到 pointLineSegsNdarray
    #         pointLineSegsNdarray = pointLineSegNdarray if pointLineSegsNdarray is None else np.concatenate((pointLineSegsNdarray, pointLineSegNdarray), axis=0)
    #
    #     # 返回
    #     return pointLineSegsNdarray


    def __calcIsIntersectedMatForTwoLineSegTensors(self) -> np.ndarray:
        """
        计算 两个线段张量(来自于self.polygonEdgesNdarray, self.pointLineSegsNdarray) 中的 线段 两两之间 是否 存在 公共点 的 矩阵
        Returns: 这一矩阵
        """
        # 实例化 线段张量相交线段张量判定器
        lineSegTensorIntersectionInspector = LineSegTensorIntersectWithLineSegTensorInspector(self.polygonEdgesNdarray, self.pointLineSegsNdarray)

        # 计算 相交矩阵
        isIntersectMat = lineSegTensorIntersectionInspector.calcIntersectingMatForTwoLineSegTensors()

        # 返回
        return isIntersectMat


    def __calcIsIncludedMatForPTAndLT(self) -> bool:
        """
        为 点张量 和 线段张量 计算 位于矩阵
        Returns: 这一矩阵
        """
        # 实例化 点张量位于线段张量判定器
        pointTensorAtLineSegTensorInspector = PointTensorAtLineSegTensorInspector(self.points, self.polygonEdgesNdarray)

        # 计算 位于矩阵
        isAtLineSegMat = pointTensorAtLineSegTensorInspector.calcIsIncludedMatForPTAndLT(isLoose=False) #严格位于

        # 返回
        return isAtLineSegMat


    def calcIncludingMatForPolygonAndPointTensor(self) -> np.ndarray:
        """
        计算 多边形 是否包含 点张量中的 点 的 矩阵
        Returns: 这一矩阵
        """
        # 由 引射线法 计算得到的 包含 矩阵
        isIntersectMat = self.__calcIsIntersectedMatForTwoLineSegTensors()
        isIntersectMat = np.sum(isIntersectMat, axis=0)
        isIntersectMat = (isIntersectMat & 1).astype(bool)

        # 如果 点 在 边 上，则该位置上的值为0
        isAtLineSegMat = self.__calcIsIncludedMatForPTAndLT()
        isAtLineSegMat = np.sum(isAtLineSegMat, axis=1).transpose()
        isAtLineSegMat = (isAtLineSegMat | 0).astype(bool)

        # 结果矩阵
        includingMat = np.ones(isAtLineSegMat.shape)
        includingMat[isAtLineSegMat == True] = 0
        includingMat[(includingMat == 1) * (isIntersectMat == False)] = -1

        # 返回
        return includingMat

    pass




if __name__ == '__main__':
    from Welt.Structs.StructsMultiParticles.Surface.Polygon.Polygon import Polygon
    # from Welt.Tensor.Inspectors.PolygonIncludePointTensorInspector import PolygonIncludePointTensorInspector

    # 实例化 多边形 方法1 <点>
    polygon = Polygon([
        [0.0, 1.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [-1.0, -1.0]
    ])

    # # 实例化 多边形 方法2 <边>
    # polygon = Polygon([
    #     [[0.0, 1.0], [1.0, -1.0]],
    #     [[1.0, -1.0], [0.0, 0.0]],
    #     [[0.0, 0.0], [-1.0, -1.0]],
    #     [[-1.0, -1.0], [0.0, 1.0]],
    # ])

    # 点
    points = np.array([
        [[0.0, 0.0]],
        [[0.0, 0.5]],
        [[0.1, 0.2]],
        [[-1.0, -1.0]],
        [[-0.2, 0.0]],
        [[1.0, -2.0]],
        [[0.5, -0.5]],
        [[-0.5, -0.5]],
    ])

    # 实例化 判定器
    polygonIncludePointTensorInspector = PolygonIncludePointTensorInspector(polygon, points)

    # 判定
    res = polygonIncludePointTensorInspector.calcIncludingMatForPolygonAndPointTensor()

    # 打印
    print(res)
