# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/05 11:51
# @Function：
# @Refer：

import numpy as np

from typing import Tuple

from Welt.Constants import Constants
from Welt.Structs.StructsMultiParticles.Surface.Polygon.Polygon import Polygon
# from Welt.Tensor.Executors.Inspector import Inspector
from Welt.Tensor.Executors.Executor import Executor


class PolygonPointTensorTiler(Executor):
    """
    多边形 与 点张量 的 平铺器（向量平铺为矩阵）
    """
    def __init__(self, polygon: Polygon, points: np.ndarray):
        """"""
        self.polygonEdgesNdarray, self.pointLineSegsNdarray = self.__tileRawData(polygon, points.astype(np.float64))


    def __tileRawData(self, polygon: Polygon, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换 原始数据
        Args:
            polygon: 任意多边形
            points: 容器 存储 点
        Returns: 元组 (多边形边数组, 点射线段数组)
        """
        return self.__convertPolygonToPolygonEdgesNdarray(polygon), self.__convertPointsToPointLineSegsNdarray(polygon, points)


    def __convertPolygonToPolygonEdgesNdarray(self, polygon: Polygon) -> np.ndarray:
        """
        将多边形 转换为 多边形边数组
        Args:
            polygon: 待转换多边形
        Returns: 多边形边数组
        """
        # 将要转换为的多边形数组
        polygonEdgesNdarray = None #np.array([0, 0]).reshape((1, 1, 2))

        # 遍历 多边形 的 每一条边
        for edgeNdarray in polygon.edges:
            # 数组化 起点 终点
            startPointNdarray = np.array(edgeNdarray.startPoint).reshape((1, 1, 2)).astype(np.float64)
            endPointNdarray = np.array(edgeNdarray.endPoint).reshape((1, 1, 2)).astype(np.float64)
            # 数组化 边
            edgeNdarray = np.concatenate((startPointNdarray, endPointNdarray), axis=1)
            # edgeNdarray 拼接到 polygonNdarray
            polygonEdgesNdarray = edgeNdarray if polygonEdgesNdarray is None else np.concatenate((polygonEdgesNdarray, edgeNdarray), axis=0)

        # 返回
        return polygonEdgesNdarray


    def __convertPointsToPointLineSegsNdarray(self, polygon: Polygon, points: np.ndarray) -> np.ndarray:
        """
        将点 转换为 线段数组
        Args:
            polygon: 待判断包含逻辑的多边形
            points: 待转换点集合
        Returns: 从这些点出发的线段，这些线段的终点一定位于多边形外部
        """
        # 限制数据形状
        assert len(points.shape) == 3
        assert points.shape[1] == 1
        assert points.shape[2] == 2

        # 将要转换为的线段数组
        pointLineSegsNdarray = None

        # 包围盒坐标
        xmin, ymin, xmax, ymax = polygon.getBoundCoord()

        # 包围盒外一点
        exteriorPoint = np.array([xmax + Constants.boundingBoxExteriorLength, ymax + Constants.boundingBoxExteriorLength]).reshape((1, 1, 2)).astype(np.float64)

        # 遍历 points 中的每一个点
        for point in points:
            # 这一点 形成 数组
            point = point.reshape((1, 1, 2))
            # 拼接 这一点 与 外部点 形成 一个 线段数组
            pointLineSegNdarray = np.concatenate((point, exteriorPoint), axis=1).astype(np.float64)
            # pointLineSegNdarray 拼接到 pointLineSegsNdarray
            pointLineSegsNdarray = pointLineSegNdarray if pointLineSegsNdarray is None else np.concatenate((pointLineSegsNdarray, pointLineSegNdarray), axis=0)

        # 返回
        return pointLineSegsNdarray

    pass




if __name__ == '__main__':
    polygonPointTensorTiler = PolygonPointTensorTiler()

