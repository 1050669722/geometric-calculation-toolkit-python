# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   AssemblyAlgos2D
Description: 
Author:      liu
Date:        1/19/22
---------------------------------------
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action="default")

from typing import Tuple, List
# from typing import Iterable as typingIterable
from copy import deepcopy
from warnings import warn

from Welt.Utils import Utils
from Welt.Structs.StructsSingleParticle.Point.Point import Point
from Welt.Structs.StructsSingleParticle.Point.PointContainer import PointContainer
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSeg import LineSeg
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSegContainer import LineSegContainer
from Welt.Structs.StructsDoubleParticles.Vector import Vector
# from Welt.Structs.StructsMultiParticles.Line.PolyLineSeg.PolyLineSeg import PolyLineSeg
from Welt.Structs.StructsMultiParticles.Surface.Quadrangle.Quadrangle import Quadrangle
from Welt.Structs.StructsMultiParticles.Surface.Triangle.Triangle import Triangle
from Welt.Structs.StructsMultiParticles.Surface.Polygon.Polygon import Polygon
from Welt.Geometry.Geometry2D.CommonAlgos2D import CommonAlgos2D
from Welt.Geometry.GeneralAlgos import GeneralAlgos

from Welt.Tensor.Inspectors.LineSegTensorIntersectWithLineSegTensorInspector import \
    LineSegTensorIntersectWithLineSegTensorInspector
from Welt.Tensor.Converters.PointTensorConverter import PointTensorConverter
from Welt.Tensor.Inspectors.PointTensorCoincideWithPointTensorInspector import \
    PointTensorCoincideWithPointTensorInspector
from Welt.Graph.UndirectedGraph import UndirectedGraph


class AssemblyAlgos2D(object):
    @staticmethod
    def boundCoord2Quadrangle(bbox):
        """
        最小最大坐标转换为面
        :param coord:
        :return: Quadrangle | **左下角为起点，逆时针**
        """
        xmin, ymin, xmax, ymax = bbox
        return Quadrangle([
            # Point([xmin, ymin, 0]),
            # Point([xmax, ymin, 0]),
            # Point([xmax, ymax, 0]),
            # Point([xmin, ymax, 0])
            Point([xmin, ymin]),
            Point([xmax, ymin]),
            Point([xmax, ymax]),
            Point([xmin, ymax])
        ])


    @staticmethod
    def isPointInQuadrangle(point: List[float], quadrangle: Quadrangle, isLoose: bool = True) -> bool:
        """
        点是否在四变形内
        :param point:
        :param face:
        :return: Bool
        """
        xf = (point[0] - quadrangle.vertexDA[0]) * (point[0] - quadrangle.vertexBC[0])
        yf = (point[1] - quadrangle.vertexDA[1]) * (point[1] - quadrangle.vertexBC[1])
        if isLoose:
            return Utils.isLessEqual(xf, 0.0) and Utils.isLessEqual(yf, 0.0)
        else:
            return xf <= 0.0 and yf <= 0.0


    @classmethod
    def isPointAtLine(cls, point: List[float], line: List[List[float]], isLoose: bool = True, threshold: float = 1e2 * Utils.Threshold) -> bool:
        """
        判断 点 是否位于 直线上
        Args:
            point: 点
            line: 直线
        Returns: 点 是否位于 直线上
        """
        crossProduct = CommonAlgos2D.cross(
            [line[1][0] - line[0][0], line[1][1] - line[0][1]],
            [point[0] - line[0][0], point[1] - line[0][1]]
        )
        if isLoose:
            return Utils.isEqual(crossProduct, 0.0, threshold)
        else:
            return crossProduct == 0.0


    @classmethod
    def isPointAtLineSeg(cls, point: List[float], lineSeg: LineSeg, pointAtLineThreshold: float = 1e2 * Utils.Threshold) -> bool:
        """
        点是否在线段上
        :param point:
        :param lineSeg:
        :return: Bool
        """
        # # 点是否在线段所在直线上
        # crossProduct = CommonAlgos2D.cross(
        #                             [lineSeg[1][0] - lineSeg[0][0], lineSeg[1][1] - lineSeg[0][1]],
        #                             [point[0] - lineSeg[0][0], point[1] - lineSeg[0][1]]
        # )

        # 点是否在以线段为对角线的矩形内
        isPointInFace = cls.isPointInQuadrangle(point, cls.boundCoord2Quadrangle(lineSeg.getBoundCoord()))

        # # 二者同时成立
        # return Utils.isEqual(crossProduct, 0.0) and isPointInFace
        return cls.isPointAtLine(point, lineSeg, isLoose=True, threshold=pointAtLineThreshold) and isPointInFace


    @classmethod
    def getUnitVerticalVectorFromPointToLine(cls, point: List[float], lineSeg: LineSeg) -> Vector:
        """
        获取 从点到线段所在直线 的 单位垂直向量
        Args:
            point: 点
            lineSeg: 线段 表征 直线
        Returns: 单位垂直向量（Vector）
        """
        # 断言 线段 起点 终点 不重合 | 严格地
        # assert lineSeg.length != 0.0
        assert lineSeg.startPoint != lineSeg.endPoint
        
        # 线段的方向向量
        lineSegDirectionVec = lineSeg.getDirectionVec()

        # 与 线段的方向向量 垂直的 两个垂直向量
        lineSegVerticalVec0, lineSegVerticalVec1 = Utils.getVerticalVectors(lineSegDirectionVec)

        # 两个候选向量
        lineSegVerticalVec0 = Vector.generateVectorFromPointAndDirection(point, lineSegVerticalVec0)
        lineSegVerticalVec1 = Vector.generateVectorFromPointAndDirection(point, lineSegVerticalVec1)

        # 两个候选向量 分别 归一化
        lineSegVerticalVec0.normalize()
        lineSegVerticalVec1.normalize()

        # 谁的终点与线段所在直线更近，就返回谁
        if GeneralAlgos.calcDistanceFromPointToLine(lineSegVerticalVec0.endPoint, lineSeg) <= \
                GeneralAlgos.calcDistanceFromPointToLine(lineSegVerticalVec1.endPoint, lineSeg):
            return lineSegVerticalVec0
        else:
            return lineSegVerticalVec1


    @classmethod
    def getFootPointFromPointToLine(cls, point: List[float], lineSeg: LineSeg) -> List[float]:
        """
        计算 从 点 到 直线 的 垂线段 的 垂足
        Args:
            point: 点
            lineSeg: 线段 表征 直线
        Returns: 垂足
        """
        # 断言 线段 起点 终点 不重合 | 严格地
        # assert lineSeg.length != 0.0
        assert lineSeg.startPoint != lineSeg.endPoint
        
        # 返回 投影点
        return GeneralAlgos.calcProjectivePointFromPointToLine(Point(point), lineSeg)


    @classmethod
    def getFootPointFromPointToLineSeg(cls, point: List[float], lineSeg: LineSeg) -> List[float]:
        """
        计算 从 点 到 线段 的 垂线段 的 垂足
        Args:
            point: 点
            lineSeg: 线段
        Returns: 垂足
        """
        # 断言 线段 起点 终点 不重合 | 严格地
        # assert lineSeg.length != 0.0
        assert lineSeg.startPoint != lineSeg.endPoint
        
        # 计算 从 点 到 线段所在直线 的 垂足
        footPoint = cls.getFootPointFromPointToLine(point, lineSeg)

        # 如果 点 在 线段 上
        if cls.isPointAtLineSeg(footPoint, lineSeg):
            # 返回 垂足
            return footPoint
        # 如果 点 不在 线段 上
        else:
            # 返回 None
            return None


    @classmethod
    def getVerticalLineFromPointToLine(cls, point: List[float], lineSeg: LineSeg, lengthThreshold: float = 1e2 * Utils.Threshold, extendingValue: float = 10.0) -> LineSeg:
        """
        获取 从 点 到 直线 的 垂线段
        Args:
            point: 点
            lineSeg: 线段 表征 直线
        Returns: 垂线段
        """
        # 断言 线段 起点 终点 不重合 | 严格地
        # assert lineSeg.length != 0.0
        assert lineSeg.startPoint != lineSeg.endPoint

        # 从 点 到 线段所在直线的 垂线段 的 垂足
        footPoint = Point(cls.getFootPointFromPointToLine(point, lineSeg))

        # # 如果 这一点 就在 线段所在的直线上
        # if cls.isPointAtLine(footPoint, lineSeg, isLoose=True, threshold=Utils.Threshold):
        # # 如果 这一垂足 与 点 重合 | 严格地
        # if footPoint == point:
        # 从 该点出发的 该直线的 单位垂直向量
        lineSegVerticalVec = cls.getUnitVerticalVectorFromPointToLine(footPoint, lineSeg)

        # 平移
        if footPoint.getDimensionNum() == 2:
            footPoint.translate2D(lineSegVerticalVec.vector)
        elif footPoint.getDimensionNum() == 3:
            footPoint.translate(lineSegVerticalVec.vector)
        else:
            raise ValueError("[ERROR] There is something wrong in the dimension of footPoint")

        # # 警告
        # warn("[WARNING] The footPoint is at the line strictly, and it has been translated along a proper vector")

        # 从 点 到 垂足 的 线段
        verticalLineSeg = LineSeg([point, footPoint])

        # 如果 这一垂线段的 长度 比较小
        if verticalLineSeg.length < lengthThreshold:
            # 则 将其两端各延伸 extendingLength这样的长度
            verticalLineSeg = cls.extendLineSegAlongTwoDirections(verticalLineSeg, extendingValue)
            # 并且 报警告
            warn("[WARNING] The length of verticalLineSeg is too short, and the verticalLineSeg has been extended at two directions")

        # 返回
        return verticalLineSeg


    @classmethod
    def getVerticalLineSegFromPointToLine(cls, point: List[float], lineSeg: LineSeg, lengthThreshold: float=1e2 * Utils.Threshold, extendingValue: float=10.0) -> LineSeg:
        """
        获取 从 点 到 直线 的 垂线段
        Args:
            point: 点
            lineSeg: 线段 表征 直线
        Returns: 垂线段
        """
        # 断言 线段 起点 终点 不重合 | 严格地
        # assert lineSeg.length != 0.0
        assert lineSeg.startPoint != lineSeg.endPoint

        # 从 点 到 线段所在直线的 垂线段 的 垂足
        footPoint = Point(cls.getFootPointFromPointToLine(point, lineSeg))

        # # 如果 这一点 就在 线段所在的直线上
        # if cls.isPointAtLine(footPoint, lineSeg, isLoose=True, threshold=Utils.Threshold):
        # 如果 这一垂足 与 点 重合 | 严格地
        if footPoint == point:
            # 从 该点出发的 该直线的 单位垂直向量
            lineSegVerticalVec = cls.getUnitVerticalVectorFromPointToLine(footPoint, lineSeg)

            # 平移
            if footPoint.getDimensionNum() == 2:
                footPoint.translate2D(-lineSegVerticalVec.vector)
            elif footPoint.getDimensionNum() == 3:
                footPoint.translate(-lineSegVerticalVec.vector)
            else:
                raise ValueError("[ERROR] There is something wrong in the dimension of footPoint")

            # 警告
            warn("[WARNING] The footPoint is at the line strictly, and it has been translated along a proper vector")

        # 从 点 到 垂足 的 线段
        verticalLineSeg = LineSeg([point, footPoint])

        # 如果 这一垂线段的 长度 比较小
        if verticalLineSeg.length < lengthThreshold:
            # 则 将其两端各延伸 extendingLength这样的长度
            verticalLineSeg = cls.extendLineSegAlongTwoDirections(verticalLineSeg, extendingValue)
            # 并且 报警告
            warn("[WARNING] The length of verticalLineSeg is too short, and the verticalLineSeg has been extended at two directions")

        # 返回
        return verticalLineSeg


    @classmethod
    def getVerticalLineFromPointToLineSeg(cls, point: List[float], lineSeg: LineSeg) -> LineSeg:
        """
        获取 从 点 到 线段 的 垂线段
        Args:
            point: 点
            lineSeg: 线段
        Returns: 垂线段 或者 None
        """
        # 从 点 到 线段所在直线 的 垂线段
        verticalLineSeg = cls.getVerticalLineFromPointToLine(point, lineSeg)

        # 垂线段 与 线段 是否存在公共点
        # if cls.areTwoLineSegsIntersect(verticalLineSeg, lineSeg):
        # 如果 点 在 线段 上
        if cls.isPointAtLineSeg(point, lineSeg):
            return verticalLineSeg
        else:
            return None

        # # 计算 从 点 到 线段 的 垂线段 的 垂足
        # footPoint = cls.getFootPointFromPointToLineSeg(point, lineSeg)
        #
        # # 如果 垂足 不是 None
        # if not (footPoint is None):
        #     # 实例化 垂线段
        #     verticalLineSeg = LineSeg([point, footPoint])
        #     # 返回 垂线段
        #     return verticalLineSeg
        # # 如果 垂足 是 None
        # else:
        #     # 返回 None
        #     return None


    @classmethod
    def getVerticalLineSegFromPointToLineSeg(cls, point: List[float], lineSeg: LineSeg, lengthThreshold: float=1e2 * Utils.Threshold, extendingValue: float=10.0) -> LineSeg:
        """
        获取 从 点 到 直线 的 垂线段
        Args:
            point: 点
            lineSeg: 线段 表征 直线
        Returns: 垂线段
        """
        # 断言 线段 起点 终点 不重合 | 严格地
        # assert lineSeg.length != 0.0
        assert lineSeg.startPoint != lineSeg.endPoint

        # 从 点 到 线段所在直线的 垂线段 的 垂足
        footPoint = cls.getFootPointFromPointToLineSeg(point, lineSeg)

        # 如果 垂足 不在 线段上
        if footPoint is None:
            return None

        # 实例化footPoint为Point类型
        footPoint = Point(footPoint)

        # 计算 垂线段
        verticalLineSeg = cls.getVerticalLineSegFromPointToLine(point, lineSeg)

        # 返回
        return verticalLineSeg


    @staticmethod
    def getVertexOnLineSegThatNEGivenPoint(givenPoint: Point, lineSeg: LineSeg) -> List[float]:
        """
        获取 线段上 不等于 给定点的 端点 （起点优先输出）
        Args:
            givenPoint: 给定的点
            lineSeg: 线段
        Returns: 满足条件的端点
        """
        startPoint = lineSeg.startPoint
        endPoint = lineSeg.endPoint
        if not startPoint.isEqualToAnother(givenPoint):
            return startPoint
        elif not endPoint.isEqualToAnother(givenPoint):
            return endPoint
        # else: #注：LineSeg类型现在允许起点与终点重合
        #     raise ValueError("[ERROR] Invalid value of \"givenPoint\", because it equals to startPoint and endpoint of lineSeg simultaneously")
        else:
            return None


    # @classmethod #原本位于Utils中，现在搬运至AssemblyAlgos2D
    # def isFourPointsClockwiseOrAnticlockwise(cls, pt0: List[float], pt1: List[float], pt2: List[float], pt3: List[float]) -> bool:
    #     """
    #     判断 四个点的排列顺序 是否为 逆时针顺序 或 顺时针顺序
    #     Args:
    #         pt0: 点0
    #         pt1: 点1
    #         pt2: 点2
    #         pt3: 点3
    #     Returns: 四个点的排列顺序 是否为 逆时针顺序 或 顺时针顺序
    #     """
    #     return cls.isEqual(
    #         cls.calcAreaOfPolygon([pt0[0], pt1[0], pt2[0], pt3[0]], [pt0[1], pt1[1], pt2[1], pt3[1]]),
    #         np.linalg.norm(Utils.cross([[pt0[0], pt0[1]], [pt1[0], pt1[1]]], [[pt0[0], pt0[1]], [pt2[0], pt2[1]]]))
    #     )


    @staticmethod
    def getVecsFromAGroupOfPointsSequentially(points: List[List[float]]) -> List[Vector]:
        """
        按照顺序地 获取 一组点从头至尾生成的向量
        Args:
            points: 存储点的容器
        Returns: 这组点从头至尾生成的向量
        """
        # 点的数量
        cnt = len(points)

        # 向量组
        vecs = []
        for idx, pnt in enumerate(points):
            vecs.append(Vector([points[idx], points[(idx + 1) % cnt]]))
        assert len(vecs) == cnt

        # 返回
        return vecs


    @staticmethod
    def getCrossProductsFromAGroupOfVecsSequentially(vecs: List[Vector]) -> List[np.ndarray]:
        """
        按照顺序地 获取 一组向量从头至尾生成的张量积
        Args:
            vecs: 存储向量的容器
        Returns: 这组向量从头至尾生成的张量积
        """
        # 向量的数量
        cnt = len(vecs)

        # 叉积组
        crossProducts = []
        for idx, vec in enumerate(vecs):
            crossProducts.append(CommonAlgos2D.cross(vecs[idx], vecs[(idx + 1) % cnt]))
        assert len(crossProducts) == cnt

        # 返回
        return crossProducts


    @classmethod
    def arePointsAntiClockwise(cls, points: List[List[float]]) -> bool:
        """
        判断 一组点的顺序 是否为 逆时针顺序
        Args:
            points: 存储点的容器
        Returns: 这组点的顺序 是否为 逆时针顺序
        """
        # # 这个方法仅适用于凸多边形
        # # 获取 向量组
        # vecs = cls.getVecsFromAGroupOfPointsSequentially(points)
        # # 获取 张量积组
        # crossProducts = cls.getCrossProductsFromAGroupOfVecsSequentially(vecs)
        # # 返回 TODO: 等于零的情况是否合理？
        # return all(crossProduct >= 0.0 for crossProduct in crossProducts) #不能写np.linalg.norm(crossProduct)，因为它一定是非负的

        # 面积法 判定 逆时针 点序 #TODO: 这一方法还是存在问题
        coordX = [point[0] for point in points]
        coordY = [point[1] for point in points]
        return Utils.calcAreaOfPolygon(coordX, coordY, isAbs=False) >= 0.0


    @classmethod
    def arePointsClockwise(cls, points: List[List[float]]) -> bool:
        """
        判断 一组点的顺序 是否为 顺时针顺序
        Args:
            points: 存储点的容器
        Returns: 这组点的顺序 是否为 顺时针顺序
        """
        # # 这个方法仅适用于凸多边形
        # # 获取 向量组
        # vecs = cls.getVecsFromAGroupOfPointsSequentially(points)
        # # 获取 张量积组
        # crossProducts = cls.getCrossProductsFromAGroupOfVecsSequentially(vecs)
        # # 返回 TODO: 等于零的情况是否合理？
        # return all(crossProduct <= 0.0 for crossProduct in crossProducts) #不能写np.linalg.norm(crossProduct)，因为它一定是非负的

        # 面积法 判定 顺时针 点序 #TODO: 这一方法还是存在问题
        coordX = [point[0] for point in points]
        coordY = [point[1] for point in points]
        return Utils.calcAreaOfPolygon(coordX, coordY, isAbs=False) < 0.0


    @classmethod #arePointsAntiClockwiseOrClockwise #TODO: 改方法名及其调用
    def arePointsAnticlockwiseOrClockwise(cls, points: List[List[float]]) -> bool:
        """
        判断 一组点的顺序 是否为 逆时针顺序 或 顺时针顺序
        Args:
            points: 存储点的容器
        Returns: 这组点的顺序 是否为 逆时针顺序 或 顺时针顺序
        """
        return cls.arePointsAntiClockwise(points) or cls.arePointsClockwise(points)


    @staticmethod
    def areTwoBoundingBoxesOverlap(quadrangeA: Quadrangle, quadrangeB: Quadrangle) -> bool:
        """
        两个包围盒是否有重叠部分
        :param quadrangeA: 包围盒A
        :param quadrangeB: 包围盒B
        :return: bool
        """
        # 断言
        assert quadrangeA.isRectangle() and quadrangeA.isOneLinesegHorizontal()
        assert quadrangeB.isRectangle() and quadrangeB.isOneLinesegHorizontal()

        # 判断
        xminQuadrangeA, yminQuadrangeA, xmaxQuadrangeA, ymaxQuadrangeA = quadrangeA.getBoundCoord()
        xminQuadrangeB, yminQuadrangeB, xmaxQuadrangeB, ymaxQuadrangeB = quadrangeB.getBoundCoord()
        return (
                max(xminQuadrangeA, xmaxQuadrangeA) >= min(xminQuadrangeB, xmaxQuadrangeB) and
                max(xminQuadrangeB, xmaxQuadrangeB) >= min(xminQuadrangeA, xmaxQuadrangeA) and
                max(yminQuadrangeA, ymaxQuadrangeA) >= min(yminQuadrangeB, ymaxQuadrangeB) and
                max(yminQuadrangeB, ymaxQuadrangeB) >= min(yminQuadrangeA, ymaxQuadrangeA)
        )


    @classmethod
    def areTwoLineSegsExclusive(cls, lineSegA, lineSegB):
        """
        两条线段是否具有公共点的快速排斥
        :param lineSegA:
        :param lineSegB:
        :return: Bool
        """
        return not cls.areTwoBoundingBoxesOverlap(
            cls.boundCoord2Quadrangle(lineSegA.getBoundCoord()),
            cls.boundCoord2Quadrangle(lineSegB.getBoundCoord())
        )


    @staticmethod
    def calcProdOfTwoCrossProdsInStraddleForTwoLineSegs(xStart_yStart_xEnd_yEnd_A: Tuple[float, float, float, float],
                                                        xStart_yStart_xEnd_yEnd_B: Tuple[float, float, float, float],
                                                        isLoose: bool = True) -> float:
        """
        计算 两条线段 在跨立检验中 的 两个张量积 的 乘积
        Args:
            xStart_yStart_xEnd_yEnd_A: 预计的 被跨立的 线段
            xStart_yStart_xEnd_yEnd_B: 预计的 跨立 线段
        Returns: 两条线段 在跨立检验中 的 两个张量积 的 乘积
        """
        # 坐标获取
        xStartLineSegA, yStartLineSegA, xEndLineSegA, yEndLineSegA = xStart_yStart_xEnd_yEnd_A
        xStartLineSegB, yStartLineSegB, xEndLineSegB, yEndLineSegB = xStart_yStart_xEnd_yEnd_B

        # (startA -> startB) x (startA -> endA)
        sAsB_x_sAeA = CommonAlgos2D.cross(
            [xStartLineSegB - xStartLineSegA, yStartLineSegB - yStartLineSegA],
            [xEndLineSegA - xStartLineSegA, yEndLineSegA - yStartLineSegA]
        )

        # (startA -> endB) x (startA -> endA)
        sAeB_x_sAeA = CommonAlgos2D.cross(
            [xEndLineSegB - xStartLineSegA, yEndLineSegB - yStartLineSegA],
            [xEndLineSegA - xStartLineSegA, yEndLineSegA - yStartLineSegA]
        )

        # 小于阈值者，赋值为0.0 #否则乘出来可能为有限值，而不是0.0
        if isLoose:
            sAsB_x_sAeA = 0.0 if sAsB_x_sAeA < Utils.Threshold else sAsB_x_sAeA
            sAeB_x_sAeA = 0.0 if sAeB_x_sAeA < Utils.Threshold else sAeB_x_sAeA

        # 返回
        return sAsB_x_sAeA * sAeB_x_sAeA


    @classmethod
    def areTwoLineSegsIntersect(cls, lineSegA: LineSeg, lineSegB: LineSeg) -> bool:
        """
        两条线段是否具有公共点
        :param lineSegA: 线段A
        :param lineSegB: 线段B
        :return: Bool
        """
        # 快速排斥
        if cls.areTwoLineSegsExclusive(lineSegA, lineSegB):
            return False

        pass
        # # 跨立检验
        # xStartLineSegA, yStartLineSegA, xEndLineSegA, yEndLineSegA = lineSegA.getStartEndCoord2D()
        # xStartLineSegB, yStartLineSegB, xEndLineSegB, yEndLineSegB = lineSegB.getStartEndCoord2D()
        #
        # # (startA -> startB) x (startA -> endA)
        # sAsB_x_sAeA = CommonAlgos2D.cross(
        #     [xStartLineSegB - xStartLineSegA, yStartLineSegB - yStartLineSegA],
        #     [xEndLineSegA - xStartLineSegA, yEndLineSegA - yStartLineSegA]
        # )
        #
        # # (startA -> endB) x (startA -> endA)
        # sAeB_x_sAeA = CommonAlgos2D.cross(
        #     [xEndLineSegB - xStartLineSegA, yEndLineSegB - yStartLineSegA],
        #     [xEndLineSegA - xStartLineSegA, yEndLineSegA - yStartLineSegA]
        # )
        #
        # # (startB -> startA) x (startB -> endB)
        # sBsA_x_sBeB = CommonAlgos2D.cross(
        #     [xStartLineSegA - xStartLineSegB, yStartLineSegA - yStartLineSegB],
        #     [xEndLineSegB - xStartLineSegB, yEndLineSegB - yStartLineSegB]
        # )
        #
        # # (startB -> endA) x (startB -> endB)
        # sBeA_x_sBeB = CommonAlgos2D.cross(
        #     [xEndLineSegA - xStartLineSegB, yEndLineSegA - yStartLineSegB],
        #     [xEndLineSegB - xStartLineSegB, yEndLineSegB - yStartLineSegB]
        # )
        #
        # return (sAsB_x_sAeA * sAeB_x_sAeA <= 0.0) and (sBsA_x_sBeB * sBeA_x_sBeB <= 0.0)
        pass

        # 跨立检验
        prodOfTwoCrossProdsAB = cls.calcProdOfTwoCrossProdsInStraddleForTwoLineSegs(lineSegA.getStartEndCoord2D(),
                                                                                    lineSegB.getStartEndCoord2D())
        prodOfTwoCrossProdsBA = cls.calcProdOfTwoCrossProdsInStraddleForTwoLineSegs(lineSegB.getStartEndCoord2D(),
                                                                                    lineSegA.getStartEndCoord2D())
        return (prodOfTwoCrossProdsAB <= 0.0) and (prodOfTwoCrossProdsBA <= 0.0)


    @classmethod
    def isLineAndLineSegIntersect(cls, lineCoeffs: Tuple[float, float, float], lineSeg: LineSeg) -> bool:
        """
        判断 直线 与 线段 是否 具有公共点
        Args:
            lineCoeffs: 直线line的一般方程系数
            lineSeg: 线段
        Returns: line 与 lineSeg 是否具有 公共点
        """
        # 获取 直线line的一般方程系数
        A, B, C = lineCoeffs

        # 直线line上的两点
        point0AtLine = [None, None]
        point1AtLine = [None, None]

        # 如果B不为零
        if not Utils.isEqual(B, 0.0, 1e-6):
            point0AtLine = [0.0, -C / B]
            point1AtLine = [1.0, (-A - C) / B]
        # 如果A不为零
        elif not Utils.isEqual(A, 0.0, 1e-6):
            point0AtLine = [-C / A, 0.0]
            point1AtLine = [(-B - C) / A, 1.0]
        # 其它
        else:
            raise ValueError("[ERROR] A, B can not be equal to 0.0, either")

        # line被lineSeg跨立 的 检验
        xStart, yStart, xEnd, yEnd = point0AtLine + point1AtLine
        prodOfTwoCrossProds = cls.calcProdOfTwoCrossProdsInStraddleForTwoLineSegs((xStart, yStart, xEnd, yEnd), lineSeg.getStartEndCoord2D())
        return prodOfTwoCrossProds <= 0.0


    @classmethod
    def calcIntersectionPointOfTwoLines(cls, lineSegA: LineSeg, lineSegB: LineSeg) -> List[float]:
        """
        计算 两个直线的交点
        Args:
            lineSegA: 线段A 表征 直线A
            lineSegB: 线段B 表征 线段B
        Returns: 交点
        """
        # 断言 两个线段不平行 （存在唯一解）
        # try:
        assert not lineSegA.isParallelToAnother(lineSegB)
        # except:
        #     # from matplotlib import pyplot as plt
        #     # fig = plt.figure()
        #     # xs = [lineSegA.startPoint.x, lineSegA.endPoint.x]
        #     # ys = [lineSegA.startPoint.y, lineSegA.endPoint.y]
        #     # plt.plot(xs, ys, 'b')
        #     # xs = [lineSegB.startPoint.x, lineSegB.endPoint.x]
        #     # ys = [lineSegB.startPoint.y, lineSegB.endPoint.y]
        #     # plt.plot(xs, ys, 'r--')
        #     # plt.axis("equal")
        #     # plt.show()
        #     # print(lineSegA)
        #     # print(lineSegB)
        #     assert not lineSegA.isParallelToAnother(lineSegB)

        # 计算两个线段所在直线的方程系数
        coeffsA = Utils.getCoeffsOfLineEquation(lineSegA.startPoint, lineSegA.endPoint)
        coeffsB = Utils.getCoeffsOfLineEquation(lineSegB.startPoint, lineSegB.endPoint)

        # 计算两个直线的交点
        x, y = Utils.calcIntersectionPointOfTwoLines(coeffsA, coeffsB)

        # 返回
        return Point([x, y])


    @classmethod
    def calcIntersectionPointOfTwoLineSegs(cls, lineSegA: LineSeg, lineSegB: LineSeg) -> List[float]:
        """
        计算两个线段的交点
        Args:
            lineSegA: 线段A
            lineSegB: 线段B
        Returns: 线段A 和 线段B 的 交点
        """
        # # 断言 两个线段存在公共点
        # assert cls.areTwoLineSegsIntersect(lineSegA, lineSegB)

        # 如果 两个线段不存在公共点
        if not cls.areTwoLineSegsIntersect(lineSegA, lineSegB):
            return list()
        # from matplotlib import pyplot as plt
        # fig = plt.figure()
        # xs = [lineSegA.startPoint.x, lineSegA.endPoint.x]
        # ys = [lineSegA.startPoint.y, lineSegA.endPoint.y]
        # plt.plot(xs, ys, 'b')
        # xs = [lineSegB.startPoint.x, lineSegB.endPoint.x]
        # ys = [lineSegB.startPoint.y, lineSegB.endPoint.y]
        # plt.plot(xs, ys, 'r')
        # plt.axis("equal")
        # plt.show()

        # # 断言 两个线段不平行
        # assert not lineSegA.isParallelToAnother(lineSegB)
        #
        # # 计算两个线段所在直线的方程系数
        # coeffsA = Utils.getCoeffsOfLineEquation(lineSegA.startPoint, lineSegA.endPoint)
        # coeffsB = Utils.getCoeffsOfLineEquation(lineSegB.startPoint, lineSegB.endPoint)
        #
        # # 计算两个直线的交点
        # x, y = Utils.calcIntersectionPointOfTwoLines(coeffsA, coeffsB)
        #
        # # 返回
        # return Point([x, y])

        # 如果 两个线段存在公共点 | 则两个线段的交点 就是它们所在直线的交点
        return cls.calcIntersectionPointOfTwoLines(lineSegA, lineSegB)


    @classmethod
    def isOneLineSegIncludedInAnother(cls, lineSegA, lineSegB):
        """
        判断lineSegA是否包含于lineSegB中
        :param lineSegA:
        :param lineSegB:
        :return: Bool
        """
        return cls.isPointAtLineSeg(lineSegA[0], lineSegB) and cls.isPointAtLineSeg(lineSegA[1], lineSegB)


    @classmethod
    def isOneLineSegEqualToAnother(cls, lineSegA, lineSegB):
        """
        判断lineSegA, lineSegB是否完全相等
        :param lineSegA:
        :param lineSegB:
        :return: Bool
        """
        return cls.isOneLineSegIncludedInAnother(lineSegA, lineSegB) and cls.isOneLineSegIncludedInAnother(lineSegB, lineSegA)


    @staticmethod
    def calcDistanceBetweenTwoPoints(pointA: Point, pointB: Point) -> float:
        """
        计算两点之间的距离
        :param pointA:
        :param pointB:
        :return: float
        """
        # 这个实现对于极大的数值不安全 #极大数值越界后会产生一个小于零的数值
        # return np.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

        # 这个实现更安全
        return Utils.calcDistanceBetweenTwoPoints(pointA, pointB)


    # @staticmethod #TODO: 默认中间两点两矩形是对应相等的
    # def fuseTowQuadranglesOverlaped(quadrangeA, quadrangeB):
    #     """
    #     quadrangeA在下方，quadrangeB在上方
    #     :param quadrangeA:
    #     :param quadrangeB:
    #     :return: Quadrangle
    #     """
    #     # quadrangle = Quadrangle([
    #     # face = Polygon([
    #     face = PolyLineSeg([
    #         quadrangeA[0],
    #         quadrangeA[1],
    #         quadrangeA[2],
    #         quadrangeB[1],
    #         quadrangeB[2],
    #         quadrangeB[3],
    #         quadrangeB[0],
    #         quadrangeA[3],
    #         quadrangeA[0]
    #     ])
    #
    #     # # if Utils.isEqual(quadrangle[2][0], quadrangle[3][0]) and \
    #     # #         Utils.isEqual(quadrangle[2][1], quadrangle[3][1]): #TODO: 判断点的重合的逻辑
    #     # #     quadrangle.pop(2)
    #     # #     quadrangle.pop(3)
    #     #
    #     # # if Utils.isEqual(quadrangle[-2][0], quadrangle[-3][0]) and \
    #     # #         Utils.isEqual(quadrangle[-2][1], quadrangle[-3][1]):  # TODO: 判断点的重合的逻辑
    #     # #     quadrangle.pop(-2)
    #     # #     quadrangle.pop(-3)
    #     #
    #     # # quadrangle.pop(-1)
    #     #
    #     # return quadrangle
    #
    #     return AssemblyAlgos2D.boundCoord2Quadrangle(face.getBoundCoord())


    @classmethod
    def fuseTowHorizontalRectsThatShareOneLineSeg(cls, rectA: Quadrangle, rectB: Quadrangle):
        # 它们都是矩形
        assert rectA.isRectangle() and rectB.isRectangle()

        # 它们都是水平放置的
        assert rectA.isOneLinesegHorizontal() and rectB.isOneLinesegHorizontal()

        # 它们有且仅有一条边重合
        oneLineSegOverlapedCount = 0
        for edgeA in rectA:
            for edgeB in rectB:
                if not cls.isOneLineSegEqualToAnother(edgeA, edgeB):
                    continue
                oneLineSegOverlapedCount += 1
        assert oneLineSegOverlapedCount == 1

        # 获取边界坐标
        coordXValues = []
        coordXValues += rectA.getAllCoordX()
        coordXValues += rectB.getAllCoordX()
        coordYValues = []
        coordYValues += rectA.getAllCoordY()
        coordYValues += rectB.getAllCoordY()

        # 这样做的前提是rectA, rectB都是水平放置的
        xmin = min(coordXValues)
        ymin = min(coordYValues)
        xmax = max(coordXValues)
        ymax = max(coordYValues)

        # 构造Quadrangle
        return cls.boundCoord2Quadrangle((xmin, ymin, xmax, ymax))


    @staticmethod
    def extendLineSegAlongTwoDirections(lineSeg: LineSeg, extendingLength: float) -> LineSeg:
        """
        沿着线段的 两个方向 延伸 线段
        Returns: 延伸后的 线段
        """
        # 断言 线段 起点 终点 不重合 | 严格地
        # assert lineSeg.length != 0.0
        assert lineSeg.startPoint != lineSeg.endPoint
        
        # 起点指向终点的向量 单位化 乘 每条矩形的 半 宽度
        vecStartToEnd = Vector([lineSeg.startPoint, lineSeg.endPoint])
        vecStartToEnd.normalize()
        vecStartToEnd.vector *= extendingLength

        # 终点指向起点的向量 单位化 乘 每条矩形的 半 宽度
        vecEndToStart = Vector([lineSeg.endPoint, lineSeg.startPoint])
        vecEndToStart.normalize()
        vecEndToStart.vector *= extendingLength

        # 起点副本
        startPointCopy = deepcopy(lineSeg.startPoint)

        # 平移 起点副本
        if startPointCopy.getDimensionNum() == 2 and \
                vecStartToEnd.getDimensionNum() == 2:
            startPointCopy.translate2D(vecStartToEnd.vector)
        elif startPointCopy.getDimensionNum() == 3 and \
                vecStartToEnd.getDimensionNum() == 3:
            startPointCopy.translate(vecStartToEnd.vector)
        else:
            raise ValueError("[ERROR] There is something wrong in dimensions")

        # 终点副本
        endPointCopy = deepcopy(lineSeg.endPoint)

        # 平移 终点副本
        if endPointCopy.getDimensionNum() == 2 and \
                vecEndToStart.getDimensionNum() == 2:
            endPointCopy.translate2D(vecEndToStart.vector)
        elif endPointCopy.getDimensionNum() == 3 and \
                vecEndToStart.getDimensionNum() == 3:
            endPointCopy.translate(vecEndToStart.vector)
        else:
            raise ValueError("[ERROR] There is something wrong in dimensions")

        # 实例化 延伸后的 线段
        extendedLineSeg = LineSeg([startPointCopy, endPointCopy])

        # 返回
        return extendedLineSeg


    @classmethod
    def expandLineSegTowardsSidesIntoRectangle(cls, lineSeg: LineSeg, expandWidth: float) -> Quadrangle:
        """
        将 线段 沿着与线段垂直的方向 向两侧分别扩展expandWidth宽度 形成一个矩形
        Args:
            lineSeg: 线段
            expandWidth: 单侧扩展宽度
        Returns: 扩展后形成的矩形
        """
        # 线段的方向向量
        lineSegDirectionVec = lineSeg.getDirectionVec()

        # 与 线段的方向向量 垂直的 两个垂直向量
        lineSegVerticalVec0, lineSegVerticalVec1 = Utils.getVerticalVectors(lineSegDirectionVec)

        # 将 两个垂直向量 实例化为Vector
        lineSegVerticalVec0 = Vector.generateVectorFromPointAndDirection(lineSeg.startPoint, lineSegVerticalVec0)
        lineSegVerticalVec1 = Vector.generateVectorFromPointAndDirection(lineSeg.startPoint, lineSegVerticalVec1)

        # 修改 两个垂直向量的 .vector的 模长
        lineSegVerticalVec0.changeNorm(expandWidth)
        lineSegVerticalVec1.changeNorm(expandWidth)

        # 按照 两个垂直向量 平移 原线段 产生两个新的线段
        translatedLineSeg0 = deepcopy(lineSeg)
        translatedLineSeg0.translate2D(lineSegVerticalVec0.vector)
        translatedLineSeg1 = deepcopy(lineSeg)
        translatedLineSeg1.translate2D(lineSegVerticalVec1.vector)

        # 实例化 扩展后的 矩形
        lineSegExpandedQuadrangle = Quadrangle([translatedLineSeg0.startPoint, translatedLineSeg0.endPoint, translatedLineSeg1.endPoint, translatedLineSeg1.startPoint])

        # 断言 lineSegExpandedQuadrangle的四个顶点沿着 逆时针 或 顺时针 方向
        assert cls.arePointsAnticlockwiseOrClockwise([
            lineSegExpandedQuadrangle.vertexDA,
            lineSegExpandedQuadrangle.vertexAB,
            lineSegExpandedQuadrangle.vertexBC,
            lineSegExpandedQuadrangle.vertexCD
        ])

        # # 断言 lineSegExpandedQuadrangle 为矩形
        # assert lineSegExpandedQuadrangle.isRectangle() #TODO: 恢复isRectangle()

        # 返回
        return lineSegExpandedQuadrangle


    @staticmethod
    def removeDuplicatedPoints(points: List[List[float]], isLoose: bool = True) -> List[List[float]]:
        """
        对于一组点，将其中的重复点去除
        Args:
            points: 容器 存储 点 | 待去重复的点
        Returns: 去重复后的点容器
        """
        # 判空
        if len(points) == 0:
            return points

        # 点集数组
        pointsArray = PointTensorConverter.convert(points)

        # 生成 点 是否重合 的 矩阵
        pointTensorCoincideWithPointTensorInspector = PointTensorCoincideWithPointTensorInspector(pointsArray, pointsArray)
        isCoincidedMat = pointTensorCoincideWithPointTensorInspector.calcIsCoincidedMatForTwoPointTensors(isLoose=isLoose)

        # 建立无向图
        undirectedGraph = UndirectedGraph(isCoincidedMat)
        classificationRes = undirectedGraph.classify()

        # 唯一索引集合
        uniqueIdxes = set()
        for class_ in classificationRes:
            assert not (class_[0] in uniqueIdxes)
            uniqueIdxes.add(class_[0])

        # uniquePoints
        uniquePoints = []
        for uniqueIdx in uniqueIdxes:
            uniquePoints.append(points[uniqueIdx])

        # 返回
        return uniquePoints


    @classmethod
    def calcCrossProdValueForThreePoints(cls, points: List[List[float]]) -> np.ndarray:
        """
        计算 三点之间的张量积 | 以首点为公共起点
        Args:
            points: 集合 存储 三个点
        Returns: 张量积
        """
        # 断言 点的数量为3
        assert len(points) == 3

        # 计算 张量积
        crossProdValue = Utils.cross(
            [[points[0][0], points[0][1]],
             [points[1][0], points[1][1]]],

            [[points[0][0], points[0][1]],
             [points[2][0], points[2][1]]]
        )

        # 返回
        return crossProdValue


    @classmethod
    def calcCrossProdValueForConvexHull(cls, convexHull: List[List[float]], numOfPntsOnConvexHull: int, points: List[List[float]], pointIdx: int) -> float:
        """
        就算 单调栈 中的 张量积
        Args:
            convexHull: 凸包
            numOfPntsOnConvexHull: 凸包上的点的数量
            points: 点集合
            pointIdx: 点索引
        Returns: crossProdValue 张量积数值 | 数值，不是模长
        """
        # 计算 张量积
        # crossProdValue = Utils.cross(
        #                     [[convexHull[numOfPntsOnConvexHull - 2][0], convexHull[numOfPntsOnConvexHull - 2][1]], [convexHull[numOfPntsOnConvexHull - 1][0], convexHull[numOfPntsOnConvexHull - 1][1]]],
        #                     [[convexHull[numOfPntsOnConvexHull - 2][0], convexHull[numOfPntsOnConvexHull - 2][1]], [points[pointIdx][0], points[pointIdx][1]]]
        #                 )
        # TODO: 这样做应该是合理的
        crossProdValue = cls.calcCrossProdValueForThreePoints([
            convexHull[numOfPntsOnConvexHull - 2],
            convexHull[numOfPntsOnConvexHull - 1],
            points[pointIdx]
        ])

        # 断言 张量积的形状长度 为零
        assert len(crossProdValue.shape) == 0

        # 转换 张量积 为 浮点型 | 获得 此张量积的数值
        crossProdValue = (float)(crossProdValue)

        # 返回
        return crossProdValue


    @classmethod
    def removePointsOnLineSegForConvexHull(cls, points: List[List[float]]) -> List[List[float]]:
        """
        去除 那些位于线段上的点 | 为了凸包
        Args:
            points: 点集合
        Returns: 去除 指定点 之后 的 点集合
        """
        # 栈 存储 点
        pointsStk = []

        # 遍历 点集合中的 每一个点
        for point in points:
            while (len(pointsStk) >= 2) and Utils.isLessEqual(cls.calcCrossProdValueForThreePoints([pointsStk[-2], pointsStk[-1], point]), 0.0, Utils.Threshold):
                pointsStk.pop()
            pointsStk.append(point)

        # 返回
        return pointsStk


    @classmethod
    def getConvexHull(cls, points: List[List[float]], allowPointOnLineSegInwardly: bool = False) -> List[List[float]]:
        """
        获取一组点的凸包
        Args:
            points: 容器 存储 点
            allowPointOnLineSegInwardly: 是否允许点位于线段的内部（两个端点之间）
        Returns: 这一组点的凸包
        """
        # 对于 原来的点集合 去重复
        points = cls.removeDuplicatedPoints(points)

        # 排序
        points = Utils.mergeSortPoints(points)

        # 点的数量
        pointsNum = len(points)

        # 容器 存储 凸包上的点
        convexHull = [[0.0, 0.0]] * pointsNum

        # 凸包上的点的数量 #栈顶指针
        numOfPntsOnConvexHull = 0

        # 求下凸包
        for i in range(0, pointsNum, 1):
            while (numOfPntsOnConvexHull > 1) and (cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i) <= 0.0 if (not allowPointOnLineSegInwardly) else cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i) < 0.0):
            # while (numOfPntsOnConvexHull > 1) and (Utils.isLessEqual(cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i), 0.0, Utils.Threshold) if (not allowPointOnLineSegInwardly) else cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i) < 0.0):
                # print("============", cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i))
                numOfPntsOnConvexHull -= 1
            convexHull[numOfPntsOnConvexHull] = points[i]
            numOfPntsOnConvexHull += 1

        # 凸包上已经存在的点的数量
        numOfPntsOnLowerConvexHull = numOfPntsOnConvexHull

        # 求上凸包
        for i in range(pointsNum - 2, -1, -1):
            while (numOfPntsOnConvexHull > numOfPntsOnLowerConvexHull) and (cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i) <= 0.0 if (not allowPointOnLineSegInwardly) else cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i) < 0.0):
            # while (numOfPntsOnConvexHull > numOfPntsOnLowerConvexHull) and (Utils.isLessEqual(cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i), 0.0, Utils.Threshold) if (not allowPointOnLineSegInwardly) else cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i) < 0.0):
                # print("============", cls.calcCrossProdValueForConvexHull(convexHull, numOfPntsOnConvexHull, points, i))
                numOfPntsOnConvexHull -= 1
            # 为convexHull扩展容量
            while numOfPntsOnConvexHull >= len(convexHull):
                convexHull.append([0.0, 0.0])
            convexHull[numOfPntsOnConvexHull] = points[i]
            numOfPntsOnConvexHull += 1

        # 凸包上的点的数量 | 因为最后一轮循环会多加一
        if pointsNum > 1:
            numOfPntsOnConvexHull -= 1

        # 根据凸包上的点的数量截取列表
        convexHull = convexHull[:numOfPntsOnConvexHull]

        # # 去除线段上的点 #TODO: 由于不明原因，那两个allowPointOnLineSegInwardly的地方，排除线段上的点的功能可能会失效，所以这里单独做一个功能
        # if not allowPointOnLineSegInwardly:
        #     # print("allowPointOnLineSegInwardly: {}".format(convexHull))
        #     convexHull = cls.removePointsOnLineSegForConvexHull(convexHull)  # points = cls.removePointsOnLineSegForConvexHull(points)

        # 返回
        return convexHull


    @classmethod
    def isPolygonConvex(cls, polygon: Polygon) -> bool:
        """
        判断 该多边形是否为 凸多边形
        Args:
            polygon: 多边形
        Returns: 该多边形是否为 凸多边形
        """
        vertexes = Utils.removeRepeatedPoints(polygon.vertexes)
        return len(cls.getConvexHull(vertexes)) == len(vertexes)


    @classmethod
    def __getRectangleFromLineSegAndThreePoints(cls, lineSeg: LineSeg, farthestVertex: Point, positiveMarginalVertex: Point, negativeMarginalVertex: Point) -> Quadrangle:
        """
        从 线段 和 三个点 获得 矩形
        Args:
            lineSeg: 线段
            farthestVertex: 最远点
            positiveMarginalVertex: 最正向点
            negativeMarginalVertex: 最反向点
        Returns: 这一矩形
        """
        # 断言 最远点 都不在 线段上
        assert not cls.isPointAtLineSeg(farthestVertex, lineSeg)

        # 最正向边
        positiveMarginalLineSeg = cls.getVerticalLineFromPointToLine(positiveMarginalVertex, lineSeg)

        # 最正向边 与 lineSeg 的 公共点
        vertex0 = cls.getFootPointFromPointToLine(positiveMarginalVertex, lineSeg)

        # 最负向边
        negativeMarginalLineSeg = cls.getVerticalLineFromPointToLine(negativeMarginalVertex, lineSeg)

        # 最负向边 与 lineSeg 的 公共点
        vertex3 = cls.getFootPointFromPointToLine(negativeMarginalVertex, lineSeg)

        # 断言 最正向边 与 最负向边 平行
        assert positiveMarginalLineSeg.isParallelToAnother(negativeMarginalLineSeg, threshold=1e5 * Utils.Threshold)

        # 最远边 与 最正向边 的 公共点
        vertex1 = cls.getFootPointFromPointToLine(farthestVertex, positiveMarginalLineSeg)

        # 最远边 与 最负向边 的 公共点
        vertex2 = cls.getFootPointFromPointToLine(farthestVertex, negativeMarginalLineSeg)

        # 断言 vertex0, 1, 2, 3是沿着 逆时针 或者 顺时针 顺序的
        assert cls.arePointsAnticlockwiseOrClockwise([vertex0, vertex1, vertex2, vertex3])

        # 用这四个点vertex0, 1, 2, 3 实例化 四边形
        quadrangle = Quadrangle([vertex0, vertex1, vertex2, vertex3])

        # # 断言 quadrangle 为 矩形
        # assert quadrangle.isRectangle(threshold=1e5 * Utils.Threshold) #TODO: 恢复isRectangle()

        # 返回 quadrangle
        return quadrangle


    @classmethod
    def __getMarginalVertexOnConvexHullForOneEdgeOnConvexHull(cls, convexHull: Polygon, edge: LineSeg, directionSign: int) -> Point:
        """
        为 凸包上的一条边 获取 凸包上的一个边缘点
        Args:
            convexHull: 凸包
            edge: 凸包上的一条边
            directionSign: 方向符号 | 沿着边的起点->终点方向 或 沿着边的终点->起点方向
        Returns: 这一边缘点
        """
        # 断言 edge 位于 convexHull 上
        assert convexHull.isLineSegEqualToOneOfLineSegsInLineSegContainer(edge)

        # 断言 directionSign 的 值域
        assert directionSign in {1, -1}

        # 根据 directionSign 实例化 edgeVector
        edgeVector = Vector(edge) if directionSign == 1 else Vector(edge[::-1])

        # 根据 directionSign 选择 候选向量 的 起点
        candidateVectorStartPoint = edge.startPoint if directionSign == 1 else edge.endPoint

        # 最大点积
        maxDotProd = None

        # 候选点
        candidatePoint = None

        # 遍历 凸包 的 每一个顶点
        for vertex in convexHull.vertexes:
            # 当前 点积
            currDotProd = np.dot(edgeVector.vector, Vector([candidateVectorStartPoint, vertex]).vector)
            # 如果 maxDotProd 为 None 或 该 currDotProd 大于 maxDotProd
            if (maxDotProd is None) or (currDotProd > maxDotProd):
                # 则 更新 maxDotProd 并 替换 candidatePoint
                maxDotProd = currDotProd
                candidatePoint = vertex

        # 断言 候选点 不是 None
        assert not (candidatePoint is None)

        # 返回 候选点
        return candidatePoint


    @classmethod
    def __getFarthestVertexOnConvexHullForOneEdgeOnConvexHull(cls, convexHull: Polygon, edge: LineSeg) -> Point:  # List[List[List[float]]], List[List[float]]
        """
        对于 凸包上的一条边 获取 凸包上 与其距离最远的 顶点
        Args:
            convexHull: 凸包
            edge: 凸包上的一条边
        Returns: 这一顶点
        """
        # 断言 edge 位于 convexHull 上
        assert convexHull.isLineSegEqualToOneOfLineSegsInLineSegContainer(edge)

        # 最大面积
        maxArea = 0.0

        # 候选点
        candidatePoint = None

        # 遍历 凸包 的 每一个顶点
        for vertex in convexHull.vertexes:
            # 当前面积
            currArea = Triangle([vertex, edge.startPoint, edge.endPoint]).getAreaAbsAt2D3DSpace()
            # 如果 当前面积 大于 最大面积
            if currArea > maxArea:
                # 则 更新 最大面积 并 替换 候选点
                maxArea = currArea
                candidatePoint = vertex

        # 断言 候选点 不是 None
        assert not (candidatePoint is None)

        # 断言 候选点 不位于 edge 上
        assert not cls.isPointAtLineSeg(candidatePoint, edge)

        # 返回 候选点
        return candidatePoint


    @classmethod
    def getMinAreaBoundingRectangle(cls, points: List[List[float]]) -> Quadrangle:
        """
        获取 关于一群点的 一个最小面积包围矩形
        Args:
            points: 一群点
        Returns: 一个Quadrangle实例 | 返回前 断言 其为 矩形
        """
        # 由 这些点产生的 凸包
        convexHull = Polygon(cls.getConvexHull(points))

        # 断言 点的数量 大于等于 3
        assert len(points) >= 3

        # 最小面积
        minArea = None

        # 候选矩形
        candidateRect = None

        # 遍历 凸包的 每一条边
        for edge in convexHull.edges:
            # 最远点
            farthestVertex = cls.__getFarthestVertexOnConvexHullForOneEdgeOnConvexHull(convexHull, edge)

            # 最正向点
            positiveMarginalVertex = cls.__getMarginalVertexOnConvexHullForOneEdgeOnConvexHull(convexHull, edge, 1)

            # 最反向点
            negativeMarginalVertex = cls.__getMarginalVertexOnConvexHullForOneEdgeOnConvexHull(convexHull, edge, -1)

            # 三点 一边 构成 矩形
            currRect = cls.__getRectangleFromLineSegAndThreePoints(edge, farthestVertex, positiveMarginalVertex, negativeMarginalVertex)

            # 此矩形的面积
            currArea = currRect.area

            # 如果 minArea 为 None 或 该Quadrangle实例 的 面积 小于 minArea
            if (minArea is None) or (currArea < minArea):
                # 则 更新最小面积 并 替换 候选矩形
                minArea = currArea
                candidateRect = currRect

        # 断言 候选矩形 不是 None
        assert not (candidateRect is None)

        # # 断言 候选矩形 为 矩形
        # # assert candidateRect.isRectangle(threshold=1e5 * Utils.Threshold)
        # assert candidateRect.isRectangle() #TODO: 恢复isRectangle()

        # 返回 候选矩形
        return candidateRect


    @staticmethod
    def commonScale2D(points: List[List[float]], scalingCenter: List[float], factors: List[float]) -> None:
        """
        将 points 中的点 以 scalingCenter 为中心 以 factors为缩放因子 缩放
        Args:
            points: 点容器
            scalingCenter: 缩放中心
            factors: 缩放因子
        Returns: None
        """
        # 判空
        if len(points) == 0:
            return

        # 实例化 PointContainer
        pointContainer = PointContainer(points)

        # 平移向量
        positiveTranslationVec = [0.0 - scalingCenter[0], 0.0 - scalingCenter[1]]
        negativeTranslationVec = [scalingCenter[0] - 0.0, scalingCenter[1] - 0.0]

        # 平移
        pointContainer.translate2D(positiveTranslationVec)

        # 缩放
        pointContainer.scale2D(factors)

        # 平移回去
        pointContainer.translate2D(negativeTranslationVec)

        # 赋值
        points[:] = pointContainer.vertexes


    @staticmethod
    def commonRotate2D(points: List[List[float]], rotationCenter: List[float], rad: float) -> None:
        """
        将 points 中的点 围绕 rotationCenter 旋转 rad(弧度, 正-逆时针, 负-顺时针)
        Args:
            points: 点容器
            rotationCenter: 旋转中心
            rad: 旋转弧度
        Returns: None
        """
        # 判空
        if len(points) == 0:
            return

        # 实例化 PointContainer
        pointContainer = PointContainer(points)

        # 平移向量
        positiveTranslationVec = [0.0 - rotationCenter[0], 0.0 - rotationCenter[1]]
        negativeTranslationVec = [rotationCenter[0] - 0.0, rotationCenter[1] - 0.0]

        # 平移
        pointContainer.translate2D(positiveTranslationVec)

        # 旋转
        pointContainer.rotate2D(rad)

        # 平移回去
        pointContainer.translate2D(negativeTranslationVec)

        # 赋值
        points[:] = pointContainer.vertexes


    @classmethod
    def getRectShrunkFromMABR(cls, lineSegContainer: LineSegContainer) -> Quadrangle:
        """
        对于 一个线段容器 获取 一个收缩自MABR(最小面积外接矩形)的 矩形 | 该矩形的所有顶点 恰好 全都越过 线段容器中的 线段(或在其上)
        Args:
            lineSegContainer: 线段容器
        Returns: 一个Quadrangle实例 | 返回前 断言 其为 矩形
        """
        # #### plan A ###############################################################
        # 迭代缩小mabr
        # 获取 最小外接矩形
        mabr = cls.getMinAreaBoundingRectangle(lineSegContainer.vertexes)

        # 最大迭代次数
        maxIterationNum = 15

        # 缩放因子
        scalingFactor = 0.9

        # 迭代缩小
        iterationNum = 0
        while (mabr.area > 0.7 * lineSegContainer.area) and (iterationNum < maxIterationNum):
            vertexes = mabr.vertexes
            cls.commonScale2D(vertexes, mabr.getCentroid(), [scalingFactor, scalingFactor])
            mabr = Quadrangle(vertexes)
            # assert mabr.isRectangle()

        # 平移
        mabr.translate2D(np.array(lineSegContainer.getCentroid()) - np.array(mabr.getCentroid()))

        # 返回
        return mabr
        # ###########################################################################


        # #### plan B ###############################################################
        # # 目前 此模块还不成熟 用mabr作为临时结果
        # # 直接返回最小外接矩形
        # return cls.getMinAreaBoundingRectangle(lineSegContainer.vertexes)
        # ###########################################################################


        # #### plan C ###############################################################
        # # 沿着对角线缩小mabr至边界恰好完全越过原图形内部
        # # 获取 mabr
        # mabr = cls.getMinAreaBoundingRectangle(lineSegContainer.vertexes)
        #
        # # mabr中心
        # mabrCentroid = mabr.getCentroid()
        #
        # # 按照顺序的 4个 半对角线
        # semiDiagonalLineSegs = []
        # for vertex in mabr.vertexes:
        #     semiDiagonalLineSegs.append(LineSeg([mabrCentroid, vertex]))
        #
        # # 半对角线集合数组 和 线段容器数组
        # semiDiagonalLineSegsArray = np.array(semiDiagonalLineSegs)
        # lineSegContainerArray = np.array(lineSegContainer.edges)
        #
        # # 线段是否具有公共点的矩阵 #TODO: 这里的isLoose默认为True，真地合理吗？
        # intersectingMat = LineSegTensorIntersectWithLineSegTensorInspector(semiDiagonalLineSegsArray, lineSegContainerArray).calcIntersectingMatForTwoLineSegTensors()
        #
        # # 存在公共点的半对角线索引 和 存在公共点的线段索引
        # rows, cols = np.where(intersectingMat == True)
        #
        # # 容器 存储 半对角线 与 线段 的 公共点
        # intersectionPointLists = [[], [], [], []]  # 不能写成[[]] * 4，因为如果这样写，其中的子列表的地址是相同的，下面的append就会为所有的子列表添加东西
        #
        # # 遍历 索引 计算交点
        # for row, col in zip(rows, cols):
        #     intersectionPoint = AssemblyAlgos2D.calcIntersectionPointOfTwoLineSegs(semiDiagonalLineSegs[row], lineSegContainer.edges[col])  # lineSegContainer[col]
        #     # try:
        #     #     assert len(intersectionPoint) != 0
        #     # except:
        #     #     intersectionPoint = AssemblyAlgos2D.calcIntersectionPointOfTwoLineSegs(semiDiagonalLineSegs[row], lineSegContainer.edges[col])  # lineSegContainer[col]
        #     if len(intersectionPoint) == 0:
        #         continue
        #     intersectionPointLists[row].append(intersectionPoint)
        #
        # # # 添加 默认交点 | 如果 没有交点 #TODO: 此处应该添加警告
        # # for row in range(len(intersectionPointLists)):
        # #     if len(intersectionPointLists[row]) == 0:
        # #         intersectionPointLists[row].append(semiDiagonalLineSegs[row].endPoint)
        #
        # # # 各个半对角线上的交点 各自 去重复点
        # # for row, intersectionPointList in enumerate(intersectionPointLists):
        # #     intersectionPointLists[row] = Utils.removeRepeatedPoints(intersectionPointList)
        #
        # # 容器 存储 特征点 | 半对角线上 距离 mabr中心 最远的 点
        # eigenIntersectionPoints = []
        # for intersectionPointList in intersectionPointLists:  # TODO: 此处效率是可以优化的 | 1.可以不排序 2.可以不计算到距离
        #     # try:
        #     eigenIntersectionPoints.append(sorted(intersectionPointList, key=lambda interPnt: Utils.calcDistanceBetweenTwoPoints(interPnt, mabrCentroid), reverse=True)[0])
        #     # except:
        #     #     from matplotlib import pyplot as plt
        #     #     for edge in lineSegContainer.edges:
        #     #         xs = [edge.startPoint.x, edge.endPoint.x]
        #     #         ys = [edge.startPoint.y, edge.endPoint.y]
        #     #         plt.plot(xs, ys, 'b')
        #     #     for edge in mabr.edges:
        #     #         xs = [edge.startPoint.x, edge.endPoint.x]
        #     #         ys = [edge.startPoint.y, edge.endPoint.y]
        #     #         plt.plot(xs, ys, 'r--')
        #     #     plt.axis("equal")
        #     #     plt.show()
        #     #     print("lineSegContainer: {}".format(lineSegContainer))
        #     #     eigenIntersectionPoints.append(sorted(intersectionPointList, key=lambda interPnt: Utils.calcDistanceBetweenTwoPoints(interPnt, mabrCentroid), reverse=True)[0])
        #
        # # 收缩矩形 的 起始顶点 及其对应的 半对角线索引 #TODO: 此处效率是可以优化的 | 1.可以不排序 2.可以不计算到距离
        # startVertexShrunkRect = sorted(eigenIntersectionPoints, key=lambda eigenInterPnt: Utils.calcDistanceBetweenTwoPoints(eigenInterPnt, mabrCentroid))[0]
        # idxOfSemiDiagonalLineSegWithStartVertexShrunkRect = eigenIntersectionPoints.index(startVertexShrunkRect)
        #
        # # 容器 存储 与收缩矩形起始顶点 相邻的 收缩矩形 的 顶点
        # adjacentVertexesForStartVertex = []
        # # 遍历 mabr 的 前两个edge
        # for edge in mabr.edges[:2]:
        #     # 从 startVertexShrunkRect 至 edge 垂线段
        #     verticalLineSeg = cls.getVerticalLineFromPointToLineSeg(startVertexShrunkRect, edge)
        #     assert not (verticalLineSeg is None)  # 如果此行报错，则将"cls.getVerticalLineFromPointToLineSeg()"修改为"cls.getVerticalLineFromPointToLine"
        #     # 遍历 每一个 半对角线
        #     for idx, semiDiagonalLineSeg in enumerate(semiDiagonalLineSegs):
        #         # 如果 找到 与startVertexShrunkRect所在的半对角线 相邻的 半对角线 所在直线
        #         if abs(idxOfSemiDiagonalLineSegWithStartVertexShrunkRect - idx) & 1:
        #             # 将 此相邻顶点 添加至 容器中
        #             adjacentVertexesForStartVertex.append(
        #                 AssemblyAlgos2D.calcIntersectionPointOfTwoLines(verticalLineSeg, semiDiagonalLineSeg))
        #             # 跳出循环 (下一个符合条件的半对角线与此对角线是在同一直线上的)
        #             break
        #
        # # 对 adjacentVertexesForStartVertex 去重复点
        # adjacentVertexesForStartVertex = cls.removeRepeatedPoints(adjacentVertexesForStartVertex)
        # assert len(adjacentVertexesForStartVertex) == 2
        #
        # # 与收缩矩形起始顶点 相对的 收缩矩形 的 顶点
        # oppositeVertexForStartVertexContainer = [deepcopy(startVertexShrunkRect)]
        # cls.commonRotate2D(oppositeVertexForStartVertexContainer, mabrCentroid, np.pi)
        # oppositeVertexForStartVertex = oppositeVertexForStartVertexContainer[0]
        #
        # # 容器 存储 顶点
        # vertexesShrunkRect = [
        #     startVertexShrunkRect,
        #     adjacentVertexesForStartVertex[0],
        #     oppositeVertexForStartVertex,
        #     adjacentVertexesForStartVertex[1]
        # ]
        #
        # # 断言 vertexesShrunkRect点的顺序 按照逆时针 或者 顺时针
        # assert cls.arePointsAnticlockwiseOrClockwise(vertexesShrunkRect)
        #
        # # 实例化 Quadrangle
        # shrunkRect = Quadrangle(vertexesShrunkRect)
        #
        # # 断言 shrunkRect 与 mabr 中心重合
        # assert shrunkRect.getCentroid().isEqualToAnother(mabrCentroid)
        #
        # # 平移 | 使得 shrunkRect 与 lineSegContainer 中心重合
        # shrunkRect.translate2D(Vector([shrunkRect.getCentroid(), lineSegContainer.getCentroid()]).vector)
        #
        # # 断言 shrunkRect 为 矩形
        # assert shrunkRect.isRectangle(threshold=1e5 * Utils.Threshold) #TODO: 恢复isRectangle()
        #
        # # 返回
        # return shrunkRect
        # ###########################################################################


    @staticmethod
    def isExternalCornerATTwoVecsFromThreePoints(clockwiseSign: int, subTopPoint: Point, topPoint: Point, candidatePoint: Point, isLoose: bool=True, threshold: float=Utils.Threshold) -> bool:
        """
        根据 三个点组成的两个向量 判定 栈顶元素(topPoint) 是否为阳角点
        Args:
            clockwiseSign: 逆时针 顺时针 符号
            subTopPoint: 次栈顶元素
            topPoint: 栈顶元素
            candidatePoint: (当前轮即将入栈的点)候选点
        Returns: 栈顶元素 是否为阳角点
        """
        # 断言 点序符号 为 1 或 -1
        assert clockwiseSign in {1, -1}

        # 由 subTopPoint 和 topPoint 构成的 向量
        vec0 = Vector([subTopPoint, topPoint])

        # 由 subTopPoint 和 candidatePoint 构成的 向量
        vec1 = Vector([subTopPoint, candidatePoint])

        # 计算 叉积 | vec0, vec1
        crossProd = vec0.cross(vec1)

        # 判断 栈顶元素 是否为阳角点
        if isLoose:
            # 逆时针
            if clockwiseSign > 0.0:
                return crossProd > 0.0 + threshold #比0.0稍微大一点点，就是有一点点突出，看着像是平的，也不算阳角点
            # 顺时针
            else:
                return crossProd < 0.0 - threshold #比0.0稍微小一点点，就是有一点点突出，看着像是平的，也不算阳角点
        else:
            # 逆时针
            if clockwiseSign > 0.0:
                return crossProd > 0.0
            # 顺时针
            else:
                return crossProd < 0.0 #-0.0 #


    @classmethod
    def getExternalAndInternalCornerPointsOfPolygon(cls, polygon: Polygon, isLoose: bool=True, threshold: float=Utils.Threshold) -> Tuple[List[List[float]], List[List[float]]]:
        """
        获取 一个多边形上的阳角点 | 这里的阳角点是指 当从多边形外部看去时 这些点为阳角点
        要求 这一多边形上的顶点给出需要按照 逆时针 或 顺时针 顺序 给出
        这里是要判断阴角，而不是判断阴角位置，不能直接调用凸包算法 | 因为凸包算法直接将阴角位置也扣除了
        Args:
            polygon: 多边形
        Returns: 元组 (容器 存储 多边形上的阳角点, 容器 存储 多边形上的阴角点)
        """
        # TODO: 应该添加判定自相交的前置逻辑

        # 顶点 #添加辅助点 #首加最后一点 #尾加第零点
        vertexes = polygon.vertexes #[polygon.vertexes[-1]] + polygon.vertexes + [polygon.vertexes[0]]

        # 断言 多边形顶点 按照 逆时针 或 顺时针 顺序 排列
        assert cls.arePointsAnticlockwiseOrClockwise(vertexes)

        # 逆时针顺时针符号 #逆时针: 1 #顺时针: -1
        clockwiseSign = 0
        if cls.arePointsAntiClockwise(vertexes):
            clockwiseSign = 1
        elif cls.arePointsClockwise(vertexes):
            clockwiseSign = -1
        assert clockwiseSign != 0

        # 容器 存储 阳角点 #单调栈
        externalCornerPoints = []

        # 容器 存储 阴角点
        internalCornerPoints = []

        # 遍历 多边形上的每一个顶点
        for idx, vertex in enumerate(vertexes):
            # # 当 单调栈中的元素个数大于等于2 并且 次栈顶元素、栈顶元素、当前点连线成角不是阳角
            # while (len(externalCornerPoints) >= 2) and (not cls.isExternalCornerATTwoVecsFromThreePoints(clockwiseSign, externalCornerPoints[-2], externalCornerPoints[-1], vertex)):
            #     # 将 即将要弹出的 候选点 存入 internalCornerPoints中
            #     internalCornerPoints.append(externalCornerPoints[-1])
            #     # 弹出
            #     externalCornerPoints.pop()
            # # 将当前候选位置 压栈
            # externalCornerPoints.append(vertex)

            if cls.isExternalCornerATTwoVecsFromThreePoints(clockwiseSign, vertexes[idx - 1], vertex, vertexes[(idx + 1) % len(vertexes)], isLoose=isLoose, threshold=threshold):
                externalCornerPoints.append(vertex)
            else:
                internalCornerPoints.append(vertex)

        # # 去除 一开始添加的 辅助点
        # externalCornerPoints = externalCornerPoints[1:-1]

        # 返回
        return externalCornerPoints, internalCornerPoints




if __name__ == '__main__':
    # # lineSegA = LineSeg([[-1, -1], [-2, -3]])
    # # lineSegB = LineSeg([[4, 10], [5, 13]])
    # lineSegA = LineSeg([[1, 3], [5, 11]])
    # lineSegB = LineSeg([[1, 1], [4, 10]])
    # print(AssemblyAlgos2D.calcIntersectionPointOfTwoLineSegs(lineSegA, lineSegB))

    # # ### convexHull ##########################################
    # # points = [
    # #     [0.0, 0.0],
    # #     [0.5, 0.5],
    # #     [1.0, 1.0],
    # #     [0.0, 1.0],
    # # ]
    #
    # points = [
    #     [1, 0],
    #
    #     [1.1, 0],
    #     [1.2, -0.15],
    #     [1.3, 0],
    #
    #     [2, 0],
    #     [1, 1],
    #     [2, 2],
    #     [1, 1.5],
    #     [0, 2],
    #     [0.5, 1],
    #     [-1, -1],
    # ]
    #
    # # u = (1 + np.sqrt(3)) / 2.0
    # # t = (5 - np.sqrt(3)) / 2.0
    # # points = LineSegContainer([
    # #     [[0, 0], [np.sqrt(3), 1]],
    # #     [[np.sqrt(3), 1], [u, t]],
    # #     [[u, t], [0, 2]],
    # #     [[0, 2], [-1, np.sqrt(3)]],
    # #     [[-1, np.sqrt(3)], [0, 0]]
    # # ]).vertexes
    #
    # # print(AssemblyAlgos2D.getConvexHull(points, allowPointOnLineSegInwardly=False))
    #
    # convexHull = Polygon(AssemblyAlgos2D.getConvexHull(points, allowPointOnLineSegInwardly=False))
    #
    # from matplotlib import pyplot as plt
    #
    # fig = plt.figure()
    #
    # # for point in points:
    # Xs = [point[0] for point in points] # + [points[0]]
    # Ys = [point[1] for point in points] # + [points[0]]
    # plt.scatter(Xs, Ys, color='b', s=100)
    #
    # for idx in range(len(points)):
    #     xs = [points[idx][0], points[(idx + 1 )% len(points)][0]]
    #     ys = [points[idx][1], points[(idx + 1 )% len(points)][1]]
    #     plt.plot(xs, ys, color='b')
    #
    # # for point in convexHull.vertexes:
    # Xs = [point[0] for point in convexHull.vertexes] # + [points[0]]
    # Ys = [point[1] for point in convexHull.vertexes] # + [points[0]]
    # plt.scatter(Xs, Ys, color="orange")
    #
    # for idx in range(len(convexHull.vertexes)):
    #     xs = [convexHull.vertexes[idx][0], convexHull.vertexes[(idx + 1 )% len(convexHull.vertexes)][0]]
    #     ys = [convexHull.vertexes[idx][1], convexHull.vertexes[(idx + 1 )% len(convexHull.vertexes)][1]]
    #     plt.plot(xs, ys, color="orange")
    #
    # plt.axis("equal")
    # plt.show()
    # # #########################################################

    # # ### MABR ##############################################
    # # points = [
    # #     [0, 0],
    # #     [1, 0],
    # #     [1, 1],
    # #     [0, 1],
    # # ]
    # # points = [
    # #     [0, 0],
    # #     [1, 1],
    # #     [1, 0],
    # #     [0, 1],
    # # ]
    #
    # # points = [
    # #     [0, 1],
    # #     [1, -1],
    # #     [0, 0],
    # #     [-1, -1],
    # # ]
    # # points = [
    # #     [0, 1],
    # #     [0, 0],
    # #     [1, -1],
    # #     [-1, -1],
    # # ]
    #
    # points = [
    #     [0, 1],
    #     [0.5, 2],
    #     [3.5, -1],
    #     [3, -2],
    # ]
    # # points = [
    # #     [0, 1],
    # #     [3.5, -1],
    # #     [0.5, 2],
    # #     [3, -2],
    # # ]
    #
    # # points = [
    # #     [0, 0],
    # #     [2, 0],
    # #     [0.5, 0.5],
    # #     [-1, 2],
    # # ]
    #
    # # points = [
    # #     [0, 0],
    # #     [2, 0],
    # #     [0.5, 0.5],
    # #     [0, 2],
    # # ]
    #
    # # points = [
    # #     [0, 0],
    # #     [0, 2],
    # #     [1, 2],
    # #     [3, 1],
    # #     [2, -1],
    # # ]
    #
    # # points = LineSegContainer([
    # #     [[93161.07000000005, 85575.15], [97007.64999999998, 85575.14999999998]],
    # #      [[97007.64999999998, 85575.14999999998], [97007.64999999998, 85725.15]],
    # #      [[97007.64999999998, 85725.15], [97007.64999999997, 90925.15]],
    # #      [[97007.64999999997, 90925.15], [95084.36, 91075.15]],
    # #      [[95084.36, 91075.15], [93161.07000000007, 91075.15]],
    # #      [[93161.07000000007, 91075.15], [93161.07000000005, 85725.15000000005]],
    # #      [[93161.07000000005, 85725.15000000005], [93161.07000000005, 85575.15]]
    # # ]).vertexes
    #
    # # u = (1 + np.sqrt(3)) / 2.0
    # # t = (5 - np.sqrt(3)) / 2.0
    # # points = LineSegContainer([
    # #     [[0, 0], [np.sqrt(3), 1]],
    # #     [[np.sqrt(3), 1], [u, t]],
    # #     [[u, t], [0, 2]],
    # #     [[0, 2], [-1, np.sqrt(3)]],
    # #     [[-1, np.sqrt(3)], [0, 0]]
    # # ]).vertexes
    #
    # mabr = AssemblyAlgos2D.getMinAreaBoundingRectangle(points)
    #
    # from matplotlib import pyplot as plt
    #
    # fig = plt.figure()
    #
    # convexHull = Polygon(AssemblyAlgos2D.getConvexHull(points))
    # # for point in points:
    # # for point in convexHull.vertexes:
    # Xs = [point[0] for point in convexHull.vertexes]  # points + [points[0]]
    # Ys = [point[1] for point in convexHull.vertexes]  # points + [points[0]]
    # plt.scatter(Xs, Ys, color="orange")
    # Xs = [point[0] for point in points + [points[0]]] #points + [points[0]]
    # Ys = [point[1] for point in points + [points[0]]] #points + [points[0]]
    # plt.plot(Xs, Ys, color='b')
    #
    # # for point in mabr:
    # Xs = [point[0] for point in mabr.vertexes + [mabr.vertexes[0]]] #mabr.vertexes + [mabr.vertexes[0]]
    # Ys = [point[1] for point in mabr.vertexes + [mabr.vertexes[0]]] #mabr.vertexes + [mabr.vertexes[0]]
    # plt.plot(Xs, Ys, 'r--')
    #
    # plt.axis("equal")
    # plt.show()
    # #########################################################

    # # ### getRectShrunkFromMABR #############################
    # # lineSegContainer = LineSegContainer([
    # #     [[0, 0], [0, 2]],
    # #     [[0, 2], [1, 2]],
    # #     [[1, 2], [3, 1]],
    # #     [[3, 1], [2, -1]],
    # #     [[2, -1], [0, 0]]
    # # ])
    #
    # # u = (1 + np.sqrt(3)) / 2.0
    # # t = (5 - np.sqrt(3)) / 2.0
    # # lineSegContainer = LineSegContainer([
    # #     [[0, 0], [np.sqrt(3), 1]],
    # #     [[np.sqrt(3), 1], [u, t]],
    # #     [[u, t], [0, 2]],
    # #     [[0, 2], [-1, np.sqrt(3)]],
    # #     [[-1, np.sqrt(3)], [0, 0]]
    # # ])
    #
    # lineSegContainer = LineSegContainer([
    #     [[93161.07000000005, 85575.15], [97007.64999999998, 85575.14999999998]],
    #     [[97007.64999999998, 85575.14999999998], [97007.64999999998, 85725.15]],
    #     [[97007.64999999998, 85725.15], [97007.64999999997, 90925.15]],
    #     [[97007.64999999997, 90925.15], [95084.36, 91075.15]],
    #     [[95084.36, 91075.15], [93161.07000000007, 91075.15]],
    #     [[93161.07000000007, 91075.15], [93161.07000000005, 85725.15000000005]],
    #     [[93161.07000000005, 85725.15000000005], [93161.07000000005, 85575.15]]
    # ])
    #
    # shrunkRect = AssemblyAlgos2D.getRectShrunkFromMABR(lineSegContainer)
    #
    # from matplotlib import pyplot as plt
    #
    # fig = plt.figure()
    #
    # for edge in lineSegContainer.edges:
    #     Xs = [edge.startPoint.x, edge.endPoint.x]
    #     Ys = [edge.startPoint.y, edge.endPoint.y]
    #     plt.plot(Xs, Ys, color='b')
    #
    # for edge in shrunkRect.edges:
    #     Xs = [edge.startPoint.x, edge.endPoint.x]
    #     Ys = [edge.startPoint.y, edge.endPoint.y]
    #     plt.plot(Xs, Ys, color='g')
    #
    # plt.axis("equal")
    # plt.show()
    # #########################################################

    # # ### getExternalAndInternalCornerPointsOfPolygon ####################
    # # 逆时针 点序
    # # polygon = Polygon([
    # #     [0, 0],
    # #     [2, 0],
    # #     [1, 1],
    # #     [2, 2],
    # #     [0, 2],
    # # ])
    #
    # # polygon = Polygon([
    # #     [-1, -1],
    # #     [1, 0],
    # #     [2, 0],
    # #     [1, 1],
    # #     [2, 2],
    # #     [1, 1.5],
    # #     [0, 2],
    # #     [0.5, 1],
    # # ])
    #
    # # polygon = Polygon([
    # #     [1, 0],
    # #     [2, 0],
    # #     [1, 1],
    # #     [2, 2],
    # #     [1, 1.5],
    # #     [0, 2],
    # #     [0.5, 1],
    # #     [-1, -1],
    # # ])
    #
    # # polygon = Polygon([
    # #     [1, 0],
    # #
    # #     [1.1, 0],
    # #     [1.2, -0.15],
    # #     [1.3, 0],
    # #
    # #     [2, 0],
    # #     [1, 1],
    # #     [2, 2],
    # #     [1, 1.5],
    # #     [0, 2],
    # #     [0.5, 1],
    # #     [-1, -1],
    # # ])
    #
    # # 顺时针 点序
    # # polygon = Polygon([
    # #     [0, 0],
    # #     [0, 2],
    # #     [2, 2],
    # #     [1, 1],
    # #     [2, 0],
    # # ])
    #
    # polygon = Polygon(list(reversed([
    #     [-1, -1],
    #     [1, 0],
    #     [2, 0],
    #     [1, 1],
    #     [2, 2],
    #     [1, 1.5],
    #     [0, 2],
    #     [0.5, 1],
    # ])))
    #
    # # 获取 阳角点 阴角点
    # externalCornerPoints, internalCornerPoints = AssemblyAlgos2D.getExternalAndInternalCornerPointsOfPolygon(polygon)
    #
    # # 绘制多边形
    # for edge in polygon.edges:
    #     xs = [edge.startPoint.x, edge.endPoint.x]
    #     ys = [edge.startPoint.y, edge.endPoint.y]
    #     plt.plot(xs, ys, color='k')
    #
    # # 绘制 阳角点
    # for externalCornerPoint in externalCornerPoints:
    #     xs = [externalCornerPoint[0]]
    #     ys = [externalCornerPoint[1]]
    #     plt.scatter(xs, ys, color='r')
    #
    # # 绘制 阴角点
    # for internalCornerPoint in internalCornerPoints:
    #     xs = [internalCornerPoint[0]]
    #     ys = [internalCornerPoint[1]]
    #     plt.scatter(xs, ys, color='b')
    #
    # # 设置等轴
    # plt.axis("equal")
    #
    # # 展示
    # plt.show()
    # # #########################################################


    # ### calcIntersectionPointOfTwoLines ######################
    l1 = [[38484663057.0, 2551694988.0], [38484691297.0, 2551689509.0]]

    l2 = [[38484657115.0, 2551692424.0], [38484663057.0, 2551694988.0]]

    l1, l2 = LineSeg(l1), LineSeg(l2)
    res = AssemblyAlgos2D.calcIntersectionPointOfTwoLines(l1, l2)
    print(res)
    # ##########################################################
