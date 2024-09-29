# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   generalAlgos
Description: 
Author:      liu
Date:        1/21/22
---------------------------------------
"""

import numpy as np

from typing import List, Tuple

from Welt.Utils import Utils
from Welt.Structs.StructsSingleParticle.Point.Point import Point
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSeg import LineSeg
from Welt.Structs.StructsDoubleParticles.Vector import Vector
from Welt.Structs.StructsMultiParticles.Surface.Quadrangle.Quadrangle import Quadrangle
from Welt.Structs.StructsMultiParticles.Surface.Triangle.Triangle import Triangle


class GeneralAlgos(object):
    """
    本类中的方法全都是 2D空间, 3D空间中 通用的
    参数列表中的参数类型，尽量会写得抽象
    """
    @staticmethod
    def dot(vecA, vecB):
        """
        计算两个向量的点积
        :param vecA: [x, y] or [[startX, strartY], [endX, endY]]
        :param vecB: [x, y] or [[startX, strartY], [endX, endY]]
        :return: dot product of vecA and vecB
        """
        return Utils.dot(vecA, vecB)


    @staticmethod
    def cross(vecA, vecB):
        """
        计算两个向量的叉积
        :param vecA: [x, y] or [[startX, strartY], [endX, endY]]
        :param vecB: [x, y] or [[startX, strartY], [endX, endY]]
        :return: cross product of vecA and vecB
        """
        return Utils.cross(vecA, vecB)


    @staticmethod
    def calcBoundaryOfPoints(points: List[Point]) -> Tuple[float, float, float, float]:
        """
        计算 points中所有点的 边界
        Args:
            points: 容器 存储 点
        Returns: xmin, ymin, xmax, ymax
        """
        xs, ys = [point.x for point in points], [point.y for point in points]
        return min(xs), min(ys), max(xs), max(ys)


    @staticmethod
    def calcDistanceFromPointToLine(point: List[float], lineSeg: LineSeg) -> float:
        """
        计算 点 到 线段所在直线的 距离
        Args:
            point: 点
            lineSeg: 线段 表征 直线
        Returns: 点 到 线段所在直线的 距离
        """
        return np.linalg.norm(Utils.cross([lineSeg.startPoint, lineSeg.endPoint], [lineSeg.startPoint, point])) / lineSeg.getLength()


    @staticmethod #这里只是作为上层算法的中层概念工具，还没有用到关于三角形的更加具体的算法
    def calcAreaOfTriangle(pointA: List[float], pointB: List[float], pointC: List[float]) -> float:
        return Triangle([pointA, pointB, pointC]).getAreaAbsAt2D3DSpace() #return 0.5 * np.linalg.norm(Utils.cross([pointA, pointB], [pointA, pointC]))


    @staticmethod
    def calcProjectivePointFromPointToLine(point: Point, lineSeg: LineSeg) -> Point:
        """
        计算 点 到 线段所在直线的 投影点
        Args:
            point: 点
            lineSeg: 线段
        Returns: 这一投影点
        """
        # 线段 的 起点
        startPoint = lineSeg.startPoint

        # 两个Vector
        spVector = Vector([startPoint, point])
        lineSegVector = Vector(lineSeg)

        # 投影长度 = 点积 / 线段向量模长
        projectedLength = (float)(spVector.dot(lineSegVector)) / (float)(lineSegVector.getNorm())

        # 实例化 resVector
        directionWithLength = lineSegVector.getUnitDirectVec() * projectedLength
        resVector = Vector.generateVectorFromPointAndDirection(startPoint, directionWithLength)

        # 返回 resVector 的 终点
        return resVector.endPoint


    @classmethod
    def calcProjectiveLineSegFromLineSegToLine(cls, lineSegA: LineSeg, lineSegB: LineSeg) -> LineSeg:
        """
        计算 线段 到 线段所在直线的 投影线段
        Args:
            lineSegA: 待投影线段
            lineSegB: 线段 表征 直线
        Returns: 这一投影线段
        """
        # 投影起点
        projStartPoint = cls.calcProjectivePointFromPointToLine(lineSegA.startPoint, lineSegB)

        # 投影终点
        projEndPoint = cls.calcProjectivePointFromPointToLine(lineSegA.endPoint, lineSegB)

        # 返回 投影线段
        return LineSeg([projStartPoint, projEndPoint])


    @classmethod #这里只是作为上层算法的中层概念工具，还没有用到关于三角形的更加具体的算法
    def isPointInTriangle(cls, point: List[float], pointA: List[float], pointB: List[float], pointC: List[float]):
        return Utils.isEqual(
                        cls.calcAreaOfTriangle(pointA, pointB, pointC),
                        cls.calcAreaOfTriangle(point, pointA, pointB) + \
                        cls.calcAreaOfTriangle(point, pointA, pointC) + \
                        cls.calcAreaOfTriangle(point, pointB, pointC)
                    )


    @classmethod
    def isPointInQuadrangle(cls, point: List[float], quadrangle: Quadrangle):
        return cls.isPointInTriangle(point, quadrangle.vertexCD, quadrangle.vertexDA, quadrangle.vertexAB) or \
            cls.isPointInTriangle(point, quadrangle.vertexAB, quadrangle.vertexBC, quadrangle.vertexCD)


    @classmethod
    def isLineSegInQuadrangle(cls, lineSeg: LineSeg, quadrangle: Quadrangle):
        return cls.isPointInQuadrangle(lineSeg.startPoint, quadrangle) and \
            cls.isPointInQuadrangle(lineSeg.endPoint, quadrangle)


    @classmethod
    def isQuadrangleInAnother(cls, quadrangleA: Quadrangle, quadrangleB: Quadrangle):
        return cls.isLineSegInQuadrangle(quadrangleA.edgeALineSeg, quadrangleB) and \
            cls.isLineSegInQuadrangle(quadrangleA.edgeBLineSeg, quadrangleB) and \
            cls.isLineSegInQuadrangle(quadrangleA.edgeCLineSeg, quadrangleB) and \
            cls.isLineSegInQuadrangle(quadrangleA.edgeDLineSeg, quadrangleB)


    @classmethod
    def isQuadrangleEqualToAnother(cls, quadrangleA: Quadrangle, quadrangleB: Quadrangle):
        return cls.isQuadrangleInAnother(quadrangleA, quadrangleB) and \
            cls.isQuadrangleInAnother(quadrangleB, quadrangleA)




if __name__ == '__main__':
    # print(GeneralAlgos.calDistanceFromPointToLineSeg(Point([1, 1, 1]), LineSeg([[0, 0, 0], [1000, 0, 0]])))
    # print(GeneralAlgos.calDistanceFromPointToLineSeg(Point([1, 1]), LineSeg([[0, 0], [1000, 0]])))

    # print(GeneralAlgos.calcAreaOfTriangle([0, 0, 0], [1, 1, 0], [1, 1, 1]))
    # print(GeneralAlgos.isPointInTriangle([0.5, 0.5, 0.5], [0, 0, 0], [1, 1, 0], [1, 1, 1]))
    # print(GeneralAlgos.isPointInQuadrangle([0.5, 0.5, 0.5], Quadrangle([[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]])))
    # print(GeneralAlgos.isLineSegInQuadrangle(LineSeg([[0, 0, 0], [1, 1, 1]]), Quadrangle([[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]])))
    # print(GeneralAlgos.isQuadrangleInAnother(Quadrangle([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0, 0, 0.5]]), Quadrangle([[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]])))
    # print(GeneralAlgos.isQuadrangleEqualToAnother(Quadrangle([[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]]), Quadrangle([[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]])))


    # ### calcProjectivePointFromPointToLine #########
    # point = Point([2.0, 3.0])
    point = Point([2.9, 1.8])
    lineSeg = LineSeg([[1.9, 1.0], [5.0, 3.5]])
    print(GeneralAlgos.calcProjectivePointFromPointToLine(point, lineSeg))

    point = Point([0, 0, 0])
    lineSeg = LineSeg([[0, 0, 2], [2, 0, 0]])
    print(GeneralAlgos.calcProjectivePointFromPointToLine(point, lineSeg))

    import more_itertools
    print(more_itertools.ilen(point))
    # ################################################


    # ### calcProjectiveLineSegFromLineSegToLine #########
    # point = Point([2.0, 3.0])
    lineSegA = LineSeg([[2.9, 1.8], [3.2, 1.7]])
    lineSegB = LineSeg([[1.9, 1.0], [5.0, 3.5]])
    print(GeneralAlgos.calcProjectiveLineSegFromLineSegToLine(lineSegA, lineSegB))

    lineSegA = LineSeg([[0, 0, 0], [0, -1, 1]])
    lineSegB = LineSeg([[0, 0, 2], [2, 0, 0]])
    print(GeneralAlgos.calcProjectiveLineSegFromLineSegToLine(lineSegA, lineSegB))
    # ###################################################
