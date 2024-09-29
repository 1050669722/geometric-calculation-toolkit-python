# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   commonAlgos
Description: 
Author:      liu
Date:        1/19/22
---------------------------------------
"""

import numpy as np

from typing import List

from Welt.Structs.StructsSingleParticle.Point.Point import Point
from Welt.Structs.StructsDoubleParticles.Vector import Vector
from Welt.Geometry.GeneralAlgos import GeneralAlgos


class CommonAlgos2D(object):
    @staticmethod
    def dot(vecA, vecB):
        """
        计算两个向量的点积
        :param vecA: [x, y] or [[startX, strartY], [endX, endY]]
        :param vecB: [x, y] or [[startX, strartY], [endX, endY]]
        :return: dot product of vecA and vecB
        """
        return GeneralAlgos.dot(vecA, vecB)


    @staticmethod
    def cross(vecA, vecB):
        """
        计算两个向量的叉积
        :param vecA: [x, y] or [[startX, strartY], [endX, endY]]
        :param vecB: [x, y] or [[startX, strartY], [endX, endY]]
        :return: cross product of vecA and vecB
        """
        return GeneralAlgos.cross(vecA, vecB)


    @staticmethod #这是Line，不是LineSeg，所以没有放在LineSegAlgos2D中
    def isTowVecsOrthogonal(startPointVecA, endPointVecA, startPointVecB, endPointVecB): #startPointVecA, endPointVecA, startPointVecB, endPointVecB #类似于xmin, ymin, xmax, ymax #先一个点，再一个点
        """
        判断两条直线是否正交（垂直）
        :param startPointVecA:
        :param endPointVecA:
        :param startPointVecB:
        :param endPointVecB:
        :return: bool
        """
        # return \
        #     Utils.isEqual(
        #         CommonAlgos2D.dot(
        #             [startPointVecA[0] - endPointVecA[0], startPointVecA[1] - endPointVecA[1]],
        #             [startPointVecB[0] - endPointVecB[0], startPointVecB[1] - endPointVecB[1]]
        #         ),
        #         0.0
        #     )
        return Vector([startPointVecA, endPointVecA]).isOrthogonalToAnother(Vector([startPointVecB, endPointVecB]))


    @staticmethod #这是Line，不是LineSeg，所以没有放在LineSegAlgos2D中
    def areTwoVecsParallel(startPointVecA, endPointVecA, startPointVecB, endPointVecB): #startPointVecA, endPointVecA, startPointVecB, endPointVecB #类似于xmin, ymin, xmax, ymax #先一个点，再一个点
        """
        判断两条直线是否平行
        :param startPointVecA:
        :param endPointVecA:
        :param startPointVecB:
        :param endPointVecB:
        :return: bool
        """
        # return \
        #     Utils.isEqual(
        #         CommonAlgos2D.cross(
        #             [startPointVecA[0] - endPointVecA[0], startPointVecA[1] - endPointVecA[1]],
        #             [startPointVecB[0] - endPointVecB[0], startPointVecB[1] - endPointVecB[1]]
        #         ),
        #         0.0
        #     )
        return Vector([startPointVecA, endPointVecA]).isParallelToAnother(Vector([startPointVecB, endPointVecB]))


    @staticmethod
    def getRadBetweenTwoVecs(vecA: List[List[float]], vecB: List[List[float]]) -> float:
        """
        废弃：vecA相对于vecB的所夹弧度角，取值范围：(-pi, pi]
        现有：从vecA到vecB所经历的弧度角，取值范围：(-pi, pi]
        :param vecA:
        :param vecB:
        :return:
        """
        if not isinstance(vecA, Vector):
            vecA = Vector(vecA)
        if not isinstance(vecB, Vector):
            vecB = Vector(vecB)
        return vecA.getAngleToAnother(vecB)


    @staticmethod
    def scale(point: Point, scaleFactorVec):
        res = np.array(
            [[scaleFactorVec[0], 0.0, 0.0],
             [0.0, scaleFactorVec[1], 0.0],
             [0.0, 0.0, 1.0]]
        ).dot(
            np.array(
                [[point.x],
                 [point.y],
                 [1.0]]
            )
        )
        return Point(list(res))
        pass


    @staticmethod
    def rotate(point: Point, rotateRad):
        res = np.array(
            [[np.cos(rotateRad), -np.sin(rotateRad), 0.0],
             [np.sin(rotateRad), np.cos(rotateRad), 0.0],
             [0.0, 0.0, 1.0]]
        ).dot(
            np.array(
                [[point.x],
                 [point.y],
                 [1.0]]
            )
        )
        return Point(list(res))
        pass


    @staticmethod
    def translate(point: Point, translateVec):
        res = np.array(
            [[0.0, 0.0, translateVec[0]],
             [0.0, 0.0, translateVec[1]],
             [0.0, 0.0, 1.0]]
        ).dot(
            np.array(
                [[point.x],
                 [point.y],
                 [1.0]]
            )
        )
        return Point(list(res))
        pass
