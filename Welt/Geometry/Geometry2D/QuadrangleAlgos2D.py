# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   FaceAlgos2D
Description: 
Author:      liu
Date:        1/19/22
---------------------------------------
"""

from Welt.Structs.StructsMultiParticles.Surface.Quadrangle.Quadrangle import Quadrangle

from Welt.Utils import Utils
from Welt.Geometry.Geometry2D.CommonAlgos2D import CommonAlgos2D
from Welt.Geometry.Geometry2D.AssemblyAlgos2D import AssemblyAlgos2D


class QuadrangleAlgos2D(object):
    # TODO: 将这些方法全部搬运到Quadangle中的成员方法中去
    @staticmethod
    def calculateAreaOfPolygon(coordX, coordY, isAbs=True):
        """
        计算任意多边形（面）面积
        :param coordX: 所有横坐标
        :param coordY: 所有纵坐标
        :param isAbs: 是否取绝对值（否则，逆时针为正，顺时针为负）
        :return: float
        """
        return Utils.calcAreaOfPolygon(coordX, coordY, isAbs)


    # @staticmethod #TODO: 应该考虑删除，应该考虑搬运到各自的类中，调用者应该考虑改用
    # def isQuadrangleOnlyHasFourPoints(quadrangle):
    #     return len(quadrangle) == 4


    @classmethod #TODO：应该考虑搬运到各自的类中，调用者应该考虑改用
    def isQuadrangleParallelogram(cls, quadrangle):
        # # assert cls.isQuadrangleOnlyHasFourPoints(quadrangle)
        # point0, point1, point2, point3 = quadrangle #数据类型已经从PointContainer变成了LineSegContainer，所以不能这样写
        # # 两对对边对应平行
        # return \
        #     CommonAlgos2D.areTwoVecsParallel(point0, point1, point2, point3) \
        #     and \
        #     CommonAlgos2D.areTwoVecsParallel(point1, point2, point3, point0)
        return quadrangle.isParallelogram()


    @classmethod #TODO：应该考虑搬运到各自的类中，调用者应该考虑改用
    def isQuadrangleRectangle(cls, quadrangle: Quadrangle):
        # if not cls.isQuadrangleParallelogram(quadrangle):
        #     return False
        # point0, point1, point2, point3 = quadrangle
        # # 一对临边相互垂直
        # return \
        #     CommonAlgos2D.isTowVecsOrthogonal(point0, point1, point1, point2)
        return quadrangle.isRectangle()


    @staticmethod #TODO：应该考虑搬运到各自的类中，调用者应该考虑改用
    def isQuadrangleOneLinesegHorizontal(quadrangle: Quadrangle):
        # ptNums = len(quadrangle)
        # for idx, point in enumerate(quadrangle):
        #     nextPoint = quadrangle[(idx + 1) % ptNums]
        #     if CommonAlgos2D.areTwoVecsParallel(point, nextPoint, [0, 0], [1, 0]):
        #         return True
        # return False
        return quadrangle.isOneLinesegHorizontal()


    @classmethod #TODO：应该考虑搬运到各自的类中，调用者应该考虑改用
    def cvtRectNormative(cls, quadrangle: Quadrangle):
        """
        将一个某条边水平的矩形face转换为左下角为起点，逆时针排布点的face实例
        :param quadrangle:
        :return: Quadrangle | **左下角为起点，逆时针**
        """
        assert cls.isQuadrangleRectangle(quadrangle) and cls.isQuadrangleOneLinesegHorizontal(quadrangle)
        return AssemblyAlgos2D.boundCoord2Quadrangle(quadrangle.getBoundCoord())
