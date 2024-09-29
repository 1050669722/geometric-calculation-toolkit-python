# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   Point
Description: 
Author:      liu
Date:        1/18/22
---------------------------------------
"""

import more_itertools
import numpy as np

from typing import List
from typing import Iterable as typingIterable

from Welt.Utils import Utils
from Welt.Tools.StructTools import StructTools
from Welt.Structs.StructsSingleParticle.Particle import Particle


class Point(Particle, list): #TODO: 项目是否已经定义相似数据结构？
    def __init__(self, coordValues: List[float]): #*coordValues
        super(Point, self).__init__() #这里不将coordValues填入，调用list类型的空参构造器，因为赋值操作将在下面进行
        assert hasattr(coordValues, "__getitem__")

        self._activeDim = None

        self[:] = coordValues

        # 更新数据
        # self.update()
        Point.update(self)


    def update(self):
        # 判断维度，添加具有意义的属性名，属性赋值
        if len(self[:]) == 2:
            self.x, self.y = self[:]
        elif len(self[:]) == 3:
            self.x, self.y, self.z = self[:]
        else:
            raise ValueError("[ERROR] Invalid input of \"coordValues\": {}".format(self[:]))


    def getDimensionNum(self) -> int:
        """
        获取 维度数量
        Returns: 维度数量
        """
        dimensionNum = more_itertools.ilen(self[:])
        assert dimensionNum in {2, 3}
        return dimensionNum


    @property
    def activeDim(self):
        """getter函数"""
        return self._activeDim


    @activeDim.setter
    def activeDim(self, dim):
        """setter函数"""
        self._activeDim = dim


    @StructTools.runAt3DSpace
    @StructTools.modifyActiveDim
    def collapseIntoPhasePlane(self, dim):
        assert self._activeDim is not None
        assert (len(self[:]) == 3)
        if self._activeDim == 'x':
            self[:] = self[1:]
        elif self._activeDim == 'y':
            self[:] = [self[0]] + [self[2]] #self[0:len(self):2]
        elif self._activeDim == 'z':
            self[:] = self[:2]
        else:
            raise ValueError("[ERROR] Invalid value of \"dim\": {}".format(dim))
        # self.update()
        Point.update(self)


    @StructTools.runAt2DSpace
    @StructTools.modifyActiveDim
    def expandFromPhasePlane(self, dim):
        assert self._activeDim is not None
        assert (len(self[:]) == 2)
        if self._activeDim == 'x':
            self[:] = [0] + self[:]
        elif self._activeDim == 'y':
            self[:] = [self[0]] + [0] + [self[1]]
        elif self._activeDim == 'z':
            self[:] = self[:] + [0]
        else:
            raise ValueError("[ERROR] Invalid value of \"dim\": {}".format(dim))
        # self.update()
        Point.update(self)


    def isEqualToAnother(self, point) -> bool:
        if not len(self[:]) == len(point):
            ValueError("[ERROR] Two points: {} and {} have different dimensions: {} and {}, respectively".format(self[:], point, len(self[:]), len(point)))
        elif len(self[:]) == 2:
            return Utils.isEqual(self.x, point.x) and Utils.isEqual(self.y, point.y)
        else:
            return Utils.isEqual(self.x, point.x) and Utils.isEqual(self.y, point.y) and (self.z, point.z)


    @StructTools.runAt3DSpace
    def scale(self, factors: typingIterable[float]):
        """
        相对于原点O的缩放变换
        :param vec:
        :return:
        """
        assert more_itertools.ilen(factors) == 3

        assert hasattr(factors, "__getitem__")

        scaleMat = np.array([
            [factors[0], 0.0, 0.0, 0.0],
            [0.0, factors[1], 0.0, 0.0],
            [0.0, 0.0, factors[2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        oriCoord = np.array([
            [self.x],
            [self.y],
            [self.z],
            [1.0],
        ])

        resCoord = np.dot(scaleMat, oriCoord)

        self[0] = resCoord[0][0]
        self[1] = resCoord[1][0]
        self[2] = resCoord[2][0]

        # self.update()
        Point.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundX(self, rad):
        """
        绕着x轴的旋转变换
        :param rad:
        :return:
        """
        rotationMat = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(rad), -np.sin(rad), 0.0],
            [0.0, np.sin(rad), np.cos(rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        oriCoord = np.array([
            [self.x],
            [self.y],
            [self.z],
            [1.0],
        ])

        resCoord = np.dot(rotationMat, oriCoord)

        self[0] = resCoord[0][0]
        self[1] = resCoord[1][0]
        self[2] = resCoord[2][0]

        # self.update()
        Point.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundY(self, rad):
        """
        绕着y轴的旋转变换
        :param rad:
        :return:
        """
        rotationMat = np.array([
            [np.cos(rad), 0.0, np.sin(rad), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(rad), 0.0, np.cos(rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        oriCoord = np.array([
            [self.x],
            [self.y],
            [self.z],
            [1.0],
        ])

        resCoord = np.dot(rotationMat, oriCoord)

        self[0] = resCoord[0][0]
        self[1] = resCoord[1][0]
        self[2] = resCoord[2][0]

        # self.update()
        Point.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundZ(self, rad):
        """
        绕着z轴的旋转变换
        :param rad:
        :return:
        """
        rotationMat = np.array([
            [np.cos(rad), -np.sin(rad), 0.0, 0.0],
            [np.sin(rad), np.cos(rad), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        oriCoord = np.array([
            [self.x],
            [self.y],
            [self.z],
            [1.0],
        ])

        resCoord = np.dot(rotationMat, oriCoord)

        self[0] = resCoord[0][0]
        self[1] = resCoord[1][0]
        self[2] = resCoord[2][0]

        # self.update()
        Point.update(self)


    @StructTools.runAt3DSpace
    def translate(self, vec: typingIterable[float]):
        """
        平移变换
        :param vec:
        :return:
        """
        assert more_itertools.ilen(vec) == 3

        translationMat = np.array([
            [1.0, 0.0, 0.0, vec[0]],
            [0.0, 1.0, 0.0, vec[1]],
            [0.0, 0.0, 1.0, vec[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])

        oriCoord = np.array([
            [self.x],
            [self.y],
            [self.z],
            [1.0],
        ])

        resCoord = np.dot(translationMat, oriCoord)

        self[0] = resCoord[0][0]
        self[1] = resCoord[1][0]
        self[2] = resCoord[2][0]

        # self.update()
        Point.update(self)


    @StructTools.runAt3DSpace
    def isAtPlaneComposedOfThreeOther(self, pointA, pointB, pointC):
        crossProdBC = Utils.cross([self[:], pointB], [self[:], pointC])
        mixedProd = Utils.dot([self[:], pointA], crossProdBC)
        return mixedProd == 0


    @StructTools.runAt2DSpace
    def scale2D(self, factors: typingIterable[float]) -> None:
        """
        相对于原点O的缩放变换
        Args:
            factors: 各个维度的缩放变换因子容器
        Returns: None
        """
        assert more_itertools.ilen(factors) == 2

        assert hasattr(factors, "__getitem__")

        scaleMat = np.array([
            [factors[0], 0.0, 0.0],
            [0.0, factors[1], 0.0],
            [0.0, 0.0, 1.0],
        ])

        oriCoord = np.array([
            [self.x],
            [self.y],
            [1.0],
        ])

        resCoord = np.dot(scaleMat, oriCoord)

        self[0] = resCoord[0][0]
        self[1] = resCoord[1][0]

        # self.update()
        Point.update(self)


    @StructTools.runAt2DSpace
    def rotate2D(self, rad: float) -> None:
        """
        围绕着原点O的旋转
        Args:
            rad: 旋转弧度
        Returns: None
        """
        rotationMat = np.array([
            [np.cos(rad), -np.sin(rad), 0.0],
            [np.sin(rad), np.cos(rad), 0.0],
            [0.0, 0.0, 1.0],
        ])

        oriCoord = np.array([
            [self.x],
            [self.y],
            [1.0],
        ])

        resCoord = np.dot(rotationMat, oriCoord)

        self[0] = resCoord[0][0]
        self[1] = resCoord[1][0]

        # self.update()
        Point.update(self)


    @StructTools.runAt2DSpace
    def translate2D(self, vec: typingIterable[float]) -> None:
        """
        平移变换
        Args:
            vec: 平移向量
        Returns: None
        """
        assert more_itertools.ilen(vec) == 2

        assert hasattr(vec, "__getitem__")

        translationMat = np.array([
            [1.0, 0.0, vec[0]],
            [0.0, 1.0, vec[1]],
            [0.0, 0.0, 1.0],
        ])

        oriCoord = np.array([
            [self.x],
            [self.y],
            [1.0],
        ])

        resCoord = np.dot(translationMat, oriCoord)

        self[0] = resCoord[0][0]
        self[1] = resCoord[1][0]

        # self.update()
        Point.update(self)




if __name__ == '__main__':
    # point = Point([0, 1, 2])
    # point.collapseIntoPhasePlane('x')
    # # point.collapseIntoPhasePlane('y')
    # # point.collapseIntoPhasePlane('z')

    # point = Point([1, 1, 1])
    # print(point.isAtPlaneComposedOfThreeOther([0, 0, 0], [1, 1, 0], [1, 1, 1]))

    # point = Point([2, 3])
    # point.scale2D([1.5, -1.5])
    # print(point)

    # point = Point([1, 0])
    # point.rotate2D(-3 * np.pi / 4)
    # print(point)

    point = Point([2, 3])
    point.translate2D([1.5, -1])
    print(point)

    point = Point([1, 2, 3.0])
    print(point.getDimensionNum())

    pass
