# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   LineSegContainer.py
Description: 
Author:      liu
Date:        1/22/22
---------------------------------------
"""

import sys

# from typing import List
from typing import Iterable as typingIterable
from numbers import Number

if sys.version_info < (3, 8):
    from collections import Iterable
else:
    from collections.abc import Iterable

from Welt.Utils import Utils
from Welt.Tools.StructTools import StructTools
from Welt.Structs.StructsSingleParticle.Particle import Particle
from Welt.Structs.StructsSingleParticle.Point.Point import Point
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSeg import LineSeg


class LineSegContainer(Particle, list):
    def __init__(self, contour): #: List[List[List[float]]] :List[List[float]]
        super(LineSegContainer, self).__init__()
        self._activeDim = None

        assert hasattr(contour, "__getitem__")

        # 轮廓容器
        contourList = []

        # 填充轮廓容器
        if isinstance(contour[0][0], Number): #contour: List[List[float]]
            for idx in range(len(contour)):
                contourList.append([contour[idx], contour[(idx + 1) % len(contour)]])
        elif isinstance(contour[0][0], Iterable): #contour: List[List[List[float]]]
            contourList = contour
        else:
            raise TypeError("[ERROR] The parameters \"contour\": {} may be the invalid type: {}".format(contour, type(contour)))

        for lineSegData in contourList:
            self.append(LineSeg(lineSegData))

        # 更新数据
        LineSegContainer.update(self) #成员方法update被子类重载了，所以这里显式地调用


    def update(self):
        # TODO: 调整边的排列顺序为逆时针或者顺时针
        # TODO: 但是可能仅对凸多变形有效

        # 较低一级类实例化，属性赋值
        self.edges = []
        for idx in range(len(self[:])): #for lineSegStartEnd in contourList:
            self.edges.append(self[idx]) #self.edges.append(LineSeg(lineSegStartEnd))

        # # 这里认为多边形一定是闭合的
        # assert self.edges[-1].endPoint.isEqualToAnother(self.edges[0].startPoint) \
        # or self.edges[-1].startPoint.isEqualToAnother(self.edges[0].startPoint) \
        # or self.edges[-1].endPoint.isEqualToAnother(self.edges[0].endPoint) \
        # or self.edges[-1].startPoint.isEqualToAnother(self.edges[0].endPoint)

        # 较低二级类实例化，属性赋值 ###不允许出现重复的点
        self.vertexes = []
        vertexesSet = set()
        for idx, edge in enumerate(self.edges):
            if not (tuple(edge.startPoint) in vertexesSet):
                self.vertexes.append(edge.startPoint)
                vertexesSet.add(tuple(edge.startPoint))
            if not (tuple(edge.endPoint) in vertexesSet):
                self.vertexes.append(edge.endPoint) #TODO: 将这些顶点串在一块，需要另写一个判断首尾相接的方法
                vertexesSet.add(tuple(edge.endPoint))
        # self.vertexes.append(self.edges[-1].endPoint)

        # 周长
        self.perimeter = 0.0
        for edge in self.edges:
            self.perimeter += edge.length

        # 面积 #其中的线段 同一维度 and 线段集合维度为2
        if (self.areDimensionNumsIdentical()) and \
            (self.vertexes[0].getDimensionNum() == 2):
            self.area = self.getArea()


    def areDimensionNumsIdentical(self) -> bool:
        """
        检查 其中 所有图形元素的 维度数量 是否 同一 | 在本类中不调用，仅在子类中调用
        Returns: 其中 所有图形元素的 维度数量 是否 同一
        """
        dimensionNums = set()
        for edge in self.edges:
            dimensionNums.add(edge.getDimensionNum())
        return len(dimensionNums) == 1


    def isClosed(self):
        # TODO: 验证闭合，各边首尾相接，边需要按照顺序给出，边上的点也需要按照顺序给出
        for idx, edge in enumerate(self.edges):
            if not edge.endPoint.isEqualToAnother(self.edges[(idx + 1) % len(self.edges)].startPoint):
                return False
        return True


    @property
    def activeDim(self):
        """getter函数"""
        return self._activeDim


    @activeDim.setter
    def activeDim(self, dim):
        """setter函数"""
        self._activeDim = dim


    def getCentroid(self) -> Point:
        """
        获取 形心
        Returns: 形心
        """
        return Point(Utils.getCentroid(self.vertexes))


    @StructTools.runAt3DSpace
    @StructTools.modifyActiveDim
    def collapseIntoPhasePlane(self, dim):
        assert self._activeDim is not None
        for lineSeg in self[:]:
            lineSeg.collapseIntoPhasePlane(dim)
        LineSegContainer.update(self)


    @StructTools.runAt2DSpace
    @StructTools.modifyActiveDim
    def expandFromPhasePlane(self, dim):
        assert self._activeDim is not None
        for lineSeg in self[:]:
            lineSeg.expandFromPhasePlane(dim)
        LineSegContainer.update(self)


    @StructTools.runAt2DSpace
    def getBoundCoord(self):
        coordD0s, coordD1s = [], []
        for lineSeg in self[:]:
            for point in lineSeg[:]:
                coordD0s.append(point[0])
                coordD1s.append(point[1])
        return min(coordD0s), min(coordD1s), max(coordD0s), max(coordD1s)


    def getAllCoordX(self):
        return [vertex.x for vertex in self.vertexes]


    def getAllCoordY(self):
        return [vertex.y for vertex in self.vertexes]


    @StructTools.runAt3DSpace
    def getAllCoordZ(self):
        return [vertex.z for vertex in self.vertexes]


    # def isPointAtOneOfLineSeg(self, point: List[float]) -> bool:
    #     for edge in self.edges:
    #         if AssemblyAlgos2D.


    @StructTools.runAt2DSpace
    def getArea(self, isAbs=True):
        if not self.isClosed():
            return None
        return Utils.calcAreaOfPolygon(self.getAllCoordX(), self.getAllCoordY(), isAbs)


    @StructTools.runAt3DSpace
    def scale(self, factors: typingIterable[float]) -> None:
        for idx, lineSeg in enumerate(self[:]):
            lineSeg.scale(factors)
            self[idx] = lineSeg
        LineSegContainer.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundX(self, rad: float) -> None:
        for idx, lineSeg in enumerate(self[:]):
            lineSeg.rotateRadAroundX(rad)
            self[idx] = lineSeg
        LineSegContainer.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundY(self, rad: float) -> None:
        for idx, lineSeg in enumerate(self[:]):
            lineSeg.rotateRadAroundY(rad)
            self[idx] = lineSeg
        LineSegContainer.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundZ(self, rad: float) -> None:
        for idx, lineSeg in enumerate(self[:]):
            lineSeg.rotateRadAroundZ(rad)
            self[idx] = lineSeg
        LineSegContainer.update(self)


    @StructTools.runAt3DSpace
    def translate(self, vec: typingIterable[float]) -> None:
        for idx, lineSeg in enumerate(self[:]):
            lineSeg.translate(vec)
            self[idx] = lineSeg
        LineSegContainer.update(self)


    @StructTools.runAt2DSpace
    def scale2D(self, factors: typingIterable[float]) -> None:
        for idx, lineSeg in enumerate(self[:]):
            lineSeg.scale2D(factors)
            self[idx] = lineSeg
        LineSegContainer.update(self)


    @StructTools.runAt2DSpace
    def rotate2D(self, rad: float) -> None:
        for idx, lineSeg in enumerate(self[:]):
            lineSeg.rotate2D(rad)
            self[idx] = lineSeg
        LineSegContainer.update(self)
    
    
    @StructTools.runAt2DSpace
    def translate2D(self, vec: typingIterable[float]) -> None:
        for idx, lineSeg in enumerate(self[:]):
            lineSeg.translate2D(vec)
            self[idx] = lineSeg
        LineSegContainer.update(self)


    def isPointEqualToOneOfPointsInLineSegContainer(self, point: Point) -> bool:
        """ #TODO: 应该再扩展一个Container类 #此方法与PointContainer中的isPointEqualToOneOfPointsInPointContainer方法重复
        判断 一个点是否等于 LineSegContainer中的 任意一点
        Args:
            point: 待判断的点
        Returns: 这个点是否等于 LineSegContainer中的 任意一点
        """
        for vertex in self.vertexes:
            if point.isEqualToAnother(vertex):
                return True
        return False


    def isLineSegEqualToOneOfLineSegsInLineSegContainer(self, lineSeg: LineSeg) -> bool:
        """
        判断 一个线段是否等于 LineSegContainer中的 任意一线段
        Args:
            lineSeg: 待判断的线段
        Returns: 这个点是否等于 LineSegContainer中的 任意一线段
        """
        for edge in self.edges:
            if lineSeg.isEqualToAnother(edge):
                return True
        return False




if __name__ == '__main__':
    pass
