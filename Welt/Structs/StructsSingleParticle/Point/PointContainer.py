# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   PointContainer
Description: 
Author:      liu
Date:        1/19/22
---------------------------------------
"""

from typing import List
from typing import Iterable as typingIterable

from Welt.Utils import Utils
from Welt.Tools.StructTools import StructTools
from Welt.Structs.StructsSingleParticle.Particle import Particle
from Welt.Structs.StructsSingleParticle.Point.Point import Point


class PointContainer(Particle, list): #TODO: 可以试着采用register机制
    def __init__(self, contour: List[List[float]]):
        super(PointContainer, self).__init__() #这里不将contour填入，调用list类型的空参构造器，因为赋值操作将在下面进行
        self._activeDim = None

        assert hasattr(contour, "__getitem__")

        for pointData in contour:
            self.append(Point(pointData))

        # 更新数据
        PointContainer.update(self) #成员方法update被子类重载了，所以这里显式地调用


    def update(self):
        # 较低一级类实例化，属性赋值
        self.vertexes = []
        for idx in range(len(self[:])):  # for pointXY in contour:
            self.vertexes.append(self[idx])  # self.vertexes.append(Point(pointXY))


    def areDimensionNumsIdentical(self) -> bool:
        """
        检查 其中 所有图形元素的 维度数量 是否 同一 | 在本类中不调用，仅在子类中调用
        Returns: 其中 所有图形元素的 维度数量 是否 同一
        """
        dimensionNums = set()
        for vertex in self.vertexes:
            dimensionNums.add(vertex.getDimensionNum())
        return len(dimensionNums) == 1


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
        for point in self[:]:
            point.collapseIntoPhasePlane(dim)
        PointContainer.update(self)


    @StructTools.runAt2DSpace
    @StructTools.modifyActiveDim
    def expandFromPhasePlane(self, dim):
        assert self._activeDim is not None
        for point in self[:]:
            point.expandFromPhasePlane(dim)
        PointContainer.update(self)


    @StructTools.runAt2DSpace
    def getBoundCoord(self):
        coordD0s, coordD1s = [], []
        for point in self[:]:
            coordD0s.append(point[0])
            coordD1s.append(point[1])
        return min(coordD0s), min(coordD1s), max(coordD0s), max(coordD1s)


    @StructTools.runAt3DSpace
    def scale(self, factors: typingIterable[float]) -> None:
        for idx, point in enumerate(self[:]):
            point.scale(factors)
            self[idx] = point
        PointContainer.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundX(self, rad: float) -> None:
        for idx, point in enumerate(self[:]):
            point.rotateRadAroundX(rad)
            self[idx] = point
        PointContainer.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundY(self, rad: float) -> None:
        for idx, point in enumerate(self[:]):
            point.rotateRadAroundY(rad)
            self[idx] = point
        PointContainer.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundZ(self, rad: float) -> None:
        for idx, point in enumerate(self[:]):
            point.rotateRadAroundZ(rad)
            self[idx] = point
        PointContainer.update(self)


    @StructTools.runAt3DSpace
    def translate(self, vec: typingIterable[float]) -> None:
        for idx, point in enumerate(self[:]):
            point.translate(vec)
            self[idx] = point
        PointContainer.update(self)


    @StructTools.runAt2DSpace
    def scale2D(self, factors: typingIterable[float]) -> None:
        for idx, point in enumerate(self[:]):
            point.scale2D(factors)
            self[idx] = point
        PointContainer.update(self)


    @StructTools.runAt2DSpace
    def rotate2D(self, rad: float) -> None:
        for idx, point in enumerate(self[:]):
            point.rotate2D(rad)
            self[idx] = point
        PointContainer.update(self)


    @StructTools.runAt2DSpace
    def translate2D(self, vec: typingIterable[float]) -> None:
        for idx, point in enumerate(self[:]):
            point.translate2D(vec)
            self[idx] = point
        PointContainer.update(self)


    def isPointEqualToOneOfPointsInPointContainer(self, point: Point) -> bool:
        """
        判断 一个点是否等于 PointContainer中的 任意一点
        Args:
            point: 待判断的点
        Returns: 这个点是否等于 PointContainer中的 任意一点
        """
        for vertex in self.vertexes:
            if point.isEqualToAnother(vertex):
                return True
        return False


    def getConvexHull(self) -> List[Point]:
        """
        计算 这个点集 的 凸包
        Returns: 这一点集的凸包
        """
        return [Point(xy) for xy in Utils.getConvexHull(self.vertexes)]




if __name__ == '__main__':
    # pointContainer = PointContainer(
    #     [
    #         Point([0, 0, 0]),
    #         Point([1, 1, 1]),
    #     ]
    # )
    # pointContainer.collapseIntoPhasePlane('x')
    # pointContainer.expandFromPhasePlane('x')

    import numpy as np
    from typing import List
    from matplotlib import pyplot as plt
    from Welt.Structs.StructsSingleParticle.Point.PointContainer import PointContainer

    def commonRotate2D(points: List[List[float]], rotationCenter: List[float], rad: float) -> None:
        """
        将 pointContainer 中的点 围绕 rotationCenter 旋转 rad(弧度, 正-逆时针, 负-顺时针)
        Args:
            points: 点容器
            rad: 旋转弧度
            rotationCenter: 旋转中心
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

    # 点集
    points = [
        [1, 1],
        [2, 2],
        [2.3, 1],
    ]

    # 绘图
    fig = plt.figure()

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    plt.scatter(xs, ys, color='b')

    # 旋转 (本地修改)
    commonRotate2D(points, [1.5, 1.5], np.pi / 2)
    # commonRotate2D(points, [0, 2], np.pi / 2)

    # 绘图
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    plt.scatter(xs, ys, color='r')

    plt.axis("equal")
    plt.show()

    pass
