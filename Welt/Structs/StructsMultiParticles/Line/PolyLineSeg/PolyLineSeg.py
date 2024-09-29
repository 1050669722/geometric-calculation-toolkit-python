# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   PolyLineSeg
Description: 
Author:      liu
Date:        1/22/22
---------------------------------------
"""

from typing import List
from typing import Iterable as typingIterable

from Welt.Utils import Utils
from Welt.Tools.StructTools import StructTools
from Welt.Structs.StructsSingleParticle.Point.Point import Point
from Welt.Structs.StructsSingleParticle.Point.PointContainer import PointContainer


class PolyLineSeg(PointContainer): #TODO: 项目是否已经定义相似数据结构？
    def __init__(self, contour: List[List[float]]):
        super(PolyLineSeg, self).__init__(contour) #这里不将contour填入，调用list类型的空参构造器，因为赋值操作将在下面进行

        # 断言 其中的 点 的 维度数量 相同
        assert PointContainer.areDimensionNumsIdentical(self)

        # 更新数据
        self.update()


    def update(self):
        # 调用 父类的update()方法
        PointContainer.update(self)

        pass


    def getDimensionNum(self) -> int:
        """
        获取 维度数量
        Returns: 维度数量
        """
        return Point.getDimensionNum(self.vertexes[0])


    @StructTools.runAt2DSpace
    def getMinMaxCoord2D(self):
        """
        获取2D-LineSeg最小最大点坐标
        :return: xmin, ymin, xmax, ymax
        """
        return min(vertex.x for vertex in self.vertexes), min(vertex.y for vertex in self.vertexes), \
               max(vertex.x for vertex in self.vertexes), max(vertex.y for vertex in self.vertexes)


    @StructTools.runAt3DSpace
    def getMinMaxCoord3D(self):
        """
        获取3D-LineSeg最小最大点坐标
        :return: xmin, ymin, zmin, xmax, ymax, zmax
        """
        return min(vertex.x for vertex in self.vertexes), min(vertex.y for vertex in self.vertexes), min(vertex.z for vertex in self.vertexes), \
               max(vertex.x for vertex in self.vertexes), max(vertex.y for vertex in self.vertexes), max(vertex.z for vertex in self.vertexes)


    def getLength(self):
        length = 0.0
        for idx in range(len(self.vertexes) - 1):
            length += Utils.calcDistanceBetweenTwoPoints(self.vertexes[idx], self.vertexes[idx + 1])
        return length


    @StructTools.runAt3DSpace
    def scale(self, factors: typingIterable[float]) -> None:
        PointContainer.scale(self, factors)
        self.update()


    @StructTools.runAt3DSpace
    def rotateRadAroundX(self, rad: float) -> None:
        PointContainer.rotateRadAroundX(self, rad)
        self.update()


    @StructTools.runAt3DSpace
    def rotateRadAroundY(self, rad: float) -> None:
        PointContainer.rotateRadAroundY(self, rad)
        self.update()


    @StructTools.runAt3DSpace
    def rotateRadAroundZ(self, rad: float) -> None:
        PointContainer.rotateRadAroundZ(self, rad)
        self.update()


    @StructTools.runAt3DSpace
    def translate(self, vec: typingIterable[float]) -> None:
        PointContainer.translate(self, vec)
        self.update()


    @StructTools.runAt2DSpace
    def scale2D(self, factors: typingIterable[float]) -> None:
        PointContainer.scale2D(self, factors)
        self.update()


    @StructTools.runAt2DSpace
    def rotate2D(self, rad: float) -> None:
        PointContainer.rotate2D(self, rad)
        self.update()


    @StructTools.runAt2DSpace
    def translate2D(self, vec: typingIterable[float]) -> None:
        PointContainer.translate2D(self, vec)
        self.update()




if __name__ == '__main__':
    contour = [[0, 0], [1, 1], [2, 1]]
    polyLineSeg = PolyLineSeg(contour)
    print(polyLineSeg.getLength())

    print(polyLineSeg.getDimensionNum())
