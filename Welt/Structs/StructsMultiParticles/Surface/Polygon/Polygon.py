# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   Polygon
Description: 
Author:      liu
Date:        1/22/22
---------------------------------------
"""

import sys
# import numpy as np

# from typing import List
from typing import Iterable as typingIterable
# from numbers import Number

# if sys.version_info < (3, 8):
#     from collections import Iterable
# else:
#     from collections.abc import Iterable

from Welt.Utils import Utils
from Welt.Tools.StructTools import StructTools
# from Welt.Structs.StructsDoubleParticles.LineSeg.LineSeg import LineSeg
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSegContainer import LineSegContainer


class Polygon(LineSegContainer):
    def __init__(self, contour):
        # 调用父类实例化方法
        super(Polygon, self).__init__(contour)

        # 断言 顶点数量 大于等于 3
        assert len(self.vertexes) >= 3

        # 采用父类的isClosed方法
        assert LineSegContainer.isClosed(self)

        # 更新数据
        # self.update()
        Polygon.update(self)


    def update(self):
        # 调用 父类的update()方法
        LineSegContainer.update(self)

        pass


    # def getAllCoordX(self):
    #     return [vertex.x for vertex in self.vertexes]
    #
    #
    # def getAllCoordY(self):
    #     return [vertex.y for vertex in self.vertexes]
    #
    #
    # @StructTools.runAt3DSpace
    # def getAllCoordZ(self):
    #     return [vertex.z for vertex in self.vertexes]


    @StructTools.runAt2DSpace
    def getArea(self, isAbs: bool=True):
        # return Utils.calcAreaOfPolygon(self.getAllCoordX(), self.getAllCoordY(), isAbs)
        return LineSegContainer.getArea(self)


    # # 该方法已经迁移至AssemblyAlgos2D.py
    # @StructTools.runAt2DSpace
    # def isConvex(self) -> bool:
    #     """
    #     判断 该多边形是否为 凸多边形
    #     Returns: 该多边形是否为 凸多边形
    #     """
    #     vertexes = Utils.removeRepeatedPoints(self.vertexes)
    #     return len(Utils.getConvexHull(vertexes)) == len(vertexes)


    @StructTools.runAt3DSpace
    def scale(self, factors: typingIterable[float]) -> None:
        LineSegContainer.scale(self, factors)
        # self.update()
        Polygon.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundX(self, rad: float) -> None:
        LineSegContainer.rotateRadAroundX(self, rad)
        # self.update()
        Polygon.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundY(self, rad: float) -> None:
        LineSegContainer.rotateRadAroundY(self, rad)
        # self.update()
        Polygon.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundZ(self, rad: float) -> None:
        LineSegContainer.rotateRadAroundZ(self, rad)
        # self.update()
        Polygon.update(self)


    @StructTools.runAt3DSpace
    def translate(self, vec: typingIterable[float]) -> None:
        LineSegContainer.translate(self, vec)
        # self.update()
        Polygon.update(self)


    @StructTools.runAt2DSpace
    def scale2D(self, factors: typingIterable[float]) -> None:
        LineSegContainer.scale2D(self, factors)
        # self.update()
        Polygon.update(self)


    @StructTools.runAt2DSpace
    def rotate2D(self, rad: float) -> None:
        LineSegContainer.rotate2D(self, rad)
        # self.update()
        Polygon.update(self)


    @StructTools.runAt2DSpace
    def translate2D(self, vec: typingIterable[float]) -> None:
        LineSegContainer.translate2D(self, vec)
        # self.update()
        Polygon.update(self)




if __name__ == '__main__':
    polygon0 = Polygon([
        [0, 0],
        [2000, 0],
        [1800, 1000],
        [0, 1000]
    ])

    polygon1 = Polygon([
        [0, 0],
        [2000, 0],
        [1800, 1000],
        [0, 1000],
        [0, 0],
    ])

    # 这是一个异常的多边形 #可能是由于数值的巨大，而使得MABR或者convexHull出现问题
    polygon2 = Polygon([
        [-14328856858.05197, 1806413674.5490067],
        [78972.7599999983, 316814.5283828284],
        [85931.05766143135, 315937.46975409304],
        [829428488.8714768, -104218545.22989085],
        [12264670413.370293, -1545573613.6033795],
        [86734.9733622088, 321979.3660839322],
        [86118.2657380519, 322057.0989964191],
    ])

    # ### 绘图逻辑 ##############################
    from matplotlib import pyplot as plt

    # for edge in polygon2.edges:
    #     xs = [edge.startPoint.x, edge.endPoint.x]
    #     ys = [edge.startPoint.y, edge.endPoint.y]
    #     plt.plot(xs, ys, color='b')
    #     plt.scatter(xs, ys, color='r')

    xs, ys = [], []
    for vertex in polygon2.vertexes:
        xs.append(vertex.x)
        ys.append(vertex.y)
    plt.plot(xs, ys, color='b')
    plt.scatter(xs, ys, color='r')

    plt.axis("equal")
    plt.show()
    # #########################################

    pass
