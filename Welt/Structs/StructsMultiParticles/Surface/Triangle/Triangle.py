# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   Triangle
Description: 
Author:      liu
Date:        1/28/22
---------------------------------------
"""

import sys
import numpy as np

from typing import List
from numbers import Number

if sys.version_info < (3, 8):
    from collections import Iterable
else:
    from collections.abc import Iterable

from Welt.Utils import Utils
from Welt.Tools.StructTools import StructTools
from Welt.Structs.StructsSingleParticle.Point.Point import Point
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSeg import LineSeg
from Welt.Structs.StructsDoubleParticles.Vector import Vector
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSegContainer import LineSegContainer


class Triangle(LineSegContainer):
    def __init__(self, contour):
        # 四变形 3个顶点 或者 3条边
        assert len(contour) == 3

        # 调用父类实例化方法
        super(Triangle, self).__init__(contour)

        # 断言 其中的 线段 的 维度数量 相同
        assert LineSegContainer.areDimensionNumsIdentical(self)

        # 采用父类的isClosed方法
        assert LineSegContainer.isClosed(self)

        # 更新数据
        # self.update()
        Triangle.update(self)


    def update(self):
        """底边为A，逆时针顺序"""

        # 调用 父类的update()方法
        LineSegContainer.update(self)

        # 添加具有意义的属性名，较低一级类实例化，属性赋值
        self.edgeALineSeg = self[0]
        self.edgeBLineSeg = self[1]
        self.edgeCLineSeg = self[2]

        # 添加具有意义的属性名，较低二级类实例化，属性赋值
        self.vertexCA = self[0][0]
        self.vertexAB = self[1][0]
        self.vertexBC = self[2][0]
        
        # 面积
        if not hasattr(self, "area"):
            self.area = self.getAreaAbsAt2D3DSpace()


    def getDimensionNum(self) -> int:
        """
        获取 维度数量
        Returns: 维度数量
        """
        return LineSeg.getDimensionNum(self.edges[0])


    @StructTools.runAt2DSpace
    def getArea(self, isAbs=True):
        return Utils.calcAreaOfPolygon(self.getAllCoordX(), self.getAllCoordY(), isAbs)


    def getAreaAbsAt2D3DSpace(self) -> float:
        return 0.5 * np.linalg.norm(Utils.cross(self.edgeALineSeg, self.edgeCLineSeg))




if __name__ == '__main__':
    triangle = Triangle([[0, 0, 1], [1, 1, 2], [1, 2, 3]])
    print(triangle.getAreaAbsAt2D3DSpace())

    print(triangle.getDimensionNum())
