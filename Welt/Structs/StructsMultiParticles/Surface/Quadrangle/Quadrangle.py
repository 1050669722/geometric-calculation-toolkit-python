# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   Quadrangle
Description: 
Author:      liu
Date:        1/18/22
---------------------------------------
"""

import sys

# from typing import List
from typing import Iterable as typingIterable
# from numbers import Number

# if sys.version_info < (3, 8):
#     from collections import Iterable
# else:
#     from collections.abc import Iterable

from Welt.Utils import Utils
from Welt.Tools.StructTools import StructTools
# from Welt.Structs.StructsSingleParticle.Point.Point import Point
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSeg import LineSeg
from Welt.Structs.StructsDoubleParticles.Vector import Vector
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSegContainer import LineSegContainer


class Quadrangle(LineSegContainer):
    def __init__(self, contour):
        # 四变形 4个顶点 或者 4条边
        assert len(contour) == 4

        # 调用父类实例化方法
        super(Quadrangle, self).__init__(contour)

        # 断言 其中的 线段 的 维度数量 相同
        assert LineSegContainer.areDimensionNumsIdentical(self)

        # 采用父类的isClosed方法
        assert LineSegContainer.isClosed(self)

        # 更新数据
        # self.update()
        Quadrangle.update(self)


    def update(self):
        """底边为A，逆时针顺序"""

        # 调用 父类的update()方法
        LineSegContainer.update(self)

        # 添加具有意义的属性名，较低一级类实例化，属性赋值
        self.edgeALineSeg = self[0] #LineSeg(contourList[0])
        self.edgeBLineSeg = self[1] #LineSeg(contourList[1])
        self.edgeCLineSeg = self[2] #LineSeg(contourList[2])
        self.edgeDLineSeg = self[3] #LineSeg(contourList[3])

        # 添加具有意义的属性名，较低二级类实例化，属性赋值
        self.vertexDA = self[0][0] #Point(contourList[0][0])
        self.vertexAB = self[1][0] #Point(contourList[1][0])
        self.vertexBC = self[2][0] #Point(contourList[2][0])
        self.vertexCD = self[3][0] #Point(contourList[3][0]) #TODO： 添加获取最小和最大的坐标的方法


    def getDimensionNum(self) -> int:
        """
        获取 维度数量
        Returns: 维度数量
        """
        return LineSeg.getDimensionNum(self.edges[0])


    # def getAllCoordX(self):
    #     return [self.vertexDA.x, self.vertexAB.x, self.vertexBC.x, self.vertexCD.x]
    #
    #
    # def getAllCoordY(self):
    #     return [self.vertexDA.y, self.vertexAB.y, self.vertexBC.y, self.vertexCD.y]
    #
    #
    # @StructTools.runAt3DSpace
    # def getAllCoordZ(self):
    #     return [self.vertexDA.z, self.vertexAB.z, self.vertexBC.z, self.vertexCD.z]


    # @StructTools.runAt2DSpace
    # def getArea(self, isAbs=True):
    #     # return Utils.calcAreaOfPolygon(self.getAllCoordX(), self.getAllCoordY(), isAbs)
    #     return LineSegContainer.getArea(isAbs)


    @StructTools.runAt3DSpace
    def fourVertexesOnTheSamePlane(self):
        return self.vertexDA.isAtPlaneComposedOfThreeOther(self.vertexAB, self.vertexBC, self.vertexCD)


    # TODO: 以下三个方法都可以扩展为3D的
    @StructTools.runAt2DSpace
    def isParallelogram(self, threshold: float=Utils.Threshold) -> bool:
        # 两对对边对应平行
        return \
            self.edgeALineSeg.isParallelToAnother(self.edgeCLineSeg, threshold) \
            and \
            self.edgeBLineSeg.isParallelToAnother(self.edgeDLineSeg, threshold)


    @StructTools.runAt2DSpace
    def isRectangle(self, threshold: float=Utils.Threshold) -> bool:
        # 两对对边对应平行
        if not self.isParallelogram(threshold):
            return False
        # 一对临边相互垂直
        return self.edgeALineSeg.isOrthogonalToAnother(self.edgeBLineSeg, threshold)


    @StructTools.runAt2DSpace
    def isOneLinesegHorizontal(self):
        """
        某条边是否水平
        :return: bool
        """
        return \
            any([
                self.edgeALineSeg.isParallelToAnother(Vector([[0, 0], [1, 0]])), #或者写LineSeg([[0, 0], [1, 0]])
                self.edgeBLineSeg.isParallelToAnother(Vector([[0, 0], [1, 0]])),
                self.edgeCLineSeg.isParallelToAnother(Vector([[0, 0], [1, 0]])),
                self.edgeDLineSeg.isParallelToAnother(Vector([[0, 0], [1, 0]]))
            ])


    @StructTools.runAt3DSpace
    def hasOneEdgeOrthogonalToPlaneXOY(self):
        """
        判断四变形至少存在一条边垂直于xOy平面
        :return:
        """
        for edge in self[:]:
            if edge.isOrthogonalToPlaneXOY():
                return True
        return False


    @StructTools.runAt3DSpace
    def scale(self, factors: typingIterable[float]) -> None:
        LineSegContainer.scale(self, factors)
        # self.update()
        Quadrangle.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundX(self, rad: float) -> None:
        LineSegContainer.rotateRadAroundX(self, rad)
        # self.update()
        Quadrangle.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundY(self, rad: float) -> None:
        LineSegContainer.rotateRadAroundY(self, rad)
        # self.update()
        Quadrangle.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundZ(self, rad: float) -> None:
        LineSegContainer.rotateRadAroundZ(self, rad)
        # self.update()
        Quadrangle.update(self)


    @StructTools.runAt3DSpace
    def translate(self, vec: typingIterable[float]) -> None:
        LineSegContainer.translate(self, vec)
        # self.update()
        Quadrangle.update(self)


    @StructTools.runAt2DSpace
    def scale2D(self, factors: typingIterable[float]) -> None:
        LineSegContainer.scale2D(self, factors)
        # self.update()
        Quadrangle.update(self)


    @StructTools.runAt2DSpace
    def rotate2D(self, rad: float) -> None:
        LineSegContainer.rotate2D(self, rad)
        # self.update()
        Quadrangle.update(self)


    @StructTools.runAt2DSpace
    def translate2D(self, vec: typingIterable[float]) -> None:
        LineSegContainer.translate2D(self, vec)
        # self.update()
        Quadrangle.update(self)




if __name__ == '__main__':
    # contour = [Point([0, 0, 0]), Point([0, 3000, 0]), Point([0, 3000, 3000]), Point([0, 0, 3000])]
    # quadrangle = Quadrangle(contour)

    contour = [[[0, 0, 0], [1, 0, 0]],
               [[1, 0, 0], [1, 1, 0]],
               [[1, 1, 0], [0, 1, 0]],
               [[0, 1, 0], [0, 0, 0]]]
    quadrangle = Quadrangle(contour)

    print(quadrangle.getDimensionNum())
    pass
