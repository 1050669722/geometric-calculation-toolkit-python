# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   LineSeg
Description: 
Author:      liu
Date:        1/19/22
---------------------------------------
"""

import numpy as np

from typing import List
from typing import Iterable as typingIterable

from Welt.Utils import Utils
from Welt.Tools.StructTools import StructTools
from Welt.Structs.StructsSingleParticle.Point.Point import Point
from Welt.Structs.StructsSingleParticle.Point.PointContainer import PointContainer


class LineSeg(PointContainer): #TODO: 项目是否已经定义相似数据结构？
    def __init__(self, contour: List[List[float]]):
        super(LineSeg, self).__init__(contour) #这里不将contour填入，调用list类型的空参构造器，因为赋值操作将在下面进行
        assert len(self) == 2

        # 断言 线段 的 两个端点 的 维度数量相等
        assert PointContainer.areDimensionNumsIdentical(self)

        # 更新数据
        LineSeg.update(self) #成员方法update被子类重载了，所以这里显式地调用

        # # 断言 LineSeg的 起点 终点 不相等 #不能这样断言，因为需要首位重合的线段，比如退化为线段的矩形
        # assert not self.startPoint.isEqualToAnother(self.endPoint)


    def update(self):
        # 调用 父类的update()方法
        PointContainer.update(self)
        
        # 添加具有意义的属性名，较低一级类实例化，属性赋值
        self.startPoint = self[0] #Point(contour[0])
        self.endPoint = self[1] #Point(contour[1])

        # 计算 线段 长度
        self.length = Utils.calcDistanceBetweenTwoPoints(self[0], self[1])


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
        return min(self[0][0], self[1][0]), min(self[0][1], self[1][1]), \
               max(self[0][0], self[1][0]), max(self[0][1], self[1][1])


    @StructTools.runAt3DSpace
    def getMinMaxCoord3D(self):
        """
        获取3D-LineSeg最小最大点坐标
        :return: xmin, ymin, zmin, xmax, ymax, zmax
        """
        return min(self[0][0], self[1][0]), min(self[0][1], self[1][1]), min(self[0][2], self[1][2]), \
               max(self[0][0], self[1][0]), max(self[0][1], self[1][1]), max(self[0][2], self[1][2])


    def getLength(self):
        # return Utils.calcDistanceBetweenTwoPoints(self[0], self[1])
        return self.length


    @StructTools.runAt2DSpace
    def getStartEndCoord2D(self):
        """
        获取2D-LineSeg起始点坐标
        :return: xStart, yStart, xEnd, yEnd
        """
        return self[0][0], self[0][1], self[1][0], self[1][1]


    @StructTools.runAt3DSpace
    def getStartEndCoord3D(self):
        """
        获取3D-LineSeg起始点坐标
        :return: xStart, yStart, zStart, xEnd, yEnd, zEnd
        """
        return self[0][0], self[0][1], self[0][2], self[1][0], self[1][1], self[1][2]


    def getDirectionVec(self) -> np.ndarray:
        return np.array([self.endPoint.x - self.startPoint.x, self.endPoint.y - self.startPoint.y])


    def isEqualToAnother(self, lineSeg) -> bool:
        """
        是否与另一个lineSeg相等
        Args:
            lineSeg: 另一个lineSeg
        Returns: 是否相等
        """
        return (self.startPoint.isEqualToAnother(lineSeg.startPoint) and
                self.endPoint.isEqualToAnother(lineSeg.endPoint)) \
               or \
               (self.startPoint.isEqualToAnother(lineSeg.endPoint) and
                self.endPoint.isEqualToAnother(lineSeg.startPoint))


    def isOrthogonalToAnother(self, lineSeg, threshold: float=Utils.Threshold) -> bool:
        # return Utils.isEqual(Utils.dot(self, lineSeg), 0.0, 1e2 * Utils.Threshold)
        # return Utils.isEqual(Utils.dot(self, lineSeg), 0.0, Utils.Threshold)
        return Utils.isEqual(Utils.dot(self, lineSeg), 0.0, threshold)


    def isParallelToAnother(self, lineSeg, threshold: float=Utils.Threshold) -> bool:
        # return Utils.isEqual(np.linalg.norm(Utils.cross(self, lineSeg)), 0.0, 1e2 * Utils.Threshold)
        # return Utils.isEqual(np.linalg.norm(Utils.cross(self, lineSeg)), 0.0, Utils.Threshold)
        return Utils.isEqual(np.linalg.norm(Utils.cross(self, lineSeg)), 0.0, threshold)


    @StructTools.runAt3DSpace
    def isOrthogonalToPlane(self, vecA, vecB):
        """
        判断这个lineSeg是否与某一平面垂直
        这个平面以向量vecA, vecB表示，这两个向量不平行
        :param vecA:
        :param vecB:
        :return:
        """
        assert not vecA.isParallelToAnother(vecB)
        return self.isOrthogonalToAnother(vecA) and self.isOrthogonalToAnother(vecB)


    @StructTools.runAt3DSpace
    def isOrthogonalToPlaneXOY(self):
        return self.isOrthogonalToPlane(LineSeg([[0, 0, 0], [1, 0, 0]]), LineSeg([[0, 0, 0], [0, 1, 0]]))


    @StructTools.runAt3DSpace
    def isOrthogonalToPlaneYOZ(self):
        return self.isOrthogonalToPlane(LineSeg([[0, 0, 0], [0, 1, 0]]), LineSeg([[0, 0, 0], [0, 0, 1]]))


    @StructTools.runAt3DSpace
    def isOrthogonalToPlaneXOZ(self):
        return self.isOrthogonalToPlane(LineSeg([[0, 0, 0], [1, 0, 0]]), LineSeg([[0, 0, 0], [0, 0, 1]]))


    # TODO: 平行于某个平面
    # def isParallelToAPlane(self, vecA, vecB):
    #     pass


    @StructTools.runAt3DSpace
    def scale(self, factors: typingIterable[float]) -> None:
        PointContainer.scale(self, factors)
        LineSeg.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundX(self, rad: float) -> None:
        PointContainer.rotateRadAroundX(self, rad)
        LineSeg.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundY(self, rad: float) -> None:
        PointContainer.rotateRadAroundY(self, rad)
        LineSeg.update(self)


    @StructTools.runAt3DSpace
    def rotateRadAroundZ(self, rad: float) -> None:
        PointContainer.rotateRadAroundZ(self, rad)
        LineSeg.update(self)


    @StructTools.runAt3DSpace
    def translate(self, vec: typingIterable[float]) -> None:
        PointContainer.translate(self, vec)
        LineSeg.update(self)


    @StructTools.runAt2DSpace
    def scale2D(self, factors: typingIterable[float]) -> None:
        PointContainer.scale2D(self, factors)
        LineSeg.update(self)


    @StructTools.runAt2DSpace
    def rotate2D(self, rad: float) -> None:
        PointContainer.rotate2D(self, rad)
        LineSeg.update(self)


    @StructTools.runAt2DSpace
    def translate2D(self, vec: typingIterable[float]) -> None:
        PointContainer.translate2D(self, vec)
        LineSeg.update(self)




if __name__ == '__main__':
    lineSeg = LineSeg(
        [
            Point([0, 0, 0]),
            Point([1, 1, 1]),
        ]
    )
    lineSeg.collapseIntoPhasePlane('x')
    lineSeg.expandFromPhasePlane('x')

    print(lineSeg.getDimensionNum())
    pass
