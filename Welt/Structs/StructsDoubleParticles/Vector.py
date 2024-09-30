# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   Vector
Description: 
Author:      liu
Date:        1/20/22
---------------------------------------
"""

import more_itertools
import numpy as np

from typing import List
from typing import Tuple
from copy import deepcopy
from numbers import Real

from Welt.Utils import Utils
from Welt.Structs.StructsSingleParticle.Point.Point import Point
from Welt.Structs.StructsDoubleParticles.LineSeg.LineSeg import LineSeg


class Vector(LineSeg):
    """
    向量 有向线段
    本类中所返回的向量均是以[0, 0]或[0, 0, 0]为起点的numpy.array类型
    TODO： 可能需要将本类做成类型封闭的类型，尤其是对于dot, cross等方法
    TODO： 或者本类只是作为一个工具，那些原本不能返回本类向量的方法，就仅作为内部方法，而本类对外部提供高级功能的调用接口；这可能就是个量角器；
    """
    def __init__(self, contour: List[List[float]]):
        super(Vector, self).__init__(contour)

        # 更新数据
        # self.update()
        Vector.update(self)


    def update(self):
        # # 这里的父类方法顺序不对劲，所以这里写了LineSeg.update(self)
        # # 问题应该这样解释，super().__init__()会调用父类的初始化方法，包括其中的update方法，但是子类又重载了update方法，所以super().__init__()会寻找到子类重载的方法，而不是父类的方法；如果仍然想要使用父类的被重载方法，解决的办法也很简单：1.在父类的调用被重载方法处显式地调用这一被重载方法；2.在子类的重载方法中调用父类的被重载方法；
        # # 采用父类的update方法
        # LineSeg.update(self) #已经在LineSeg中修改过来

        # 维护一个__vector属性，用于进行向量相关计算
        self.vector = np.array(self.endPoint) - np.array(self.startPoint)


    def getNorm(self, ord=2):
        return np.linalg.norm(self.vector, ord=ord)


    # def getStartEndVertexes(self):
    #     return list(np.array(self[:]))


    # def getDirectionVectorStartEndVertexes(self):
    #     return list(np.array(self.getStartEndVertexes()) / self.getNorm())


    def getUnitDirectVec(self):
        # return Vector(self.getDirectionVectorStartEndVertexes())
        return self.vector / self.getNorm()


    def getUnitVerticalVecs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算self.vector的垂直向量
        Returns: 元组 存储 单位垂直向量
        """
        unitVerVec0, unitVerVec1 = Utils.getVerticalVectors(self.vector)
        unitVerVec0, unitVerVec1 = Utils.normalize(unitVerVec0), Utils.normalize(unitVerVec1)
        return unitVerVec0, unitVerVec1


    def normalize(self):
        # self[:] = self.getDirectionVectorStartEndVertexes()
        # self.update()
        self.vector = self.getUnitDirectVec()


    def __mul__(self, scalar: Real) -> np.ndarray: # -> None:
        """
        将 Vector中的 vector进行数乘运算
        Args:
            scalar: 数乘因子
        Returns: None
        """
        # TODO: 须检查项目中有没有用过此方法
        # self.vector *= scalar
        return self.vector * scalar


    def changeNorm(self, norm: Real) -> None:
        """
        将 Vector中的 vector的模长 更改为 norm
        Args:
            norm: 修改后的向量模长
        Returns: None
        """
        self.normalize()
        self.vector = self.__mul__(norm)


    def dot(self, vec):
        # return Utils.dot(self.getStartEndVertexes(), vec.getStartEndVertexes())
        return np.dot(self.vector, vec.vector) #self.__vector.dot(vec.__vector)


    def cross(self, vec):
        # return Utils.cross(self.getStartEndVertexes(), vec.getStartEndVertexes())
        return np.cross(self.vector, vec.vector)


    # def getRadToAnother(self, vec): #已经修改成调用Utils.calcAngleBetTwoVecs()方法的形式
    #     """
    #     废弃：注释掉的这套逻辑好像原本只是用来判断二维向量之间所夹弧度的，
    #     废弃：self相对于vec的所夹弧度角[，就是从vec转到self]，取值范围：(-pi, pi]
    #
    #     现有：新增的这套逻辑可以计算空间中任意两个向量所夹弧度角
    #     现有：从self到vec所经历的弧度，取值范围(-pi, pi]
    #
    #     :param vec:
    #     :return:
    #     """
    #     # # 自创方法
    #     # dotProd = vec.__getDirectionVector().dot(self.__getDirectionVector())
    #     # crossProd = np.cross(vec.__getDirectionVector(), self.__getDirectionVector()) #vec.__getDirectionVector().cross(self.__getDirectionVector())
    #     #
    #     # if len(crossProd.shape) != 0:
    #     #     crossProd = crossProd[2]
    #     #
    #     # if crossProd >= 0.0:
    #     #     return np.arccos(dotProd)
    #     # else:
    #     #     return -np.arccos(dotProd)
    #
    #     # 方法来自于 https://www.cnblogs.com/lovebay/p/11411512.html
    #     # 叉积
    #     crossProd = self.cross(vec) #np.cross(self.__vector, vec.__vector)
    #     # 点积
    #     dotProd = self.dot(vec) #np.dot(self.__vector, vec.__vector)
    #     # 由正切值计算弧度
    #     rad = np.arctan2(np.linalg.norm(crossProd), dotProd)
    #     # 判断方向
    #     # 3D
    #     if len(crossProd.shape) != 0:
    #         isOrientationPositive = crossProd[2] >= 0.0
    #     # 2D
    #     else:
    #         isOrientationPositive = crossProd >= 0.0
    #     # 是否反向
    #     if not isOrientationPositive:
    #         rad *= -1
    #     # 返回
    #     return rad# * 180.0 / np.pi


    def getAngleToAnother(self, vec, isRadian: bool=True) -> float:
        return Utils.calcAngleBetTwoVecs(self.vector, vec.vector, isRadian)


    @staticmethod
    def generateVectorFromPointAndDirection(point: List[float], directionWithLength: np.ndarray):
        """
        根据 点, 方向 实例化一个Vector
        Args:
            point: 点
            directionWithLength: 方向
        Returns: Vector
        """
        # 维度
        dim = 0

        # 如果point非Point类对象
        if not isinstance(point, Point):
            if more_itertools.ilen(point) == 2:
                dim = 2
                point = Point([point[0], point[1]])
            elif more_itertools.ilen(point) == 3:
                dim = 3
                point = Point([point[0], point[1], point[2]])
            else:
                raise ValueError("[ERROR] Invalid shape of parameter \"point\": {}".format(point))

        # 如果point为Point类对象
        else:
            if len(point[:]) == 2:
                dim = 2
            elif len(point[:]) == 3:
                dim = 3

        # 断言 direction的维度 与 dim 相符
        assert directionWithLength.shape[0] == dim

        # point的副本
        pointDup = deepcopy(point)

        # 根据dim 平移pointDup
        if dim == 2:
            pointDup.translate2D(directionWithLength)
        elif dim == 3:
            pointDup.translate(directionWithLength)
        else:
            raise ValueError("[ERROR] Invalid value of \"dim\": {}".format(dim))

        # 实例化Vector
        vec = Vector([point, pointDup])

        # 返回
        return vec


# # TODO: 删除这个函数，在现有的应用项目中，解除这个依赖
# def generateVectorFromPointAndDirection(point: List[float], direction: np.ndarray) -> Vector:
#     """
#     根据 点, 方向 实例化一个Vector
#     Args:
#         point: 点
#         direction: 方向
#     Returns: Vector
#     """
#     # 维度
#     dim = 0
#
#     # 如果point非Point类对象
#     if not isinstance(point, Point):
#         if more_itertools.ilen(point) == 2:
#             dim = 2
#             point = Point([point[0], point[1]])
#         elif more_itertools.ilen(point) == 3:
#             dim = 3
#             point = Point([point[0], point[1], point[2]])
#         else:
#             raise ValueError("[ERROR] Invalid shape of parameter \"point\": {}".format(point))
#
#     # 如果point为Point类对象
#     else:
#         if len(point[:]) == 2:
#             dim = 2
#         elif len(point[:]) == 3:
#             dim = 3
#
#     # 断言 direction的维度 与 dim 相符
#     assert direction.shape[0] == dim
#
#     # point的副本
#     pointDup = deepcopy(point)
#
#     # 根据dim 平移pointDup
#     if dim == 2:
#         pointDup.translate2D(direction)
#     elif dim == 3:
#         pointDup.translate(direction)
#     else:
#         raise ValueError("[ERROR] Invalid value of \"dim\": {}".format(dim))
#
#     # 实例化Vector
#     vec = Vector([point, pointDup])
#
#     # 返回
#     return vec




if __name__ == '__main__':
    # vec0 = Vector([[0, 0], [1, 0]])
    # # vec1 = Vector([[0, 0], [1, 1]])
    # # vec1 = Vector([[0, 0], [0, 1]])
    # # vec1 = Vector([[0, 0], [-1, 1]])
    # vec1 = Vector([[0, 0], [-1, 0]])
    # # vec1 = Vector([[0, 0], [-1, -1]])
    # # vec1 = Vector([[0, 0], [0, -1]])
    # # vec1 = Vector([[0, 0], [1, -1]])
    # # vec1 = Vector([[0, 0], [1, 0]])
    # print(vec0.__getDirectionVector().dot(vec1.__getDirectionVector()))
    # print(vec0.__getDirectionVector().cross(vec1.__getDirectionVector()))
    # print(vec1.getRadToAnother(vec0) * 180 / np.pi)

    # vecA = Vector([[0, 0, 1], [1, 0, 1]])
    # vecB = Vector([[0, 0, 0], [0, -1, 0]])
    # # vecB = Vector([[0, 0, 0], [-1, 0, 0]])
    # # vecB = Vector([[0, 0, 0], [0, 1, 0]])
    # # vecB = Vector([[0, 0, 0], [1, 1, 0]])
    # # vecB = Vector([[0, 0, 0], [1, -1, 0]])
    # # vecB = Vector([[0, 0, 0], [1, 1, 1]])
    # # vecB = Vector([[0, 0, 0], [-1, -1, 1]])
    # print(vecA.getRadToAnother(vecB))

    # vecA = Vector([[0, 0], [1, 0]])
    # vecB = Vector([[0, 0], [0, 1]])
    # print(vecA.getRadToAnother(vecB))


    point = [2.0, 3.0]
    vec = np.array([0, 1])
    resVec = Vector.generateVectorFromPointAndDirection(point, vec)

    pass
