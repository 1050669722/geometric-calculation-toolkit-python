# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   Utils
Description: 
Author:      liu
Date:        1/19/22
---------------------------------------
"""

import os
import sys
import time
import numpy as np
import sympy as sp
import more_itertools
import pickle
import json

from typing import List
from typing import Tuple
from typing import Iterable as typingIterable
from numbers import Number
from inspect import isfunction, ismethod

if sys.version_info < (3, 8):
    from collections import Iterable
else:
    from collections.abc import Iterable

from Welt.Constants import Constants


class Utils(object):
    """
    这里的数据结构仅仅是Python内部的数据结构
    不依赖任何自定义的数据结构
    """
    Threshold = Constants.Threshold

    @staticmethod
    def timeCounter(oriFunc):
        def dstFunc(*args, **kwargs):  # *args, **kwargs为了传递原来函数的参数
            print("[INFO] \"{}\" start".format(oriFunc.__name__))
            time0 = time.perf_counter()
            results = oriFunc(*args, **kwargs)
            print("[INFO] \"{}\" done, {:.15f} seconds elapsed".format(oriFunc.__name__, time.perf_counter() - time0))
            return results
        return dstFunc


    @staticmethod
    def feedBackRunningCode(oriFunc):
        """
        反馈状态码
        Returns: results, 状态码
        """
        def dstFunc(*args, **kwargs):
            msgs = args[0] #TODO: 装饰器参数 #异常 错误码 字典
            # TODO: 临时这样写，是为了调试
            try:
                results = oriFunc(*args, **kwargs)
                msgs.append("Completed") #msgs.append(0)
                return results
            # TODO: 针对不同异常的错误码的描述
            except:
                print("[ERROR] \"{}\" failed".format(oriFunc.__name__))
                msgs.append("Failed") #msgs.append(-1)
                return None
        return dstFunc


    @staticmethod
    def getPathNameExtFromPath(path):
        """
        从文件的路径中 获取 文件所在路径 文件名 扩展名
        path: 文件路径
        return: 文件所在路径 文件名 扩展名
        """
        # 分离 扩展名
        filename, extname = os.path.splitext(path)[0], os.path.splitext(path)[1]
        # 分离 文件名
        filepath, filename = os.path.split(filename)[:-1], os.path.split(filename)[-1]
        if isinstance(filepath, Iterable):
            filepath = filepath[0]
        # 返回 文件所在路径 文件名 扩展名
        return filepath, filename, extname


    @classmethod
    def save(cls, obj, name, format: str="pkl") -> None:
        if format.lower() == "pkl":
            cls._savePkl(obj, name, mode="wb")
        elif format.lower() == "json":
            cls._saveJson(obj, name, mode="w")
        else:
            raise ValueError("Expect parameter 'format' be 'pkl' or 'json'")


    @staticmethod
    def _savePkl(obj, name, mode="wb"):
        """
        保存
        :param obj: 对象
        :param name: 名字
        :param mode: 打开方式，默认"wb"
        :return: None
        """
        if not name.endswith('.pkl'):
            name += '.pkl'
        with open(name, mode) as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return None


    @staticmethod
    def _saveJson(obj, name, mode="w"):
        """
        保存
        :param obj: 对象
        :param name: 名字
        :param mode: 打开方式，默认"wb"
        :return: None
        """
        if not name.endswith('.json'):
            name += '.json'
        with open(name, mode, encoding='utf-8') as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)
        return None


    @classmethod
    def load(cls, name, format: str="pkl"):
        if format.lower() == "pkl":
            return cls._loadPkl(name)
        elif format.lower() == "json":
            return cls._loadJson(name)
        else:
            raise ValueError("Expect parameter 'format' be 'pkl' or 'json'")


    @staticmethod
    def _loadPkl(name):
        """
        读取
        :param name: 名字
        :return: 对象
        """
        if not name.endswith('.pkl'):
            name += '.pkl'
        with open(name, 'rb') as f:
            return pickle.load(f)


    @staticmethod
    def _loadJson(name):
        """
        读取
        :param name: 名字
        :return: 对象
        """
        if not name.endswith('.json'):
            name += '.json'
        with open(name, 'r', encoding='utf-8') as f:
            return json.load(f)


    @staticmethod
    def calcIndsATTotalNumGroupNum(totalNum: int, groupNum: int) -> Tuple[List[int], List[int]]:
        """
        根据 总数量 和 分组数 计算 索引 | 如果多余，在后面追加
        Args:
            totalNum: 总数量
            groupNum: 分组数
        Returns: 元组[容器 存储 索引]
        """
        # 断言 totalNum, groupNum 为 整数
        assert isinstance(totalNum, int) and isinstance(groupNum, int)

        # groupNum 不能大于 totalNum
        groupNum = min(groupNum, totalNum)

        # 单元尺寸
        unitSize = totalNum // groupNum

        # 容器 存储 索引
        startInds, endInds = [], []

        # 初始 起点索引 和 终点索引
        startInd = 0
        endInd = startInd + unitSize

        # 当终点索引 小于 总数量时
        while (endInd < totalNum):
            # 添加 起点索引 终点索引
            startInds.append(startInd)
            endInds.append(endInd)

            # 更新 起点索引 终点索引
            startInd = endInd
            endInd = startInd + unitSize

        # 最后 添加 起点索引 终点索引
        startInds.append(startInd)
        endInds.append(totalNum)

        # 返回
        return startInds, endInds



    @classmethod
    def isEqual(cls, a, b, threshold=Threshold) -> bool:
        return abs(a - b) <= threshold


    @classmethod
    def isLessEqual(cls, a, b, threshold=Threshold) -> bool:
        return a < b or cls.isEqual(a, b, threshold)


    @classmethod
    def isGreatEqual(cls, a, b, threshold=Threshold) -> bool:
        return a > b or cls.isEqual(a, b, threshold)


    @staticmethod
    def mksureValueInRange(value: float, floor: float, ceil: float) -> float:
        """
        确保value位于[floor, ceil]范围
        Args:
            value: 待修改值
            floor: 下限
            ceil: 上限
        Returns:
        """
        return min(max(value, floor), ceil)


    @staticmethod #2D 3D
    def calcDistanceBetweenTwoPoints(pA: List[float], pB: List[float]) -> float:
        """
        计算两点之间的距离
        :param pA:
        :param pB:
        :return: float
        """
        assert len(pA) == len(pB)
        # if len(pA) == 2:
        #     return np.sqrt((pA[0] - pB[0]) ** 2 + (pA[1] - pB[1]) ** 2)
        # elif len(pA) == 3:
        #     return np.sqrt((pA[0] - pB[0]) ** 2 + (pA[1] - pB[1]) ** 2 + (pA[2] - pB[2]) ** 2)
        # else:
        #     return np.linalg.norm(np.array(pA) - np.array(pB), ord=2)
        return np.linalg.norm(np.array(pA) - np.array(pB), ord=2)


    @classmethod
    def areTwoPointsCoincide(cls, pt0: typingIterable[float], pt1: typingIterable[float], isLoose: bool=True) -> bool:
        """
        判断两点是否重合
        Args:
            pt0: 点0
            pt1: 点1
        Returns: 这两点是否重合
        """
        # 确保输入的点为二维平面中的点
        assert more_itertools.ilen(pt0) == 2 and more_itertools.ilen(pt1) == 2

        # 确保pt0, pt1, pt2可以用“[]”索引
        assert hasattr(pt0, "__getitem__") and hasattr(pt1, "__getitem__")

        # 构造一个向量
        vec = np.array([pt1[0] - pt0[0], pt1[1] - pt0[1]])

        # 返回 这个向量 的模长 是否为0
        if isLoose:
            return cls.isEqual(np.linalg.norm(vec), 0.0)
        else:
            return np.linalg.norm(vec) == 0.0


    @classmethod
    def areThreePointsCollinear(cls, pt0: typingIterable[float], pt1: typingIterable[float], pt2: typingIterable[float]) -> bool:
        """
        判断三点是否共线
        Args:
            pt0: 点0
            pt1: 点1
            pt2: 点2
        Returns: 这三点是否共线
        """
        # 确保输入的点为三维空间中的点
        assert more_itertools.ilen(pt0) == 3 and more_itertools.ilen(pt1) == 3 and more_itertools.ilen(pt2) == 3

        # 确保pt0, pt1, pt2可以用“[]”索引
        assert hasattr(pt0, "__getitem__") and hasattr(pt1, "__getitem__") and hasattr(pt2, "__getitem__")

        # 构造两个向量
        vecA = np.array([pt1[0] - pt0[0], pt1[1] - pt0[1], pt1[2] - pt0[2]])
        vecB = np.array([pt2[0] - pt0[0], pt2[1] - pt0[1], pt2[2] - pt0[2]])

        # 返回 这两个向量 的叉积 的模长 是否为0
        # return np.linalg.norm(cls.cross(vecA, vecB)) == 0
        return cls.isEqual(np.linalg.norm(cls.cross(vecA, vecB)), 0.0, 1e2 * cls.Threshold)


    @staticmethod
    def getCentroid(vertexes: List[List[float]]) -> List[float]:
        """
        获取 形心
        Args:
            vertexes: 顶点容器
        Returns: 形心
        """
        # 维度判定
        if all([more_itertools.ilen(vertex) == 2 for vertex in vertexes]):
            dim = 2
        elif all([more_itertools.ilen(vertex) == 3 for vertex in vertexes]):
            dim = 3
        else:
            raise ValueError("[ERROR] Invalid dimension of input vertexes")

        # 计算形心 x, y
        x, y = 0.0, 0.0
        for vertex in vertexes:
            x += vertex[0]
            y += vertex[1]
        x /= len(vertexes)
        y /= len(vertexes)

        # 计算形心 z
        if dim == 2:
            return [x, y]
        elif dim == 3:
            z = 0.0
            for vertex in vertexes:
                z += vertex[2]
            z /= len(vertexes)
            return [x, y, z]


    @staticmethod #2D
    def calcAreaOfPolygon(coordX, coordY, isAbs=True):
        """
        计算任意多边形（面）面积
        :param coordX: 所有横坐标
        :param coordY: 所有纵坐标
        :param isAbs: 是否取绝对值（否则，逆时针为正，顺时针为负）
        :return: float
        """
        area = 0.0

        coordX = list(coordX) + [coordX[0]]
        coordY = list(coordY) + [coordY[0]]

        for idx in range(0, len(coordX) - 1, 1):
            area += (coordX[idx] + coordX[idx + 1]) * (coordY[idx + 1] - coordY[idx])

        if isAbs:
            area = abs(area)

        return 0.5 * area


    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        """
        将向量归一化，并返回；non in-place
        Args:
            vec: 待归一化向量
        Returns: 归一化后的向量
        """
        return vec / np.linalg.norm(vec)


    @staticmethod #2D 3D
    def dot(vecA, vecB) -> np.ndarray:
        """
        计算两个向量的点积
        向量可以由一组坐标表示，也可以由两组点坐标表示
        :param vecA: [x, y, (z)] or [[startX, strartY, (startZ)], [endX, endY, (endZ)]]
        :param vecB: [x, y, (z)] or [[startX, strartY, (startZ)], [endX, endY, (endZ)]]
        :return: dot product of vecA and vecB
        """
        if isinstance(vecA[0], Number) and isinstance(vecA[1], Number) and \
            isinstance(vecB[0], Number) and isinstance(vecB[1], Number):
            return np.dot(np.array(vecA), np.array(vecB))

        elif isinstance(vecA[0], Iterable) and isinstance(vecA[1], Iterable) and \
            isinstance(vecB[0], Iterable) and isinstance(vecB[1], Iterable):
            return np.dot(np.array(vecA[1]) - np.array(vecA[0]),
                          np.array(vecB[1]) - np.array(vecB[0]))

        elif isinstance(vecA[0], Number) and isinstance(vecA[1], Number) and \
            isinstance(vecB[0], Iterable) and isinstance(vecB[1], Iterable):
            return np.dot(np.array(vecA), np.array(vecB[1]) - np.array(vecB[0]))

        elif isinstance(vecA[0], Iterable) and isinstance(vecA[1], Iterable) and \
            isinstance(vecB[0], Number) and isinstance(vecB[1], Number):
            return np.dot(np.array(vecA[1]) - np.array(vecA[0]),
                          np.array(vecB))

        else:
            raise TypeError("[ERROR] The parameters \"vecA\": {} and \"vecB\": {} may be the instance of different or invalid type: {} and {}"\
                            .format(vecA, vecB, type(vecA), type(vecB)))


    @staticmethod #2D 3D
    def cross(vecA, vecB) -> np.ndarray:
        """
        计算两个向量的叉积
        向量可以由一组坐标表示，也可以由两组点坐标表示
        :param vecA: [x, y, (z)] or [[startX, strartY, (startZ)], [endX, endY, (endZ)]] or np.ndarray
        :param vecB: [x, y, (z)] or [[startX, strartY, (startZ)], [endX, endY, (endZ)]] or np.ndarray
        :return: cross product of vecA and vecB
        """
        if isinstance(vecA[0], Number) and isinstance(vecA[1], Number) and \
                isinstance(vecB[0], Number) and isinstance(vecB[1], Number):
            return np.cross(np.array(vecA), np.array(vecB))

        elif isinstance(vecA[0], Iterable) and isinstance(vecA[1], Iterable) and \
                isinstance(vecB[0], Iterable) and isinstance(vecB[1], Iterable):
            return np.cross(np.array(vecA[1]) - np.array(vecA[0]),
                          np.array(vecB[1]) - np.array(vecB[0]))

        elif isinstance(vecA[0], Number) and isinstance(vecA[1], Number) and \
                isinstance(vecB[0], Iterable) and isinstance(vecB[1], Iterable):
            return np.cross(np.array(vecA), np.array(vecB[1]) - np.array(vecB[0]))

        elif isinstance(vecA[0], Iterable) and isinstance(vecA[1], Iterable) and \
                isinstance(vecB[0], Number) and isinstance(vecB[1], Number):
            return np.cross(np.array(vecA[1]) - np.array(vecA[0]),
                          np.array(vecB))

        elif isinstance(vecA, np.ndarray) and isinstance(vecB, np.ndarray):
            return np.cross(vecA, vecB)

        else:
            raise TypeError(
                "[ERROR] The parameters \"vecA\": {} and \"vecB\": {} may be the instance of different or invalid type: {} and {}" \
                .format(vecA, vecB, type(vecA), type(vecB)))


    @classmethod
    def calcAngleBetTwoVecs(cls, vecA: np.ndarray, vecB: np.ndarray, isRadian: bool=True) -> float:
        """
        计算两个向量之间的夹角（弧度制），取值范围(-pi, pi]
        Args:
            vecA: 向量A
            vecB: 向量B
        Returns: 从 向量A 到 向量B 的夹角（默认为弧度制）
        """
        # 确保vecA和vecB是二维或者三维向量
        assert len(vecA.shape) == 1 and len(vecB.shape) == 1
        assert vecA.shape[0] in {2, 3} and vecB.shape[0] in {2, 3} and vecA.shape[0] == vecB.shape[0]

        # 归一化
        vecA = cls.normalize(vecA)
        vecB = cls.normalize(vecB)

        # 点积
        dotProd = np.dot(vecA, vecB)

        # 叉积
        crossProd = np.cross(vecA, vecB)

        # 返回值符号判据
        # 如果点是二维空间的点
        if vecA.shape[0] == 2:
            criterion = crossProd
        # 如果点是三维空间的点
        else:
            # 如果两个向量共线
            if cls.areTwoVectorsCollinear(vecA, vecB):
                criterion = np.linalg.norm(crossProd)  # 0.0，即向量共线时，criterion取0.0，此时返回值为pi，对应取值范围是(-pi, pi]
            # 如果两个向量不共线
            else:
                A, B, C, D = cls.getCoeffsOfPlaneEquation([0, 0, 0], vecA, vecB)
                criterion = A * crossProd[0] + B * crossProd[1] + C * crossProd[2] + D

        # 返回
        if criterion >= 0:
            angle = np.arccos(cls.mksureValueInRange(float(dotProd), -1.0, 1.0))
        else:
            angle = -np.arccos(cls.mksureValueInRange(float(dotProd), -1.0, 1.0))
        return angle if isRadian else angle * 180 / np.pi


    @classmethod
    def areTwoVectorsCollinear(cls, vecA: np.ndarray, vecB: np.ndarray) -> bool:
        """
        判断两个向量是否共线（平行）
        Args:
            vecA: 向量A
            vecB: 向量B
        Returns: 是否共线
        """
        # 确保vecA和vecB是二维或者三维向量
        assert len(vecA.shape) == 1 and len(vecB.shape) == 1
        assert vecA.shape[0] in {2, 3} and vecB.shape[0] in {2, 3} and vecA.shape[0] == vecB.shape[0]

        # 返回 这两个向量 的叉积 的模长 是否为0
        return cls.isEqual(np.linalg.norm(cls.cross(vecA, vecB)), 0.0, 1e2 * cls.Threshold)

        # 不能用cls.calcAngleBetTwoVecs()，因为cls.calcAngleBetTwoVecs中已经用到了areTwoVectorsCollinear()这个方法
        # # 返回 这两个向量的夹角 是否为-pi或0或pi #事实上，-pi的情况并不存在
        # angle = cls.calcAngleBetTwoVecs(vecA, vecB)
        # return (Utils.isEqual(angle, -np.pi)) or (Utils.isEqual(angle, 0.0)) or (Utils.isEqual(angle, np.pi))


    @classmethod
    def areTwoVectorsInSameDirection(cls, vecA: np.ndarray, vecB: np.ndarray) -> bool:
        """
        判断两个向量 是否 同方向
        Args:
            vecA: 向量A
            vecB: 向量B
        Returns: 是否 同方向
        """
        if not cls.areTwoVectorsCollinear(vecA, vecB):
            return False
        return Utils.isEqual(Utils.calcAngleBetTwoVecs(vecA, vecB), 0.0, 1e2 * cls.Threshold)


    def areTwoVectorsInReverseDirections(cls, vecA: np.ndarray, vecB: np.ndarray) -> bool:
        """
        判断两个向量 是否 反方向
        Args:
            vecA: 向量A
            vecB: 向量B
        Returns: 是否 反方向
        """
        if not cls.areTwoVectorsCollinear(vecA, vecB):
            return False
        return Utils.isEqual(Utils.calcAngleBetTwoVecs(vecA, vecB), np.pi, 1e2 * cls.Threshold) or \
               Utils.isEqual(Utils.calcAngleBetTwoVecs(vecA, vecB), -np.pi, 1e2 * cls.Threshold)


    @classmethod
    def areTwoVectorsOrthogonal(cls, vecA: np.ndarray, vecB: np.ndarray) -> bool:
        """
        判断两个向量是否正交（垂直）
        Args:
            vecA: 向量A
            vecB: 向量B
        Returns: 是否正交
        """
        # 确保vecA和vecB是二维或者三维向量
        assert len(vecA.shape) == 1 and len(vecB.shape) == 1
        assert vecA.shape[0] in {2, 3} and vecB.shape[0] in {2, 3} and vecA.shape[0] == vecB.shape[0]

        # # 返回 这两个向量 的点积 是否为0
        # return cls.dot(vecA, vecB) == 0.0

        # 返回 这两个向量的夹角 是否为-pi/2或pi/2
        angle = cls.calcAngleBetTwoVecs(vecA, vecB)
        return (cls.isEqual(angle, -0.5 * np.pi)) or (cls.isEqual(angle, 0.5 * np.pi))


    @staticmethod #2D
    def getVerticalVectors(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算一个二维向量的垂直向量
        Args:
            vec: 一个二维向量
        Returns: 元组[垂直向量0, 垂直向量1]
        """
        # 确保vec是二维向量
        assert len(vec.shape) == 1
        assert vec.shape[0] == 2

        # 垂直向量0
        verticalVec0 = np.array([vec[1], -vec[0]])

        # 垂直向量1
        verticalVec1 = np.array([-vec[1], vec[0]])

        # 返回
        return verticalVec0, verticalVec1


    @staticmethod
    def mksureFirstCoeffPositive(A: float, B: float, C: float, D: float=None) -> Tuple:
        """
        确保直线方程或平面方程的首系数为正
        Args:
            A: 直线方程或平面方程系数A
            B: 直线方程或平面方程系数B
            C: 直线方程或平面方程系数C
            D: 平面方程系数D
        Returns: 确保了首系数为正后的直线方程或平面方程系数元组
        """
        # 系数容器
        coeffs = [A, B, C, D] if D is not None else [A, B, C]

        # 初始化符号因子
        sign = 1

        # 遍历每一个系数
        for coeff in coeffs:
            # 为0
            if coeff == 0:
                continue
            # 不为0
            else:
                sign = -sign if coeff < 0 else sign
                break

        # 符号因子 乘 系数
        coeffs = tuple(map(lambda x: sign * x, coeffs))

        # 返回
        return coeffs


    @classmethod
    def getCoeffsOfLineEquation(cls, pt0: typingIterable[float], pt1: typingIterable[float]) -> Tuple[float, float, float]:
        """
        由二维平面中的两个点确定直线，计算这一直线方程的系数
        Args:
            pt0: 点0
            pt1: 点1
        Returns: 直线方程的系数
        """
        # 确保输入的点为二维平面中的点
        assert more_itertools.ilen(pt0) == 2 and more_itertools.ilen(pt1) == 2

        # 确保这两点不重合
        assert not cls.areTwoPointsCoincide(pt0, pt1)

        # 获取坐标值
        x0, y0 = pt0
        x1, y1 = pt1

        # 直线方程系数
        A = y1 - y0
        B = x0 - x1
        C = x1 * y0 - x0 * y1

        # 确保首系数非负
        A, B, C = cls.mksureFirstCoeffPositive(A, B, C)

        # 首系数化为1.0
        if not cls.isEqual(A, 0.0):
            B /= A
            C /= A
            A /= A
        elif not cls.isEqual(B, 0.0):
            A /= B
            C /= B
            B /= B
        else:
            raise ValueError("[ERROR] \"A\" and \"B\" can not be 0.0 either")

        # 返回
        return A, B, C


    @classmethod
    def getCoeffsOfPlaneEquation(cls, pt0: typingIterable[float], pt1: typingIterable[float], pt2: typingIterable[float]) -> Tuple[float, float, float, float]:
        """
        由三维空间中的三个点确定平面，计算这一平面方程的系数
        Args:
            pt0: 点0
            pt1: 点1
            pt2: 点2
        Returns: 平面方程的系数
        """
        # 确保输入的点为三维空间中的点
        assert more_itertools.ilen(pt0) == 3 and more_itertools.ilen(pt1) == 3 and more_itertools.ilen(pt2) == 3

        # 确保这三点不共线
        assert not cls.areThreePointsCollinear(pt0, pt1, pt2)

        # 获取坐标值
        x0, y0, z0 = pt0
        x1, y1, z1 = pt1
        x2, y2, z2 = pt2

        # 平面方程系数
        A = (y2 - y0) * (z2 - z0) - (z1 - z0) * (y2 - y0)
        B = (x2 - x0) * (z1 - z0) - (x1 - x0) * (z2 - z0)
        C = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        D = -(A * x0 + B * y0 + C * z0)

        # 确保首系数非负
        A, B, C, D = cls.mksureFirstCoeffPositive(A, B, C, D)

        # 首系数化为1.0
        if not cls.isEqual(A, 0.0):
            B /= A
            C /= A
            A /= A
        elif not cls.isEqual(B, 0.0):
            A /= B
            C /= B
            B /= B
        elif not cls.isEqual(C, 0.0):
            A /= C
            B /= C
            C /= C
        else:
            raise ValueError("[ERROR] \"A\", \"B\" and \"C\" can not be 0.0 either")

        # 返回
        return A, B, C, D


    @classmethod
    def calcIntersectionPointOfTwoLines(cls, coeffs0: typingIterable[float], coeffs1: typingIterable[float]) -> Tuple[float, float]:
        """
        计算两个直线的交点
        Args:
            coeffs0: 直线A一般方程系数 A0, B0, C0
            coeffs1: 直线B一般方程系数 A1, B1, C1
        Returns: 直线A 和 直线B 的 交点
        """
        # 获取系数
        A0, B0, C0 = coeffs0
        A1, B1, C1 = coeffs1

        # 断言 非 不存在公共点或存在无数公共点的情况
        assert not A0 * B1 == B0 * A1

        # 解方程
        mat = np.array([[A0, B0, C0], [A1, B1, C1]])#.astype(np.float64)
        mat = sp.Matrix(mat)
        mat = mat.rref()[0].tolist()
        x = float(-mat[0][-1])
        y = float(-mat[1][-1])

        # 返回
        return x, y


    @staticmethod
    def removeRepeatedPoints(points: List[List[float]]) -> List[List[float]]: #, isLoose: bool=True
        """
        对于一组点，将其中的重复点去除 | 精简地，严格地 | 并没有涉及Tensor中的矩阵和图论的方法，也没有涉及相等阈值
        Args:
            points: 容器 存储 点 | 待去重复的点
        Returns: 去重复后的点容器
        """
        if len(points) == 0:
            return points

        st = set()
        for point in points:
            st.add(tuple(point))
        ls = list()
        for point in st:
            ls.append(list(point))

        return ls


    @staticmethod
    def pointALEPointB(pointA: List[float], pointB: List[float]) -> bool:
        """
        判断 点A 是否 小于等于 点B | 先比较横坐标是否小于，如果等于，再判断纵坐标是否小于等于
        Args:
            pointA: 点A
            pointB: 点B
        Returns: 点A 是否 小于等于 点B
        """
        if pointA[0] > pointB[0]:
            return False
        elif pointA[0] < pointB[0]:
            return True
        else:
            if pointA[1] > pointB[1]:
                return False
            else:
                return True


    @classmethod
    def mergePoints(cls, leftPoints: List[List[float]], rightPoints: List[List[float]]) -> List[List[float]]:
        """
        归并排序 中的 归并
        Args:
            leftPoints: 左 点集合
            rightPoints: 右 点集合
        Returns: 左、右 点集合 归并后的 另一集合
        """
        p, q = 0, 0
        ans = []
        while p < len(leftPoints) and q < len(rightPoints):
            if cls.pointALEPointB(leftPoints[p], rightPoints[q]):
                ans.append(leftPoints[p])
                p += 1
            else:
                ans.append(rightPoints[q])
                q += 1
        ans += leftPoints[p:]
        ans += rightPoints[q:]
        return ans


    @classmethod
    def mergeSortPoints(cls, points: List[List[float]]) -> List[List[float]]:
        """
        对 点集 进行 归并排序
        Args:
            points: 点集
        Returns: 排序后的另一个点集合
        """
        if len(points) <= 1:
            return points
        n = len(points) // 2
        leftPoints = cls.mergeSortPoints(points[:n])
        rightPoints = cls.mergeSortPoints(points[n:])
        return cls.mergePoints(leftPoints, rightPoints)


    @staticmethod
    def merge(left, right):
        p, q = 0, 0
        ans = []
        while p < len(left) and q < len(right):
            if left[p] <= right[q]:
                ans.append(left[p])
                p += 1
            else:
                ans.append(right[q])
                q += 1
        ans += left[p:]
        ans += right[q:]
        return ans


    @classmethod
    def mergeSort(cls, nums):
        if len(nums) <= 1:
            return nums
        else:
            n = len(nums) // 2
            left = cls.mergeSort(nums[:n])
            right = cls.mergeSort(nums[n:])
            return cls.merge(left, right)


    @classmethod
    def argMergeSort(cls, nums: typingIterable, rearrangingFunc: callable) -> List[int]:
        """
        获取 将nums用rearrangingFunc排序后 新数组中的各个元素来自于原数组中的哪些元素索引
        Args:
            nums: 原数组
            rearrangingFunc: 用于重新排列nums中元素的 函数或方法
        Returns: 容器 存储 新数组中的各个元素来自于原数组中的哪些元素索引
        """
        # 断言 rearrangingFunc 为 函数 或者 方法
        assert isfunction(rearrangingFunc) or ismethod(rearrangingFunc)

        # 字典 存储 (地址 | 原索引)
        dictID2Idx = dict()

        # 遍历数组 填充字典
        for idx, num in enumerate(nums):
            dictID2Idx[id(num)] = idx

        # 重新排列
        rearrangedNums = rearrangingFunc(nums)

        # 容器 存储 排序后索引
        rearrangedIdxes = []
        for rearrangedNum in rearrangedNums:
            rearrangedIdxes.append(dictID2Idx[id(rearrangedNum)])

        # 返回
        return rearrangedIdxes




if __name__ == '__main__':
    # # coordX = [0, 1, 1, 0]
    # # coordY = [0, 0, 1, 1]
    # # print(Utils.calAreaOfPolygon(coordX, coordY))
    # # print(Utils.calAreaOfPolygon(coordX, coordY, False))
    # # print()
    # #
    # # coordX = [0, 0, 1, 1]
    # # coordY = [0, 1, 1, 0]
    # # print(Utils.calAreaOfPolygon(coordX, coordY))
    # # print(Utils.calAreaOfPolygon(coordX, coordY, False))
    # # print()
    #
    # # vecA = [1, 1]
    # # vecB = [1, 0]
    # # print(Utils.cross(vecA, vecB))
    # # print()
    # #
    # # vecA = [1, 1]
    # # vecB = [-2, -2]
    # # print(Utils.cross(vecA, vecB))
    # # print()
    #
    # print(Utils.dot([1, 0, 0], [[0, 0, 0], [0, 1, 0]]))
    # print(Utils.cross([1, 0, 0], [0, 1, 0]))
    #
    #
    # # vec = np.array(
    # #     [1, 1, 1]
    # # )
    # # print(vec.shape)
    # # print(Geometry.normalize(vec))
    #
    # # pt0 = [0, 0, 0]
    # # pt1 = [1, 0, 0]
    # # pt2 = [0, 1, 0]
    # # print(Geometry.getCoeffsOfPlaneEquation(pt0, pt1, pt2))
    #
    # # vecA = np.array([1, np.sqrt(3)])
    # # vecB = np.array([-np.sqrt(3), -1])
    # # print(Geometry.calAngleBetTwoVecs(vecA, vecB, False))
    #
    # # vecA = np.array([-1, -1, -1])
    # # vecB = np.array([1, 1, 1])
    # # print(Utils.calcAngleBetTwoVecs(vecA, vecB, False))
    #
    # vec = np.array([3, 4])
    # res = Utils.getVerticalVectors(vec)
    # print(res)
    #
    # print(Utils.getPathNameExtFromPath(__file__))


    # # 凸包测试
    # # 点数据
    # points = [
    #     [0.0, 0.0],
    #     [3.5, 3.0],
    #     [3.5, 3.0],
    #     [1.0, 2.0],
    #     [4.0, -1.0],
    #     [3.0, 2.0],
    #     [3.0, 1.0],
    # ]
    # # 去重复
    # points = Utils.removeRepeatedPoints(points)
    # # 排序
    # points = Utils.mergeSortPoints(points)
    # # 打印
    # print(points)
    # # # 向量积数值
    # # crossValue = Utils.cross(
    # #     [[0.0, 0.0], [3.5, 3.0]],
    # #     [[3.0, 1.0], [1.0, 2.0]]
    # # )
    # # # 打印
    # # print(type((float)(np.linalg.norm(crossValue))))
    # # 产生凸包
    # convexHull = Utils.getConvexHull(points)
    # # 打印
    # print(convexHull)


    # # ### calcIndsATTotalNumGroupNum ##################
    # startInds, endInds = Utils.calcIndsATTotalNumGroupNum(100, 200)
    # print(startInds)
    # print(endInds)
    # # #################################################


    # ### mergeSort #####################################
    nums = [2, 1, 4, 6, 8, 5, -3, 0]
    sortedNums = Utils.mergeSort(nums)
    print(nums)
    print(sortedNums)
    print(id(nums[6]))
    print(id(sortedNums[0]))
    # ###################################################

    # ### argMergeSort ##################################
    sortedIdxes = Utils.argMergeSort(nums, Utils.mergeSort)
    print(sortedIdxes)
    # ###################################################
    pass
