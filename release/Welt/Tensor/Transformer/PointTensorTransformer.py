# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/25 10:54
# @Function：
# @Refer：

import numpy as np

from typing import Tuple
from typing import List

from Welt.Tensor.Tensors.Tensor import Tensor
from Welt.Tensor.Tensors.PointTensor import PointTensor
from Welt.Tensor.Tensors.VectorTensor import VectorTensor
from Welt.Tensor.Executors.Transformer import Transformer
from Welt.Tensor.Tilers.PointTensorLineSegTensorTiler import PointTensorLineSegTensorTiler


class PointTensorTransformer(PointTensorLineSegTensorTiler):
    """
    点 变换器 | 仿射变换
    """
    def __init__(self, points: np.ndarray, vectors: np.ndarray, convertToPointList: bool=True, distinguishedByVecsForPointList: bool=True):
        """
        初始化 变换器
        Args:
            points: 点数组 | (m, 1, 2)
            vectors: 向量数组 | (m, 2, 2)
        """
        # 采用父类初始化方法 初始化
        super(PointTensorTransformer, self).__init__(points, vectors)

        # 定义属性
        self.vectorTensorColumn = VectorTensor(self.lineSegTensorColumn.entities)
        self.points, self.lineSegs, self.convertToPointList, self.distinguishedByVecsForPointList = points, vectors, convertToPointList, distinguishedByVecsForPointList


    def __convertForOutput(self) -> List[List[float]]:
        """
        将 点数组 转换为 点列表
        Returns: 完成转换的 点列表
        """
        # 限制形状
        assert len(self.pointTensorRow.entities.shape) == 3
        assert self.pointTensorRow.entities.shape[2] == 2

        # 点集列表
        pointsList = []

        # 遍历 每一行
        for pointArrayRow in self.pointTensorRow.entities:
            # 如果按照平移向量区分这些点
            if self.distinguishedByVecsForPointList:
                pointsList.append(pointArrayRow.tolist())
            # 否则
            else:
                pointsList.extend(pointArrayRow.tolist())

        # # 点集列表获取值
        # # pointsList = self.pointTensorRow.entities[:, 0, :].tolist()
        # pointsList = self.pointTensorRow.entities[np.diag(np.ones(len(self.pointTensorRow.entities))) != 0].tolist()

        # 返回
        return pointsList


    def translate2D(self):# -> List[List[float]]: #-> np.ndarray #
        """
        将 点数组 进行 平移变换 并 返回 点列表 #二维情况下
        Returns: 经过 平移变换后的 点列表
        """
        # 平移变换
        self.pointTensorRow.Xs += self.vectorTensorColumn.Xs
        self.pointTensorRow.Ys += self.vectorTensorColumn.Ys

        # 返回
        if self.convertToPointList:
            return self.__convertForOutput()
        else:
            return self.pointTensorRow.entities

    pass




if __name__ == '__main__':
    from Welt.Tensor.Converters.PointTensorConverter import PointTensorConverter
    from Welt.Tensor.Converters.LineSegTensorConverter import LineSegTensorConverter

    # 点
    points = [
        [-1, 1],
        [2, 1],
        [1, 3],
    ]

    # 向量
    vectors = [
        [[0, 0], [10, 10]], #(10, 10)
        # [[-1, -2], [9, 10]], #(10, 12)
        # [[-3, 4], [-4, 3]], #(-1, -1)
    ]

    # 转化
    arrayPoints = PointTensorConverter.convert(points)
    arrayVectors = LineSegTensorConverter.convert(vectors)

    # 实例化
    pointTensorTransformer = PointTensorTransformer(arrayPoints, arrayVectors, False, False)

    # 平移变换
    transformedPoints = pointTensorTransformer.translate2D()

    # 打印
    print(transformedPoints)
