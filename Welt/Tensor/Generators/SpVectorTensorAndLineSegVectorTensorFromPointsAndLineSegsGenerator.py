# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/06 10:56
# @Function：
# @Refer：

import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.VectorTensor import VectorTensor
from Welt.Tensor.Tilers.PointTensorLineSegTensorTiler import PointTensorLineSegTensorTiler
from Welt.Tensor.Converters.PointTensorConverter import PointTensorConverter
from Welt.Tensor.Converters.LineSegTensorConverter import LineSegTensorConverter


class SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator(PointTensorLineSegTensorTiler):
    """
    点张量 中的 每一个点 到 线段张量 中的 每一个线段所在直线 的 SpVectorTensor 和 LineSegVector 的 生成器
    """
    def __init__(self, points: np.ndarray, lineSegs: np.ndarray):
        """
        生成器初始化
        Args:
            points: 容器 存储 点
            lineSegs: 容器 存储 线段
        """
        # 采用父类初始化方法 初始化
        super(SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator, self).__init__(points, lineSegs)


    def generateSpVectorTensorAndLineSegVectorTensor(self) -> Tuple[VectorTensor, VectorTensor]:
        """
        生成 SpVectorTensor 和 LineSegVectorTensor
        Returns: 元组 (向量张量, 向量张量)
        """
        # 从 线段起点 到 点 的向量张量
        spVectorTensor = VectorTensor.generateVectorTensorFromStartsAndEnds(
            self.lineSegTensorColumn.startXs,
            self.lineSegTensorColumn.startYs,
            self.pointTensorRow.Xs,
            self.pointTensorRow.Ys,
        )

        # 断言 spVectorTensor的类型 为 VectorTensor
        assert isinstance(spVectorTensor, VectorTensor)

        # 线段 变成 的向量张量
        lineSegVectorTensor = VectorTensor(self.lineSegTensorColumn.entities)

        # 返回
        return spVectorTensor, lineSegVectorTensor




if __name__ == '__main__':
    # 点
    points = [
        [0, 0],
        [1, 1],
        [2, 1],
        [2, 0],
    ]

    # 线段
    lineSegs = [
        [[-1, 0], [0, 1]],
        [[-2, 0], [2, 2]],
        [[-3, -1], [-4, -4]],
    ]

    # 转换
    arrayPoints = PointTensorConverter.convert(points)
    arrayLineSegs = LineSegTensorConverter.convert(lineSegs)

    # 实例化
    spVectorTensorAndLineSegVectorTensorFromPointsTensorAndLineSegTensorGenerator = SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator(arrayPoints, arrayLineSegs)

    # 生成 向量张量
    spVectorTensor, lineSegVectorTensor = spVectorTensorAndLineSegVectorTensorFromPointsTensorAndLineSegTensorGenerator.generateSpVectorTensorAndLineSegVectorTensor()

    pass
