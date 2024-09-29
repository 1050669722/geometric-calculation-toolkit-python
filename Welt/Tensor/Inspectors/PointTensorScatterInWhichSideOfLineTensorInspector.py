# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/06 10:21
# @Function：
# @Refer：

import numpy as np

from Welt.Tensor.Tilers.PointTensorLineSegTensorTiler import PointTensorLineSegTensorTiler
# from Welt.Tensor.Tensors.VectorTensor import VectorTensor
from Welt.Tensor.Generators.SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator import SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator
from Welt.Tensor.Converters.PointTensorConverter import PointTensorConverter
from Welt.Tensor.Converters.LineSegTensorConverter import LineSegTensorConverter


class PointTensorScatterInWhichSideOfLineTensorInspector(PointTensorLineSegTensorTiler):
    """
    点张量 中的 每一个点 散落在 线段张量 中的 每一个线段所在直线 的 哪一侧 的 判定器
    1: 正侧 --- spVec x lineVec > 0
    -1: 负侧 --- spVec x lineVec < 0
    0: 其上 --- spVec x lineVec = 0
    """
    def __init__(self, points: np.ndarray, lineSegs: np.ndarray):
        """
        判定器初始化
        Args:
            points: 容器 存储 点
            lineSegs: 容器 存储 线段
        """
        # 采用父类初始化方法 初始化
        super(PointTensorScatterInWhichSideOfLineTensorInspector, self).__init__(points, lineSegs)

        # 定义属性
        self.points, self.lineSegs = points.astype(np.float64), lineSegs.astype(np.float64)


    def calcWhichSideMatForPTAndLT(self) -> np.ndarray:
        """
        计算 点 散落在 线段所在直线的 哪一侧的 矩阵
        Returns: 点 散落在 线段所在直线的 哪一侧的 矩阵
        """
        # 生成 <从 线段起点 到 点 的向量张量> 以及 <线段 变成 的向量张量>
        spVectorTensor, lineSegVectorTensor = SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator(self.points, self.lineSegs).generateSpVectorTensorAndLineSegVectorTensor()

        # 叉积
        crossProd = spVectorTensor.crossValue(lineSegVectorTensor)

        # 散落于哪一侧 矩阵
        whichSideMat = crossProd
        whichSideMat[whichSideMat > 0.0] = 1
        whichSideMat[whichSideMat < 0.0] = -1

        # 返回
        return whichSideMat

    pass




if __name__ == '__main__':
    # 点
    points = [
        [-1, 1],
        [2, 1],
        [1, 3],
        [-1, 3],
        [3, -1],
    ]

    # 线段
    lineSegs = [
        [[-2, 2], [0, 4]],
        [[0, 2], [2, 0]],
    ]

    # 转换
    points = PointTensorConverter.convert(points)
    lineSegs = LineSegTensorConverter.convert(lineSegs)

    # 实例化
    pointTensorScatterInWhichSideOfLineTensorInspector = PointTensorScatterInWhichSideOfLineTensorInspector(points, lineSegs)

    # 判定
    whichSideMat = pointTensorScatterInWhichSideOfLineTensorInspector.calcWhichSideMatForPTAndLT()

    # 获取 行、列号
    rows, cols = np.where(whichSideMat > 0)

    pass
