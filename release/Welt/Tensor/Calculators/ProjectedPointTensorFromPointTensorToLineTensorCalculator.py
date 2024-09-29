# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/05 15:36
# @Function：
# @Refer：

import numpy as np

from copy import deepcopy

from Welt.Utils import Utils
from Welt.Tensor.Tilers.PointTensorLineSegTensorTiler import PointTensorLineSegTensorTiler
# from Welt.Tensor.Tensors.VectorTensor import VectorTensor
from Welt.Tensor.Generators.SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator import SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator


class ProjectedPointTensorFromPointTensorToLineTensorCalculator(PointTensorLineSegTensorTiler):
    """
    点张量 中的 每一个点 到 线段张量 中的 每一个线段所在直线 的 投影点 的 计算器
    """
    def __init__(self, points: np.ndarray, lineSegs: np.ndarray):
        """
        计算器初始化
        Args:
            points: 容器 存储 点
            lineSegs: 容器 存储 线段
        """
        # 采用父类初始化方法 初始化
        super(ProjectedPointTensorFromPointTensorToLineTensorCalculator, self).__init__(points, lineSegs)

        # 定义属性
        self.points, self.lineSegs = points, lineSegs


    def calcProjectedPointTensorForPTAndLT(self) -> np.ndarray:
        """
        点张量 中的 每一个点 到 线段张量 中的 每一个线段所在直线 的 投影点
        Returns: 投影点张量
        """
        # 生成 <从 线段起点 到 点 的向量张量> 以及 <线段 变成 的向量张量>
        spVectorTensor, lineSegVectorTensor = SpVectorTensorAndLineSegVectorTensorFromPointsAndLineSegsGenerator(self.points, self.lineSegs).generateSpVectorTensorAndLineSegVectorTensor()

        # 模长矩阵
        lineSegVectorNormTensor = deepcopy(lineSegVectorTensor.norm)

        # 为了下面的除法而做的处理
        lineSegVectorNormTensor[lineSegVectorNormTensor < 1e-4 * Utils.Threshold] = 1e-4 * Utils.Threshold

        # 投影长度 = 点积 / 线段向量模长
        projectedLengthTensor = np.divide(
            spVectorTensor.dot(lineSegVectorTensor),
            lineSegVectorNormTensor #TODO: 尽管现在问题不是出在除法上面，但是还是应该尽量避免除法，考虑将接下来的做法换成 平移 #但是平移的时候用到的平移向量也是会涉及到除法的
        )

        # 投影长度 占 线段向量模长的 比例
        scaleFactorTensor = np.divide(projectedLengthTensor, lineSegVectorNormTensor) #TODO: 尽管现在问题不是出在除法上面，但是还是应该尽量避免除法，考虑将接下来的做法换成 平移 #但是平移的时候用到的平移向量也是会涉及到除法的
        scaleFactorTensor = np.expand_dims(scaleFactorTensor, axis=2)
        scaleFactorTensor = np.concatenate((scaleFactorTensor, scaleFactorTensor), axis=2)

        # 投影点的横坐标张量 投影点的纵坐标张量
        projectedPointXTensor = self.lineSegTensorColumn.startXs + lineSegVectorTensor.scalarValue(scaleFactorTensor)[:, :, 0]
        projectedPointYTensor = self.lineSegTensorColumn.startYs + lineSegVectorTensor.scalarValue(scaleFactorTensor)[:, :, 1]
        projectedPointXTensor = np.expand_dims(projectedPointXTensor, axis=2)
        projectedPointYTensor = np.expand_dims(projectedPointYTensor, axis=2)

        # 投影点张量
        projectedPointTensor = np.concatenate((projectedPointXTensor, projectedPointYTensor), axis=2) #(n, m, 2) #n个点, m条线段

        # 返回
        return projectedPointTensor




if __name__ == '__main__':
    from Welt.Tensor.Converters.PointTensorConverter import PointTensorConverter
    from Welt.Tensor.Converters.LineSegTensorConverter import LineSegTensorConverter

    # 点
    points = [
        [-1, 1],
        [2, 1],
        [1, 3],
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
    projectedPointTensorFromPointTensorToLineTensorCalculator = ProjectedPointTensorFromPointTensorToLineTensorCalculator(points, lineSegs)

    # 计算
    projectedPointTensor = projectedPointTensorFromPointTensorToLineTensorCalculator.calcProjectedPointTensorForPTAndLT()

    pass
