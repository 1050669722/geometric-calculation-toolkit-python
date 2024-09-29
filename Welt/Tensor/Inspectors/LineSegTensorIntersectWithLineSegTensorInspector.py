# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/22 上午10:04
# @Function:
# @Refer:

import numpy as np

from Welt.Utils import Utils
from Welt.Tensor.Tensors.LineSegTensor import LineSegTensor
from Welt.Tensor.Tensors.VectorTensor import VectorTensor
from Welt.Tensor.Tilers.LineSegTensorLineSegTensorTiler import LineSegTensorLineSegTensorTiler


class LineSegTensorIntersectWithLineSegTensorInspector(LineSegTensorLineSegTensorTiler):
    """
    线段张量 中的 线段 与 线段张量 中的 线段 是否存在公共点 的 判定器
    """
    def __init__(self, lineSegsA: np.ndarray, lineSegsB: np.ndarray):
        """
        判定器初始化
        Args:
            lineSegsA: 容器A 存储 线段
            lineSegsB: 容器B 存储 线段
        """
        # 采用父类初始化方法 初始化
        super(LineSegTensorIntersectWithLineSegTensorInspector, self).__init__(lineSegsA, lineSegsB)


    def __calcIsOverlapMatForTwoLineSegTensors(self) -> bool:
        """
        计算 是否重叠矩阵 这一矩阵记录着 两个线段张量中的线段的对角线矩形两两之间是否存在重叠部分
        Returns: 是否存在重叠部分的矩阵
        """
        # 矩阵 存储 两两判断的结果 以线段为对角线的矩形区域是否存在重叠部分
        overlapMat = (self.lineSegTensorRow.xmaxs >= self.lineSegTensorColumn.xmins) * \
                     (self.lineSegTensorColumn.xmaxs >= self.lineSegTensorRow.xmins) * \
                     (self.lineSegTensorRow.ymaxs >= self.lineSegTensorColumn.ymins) * \
                     (self.lineSegTensorColumn.ymaxs >= self.lineSegTensorRow.ymins)

        # 返回
        return overlapMat


    def __calcProdMatOfTwoCrossProdsMatInStraddleForTwoLineSegTensors(self, lineSegTensorA: LineSegTensor, lineSegTensorB: LineSegTensor, isLoose: bool=True) -> np.ndarray:
        """
        计算 两个线段张量 在跨立检验中 的 两个张量积张量 的 乘积张量
        Args:
            lineSegTensorA: 线段张量A
            lineSegTensorB: 线段张量B
        Returns: 这一乘积张量
        """
        # 限制形状
        assert lineSegTensorA.entities.shape == lineSegTensorB.entities.shape

        # 向量
        sAeA = VectorTensor.generateVectorTensorFromStartsAndEnds(lineSegTensorA.startXs, lineSegTensorA.startYs, lineSegTensorA.endXs, lineSegTensorA.endYs)
        sAsB = VectorTensor.generateVectorTensorFromStartsAndEnds(lineSegTensorA.startXs, lineSegTensorA.startYs, lineSegTensorB.startXs, lineSegTensorB.startYs)
        sAeB = VectorTensor.generateVectorTensorFromStartsAndEnds(lineSegTensorA.startXs, lineSegTensorA.startYs, lineSegTensorB.endXs, lineSegTensorB.endYs)

        # 向量叉积数值
        sAsB_x_sAeA = sAsB.crossValue(sAeA)
        sAeB_x_sAeA = sAeB.crossValue(sAeA)

        # 小于阈值者，赋值为0.0
        if isLoose:
            sAsB_x_sAeA[sAsB_x_sAeA < Utils.Threshold] = 0.0
            sAeB_x_sAeA[sAeB_x_sAeA < Utils.Threshold] = 0.0

        # 返回 向量叉积数值 的 乘积
        return sAsB_x_sAeA * sAeB_x_sAeA


    def calcIntersectingMatForTwoLineSegTensors(self) -> np.ndarray:
        """
        计算 两个线段张量 中的 线段 两两之间 是否 存在 公共点 的 矩阵
        Returns: 这一矩阵
        """
        # 快速排斥
        overlapMat = self.__calcIsOverlapMatForTwoLineSegTensors()

        # 相互跨立检验
        prodTensorOfTwoCrossProdsRowColumn = self.__calcProdMatOfTwoCrossProdsMatInStraddleForTwoLineSegTensors(self.lineSegTensorRow, self.lineSegTensorColumn)
        prodTensorOfTwoCrossProdsColumnRow = self.__calcProdMatOfTwoCrossProdsMatInStraddleForTwoLineSegTensors(self.lineSegTensorColumn, self.lineSegTensorRow)

        # 返回 线段两两之间是否存在公共点的矩阵
        return (overlapMat * (prodTensorOfTwoCrossProdsRowColumn <= 0.0) * (prodTensorOfTwoCrossProdsColumnRow <= 0.0)).astype(bool)




    pass




if __name__ == '__main__':
    lineSegsA = np.array([
        [[0.0, 0.0], [1.0, 1.0]],
        [[-1.0, 0.5], [2.0, 0.5]],
        [[1.0, 3.0], [2.0, 2.0]],
        [[1.5, 2.7], [2.0, 3.0]],
    ])

    lineSegsB = np.array([
        [[0.0, 0.0], [1.0, 1.0]],
        [[-1.0, 0.5], [2.0, 0.5]],
        [[1.0, 3.0], [2.0, 2.0]],
        [[1.5, 2.7], [2.0, 3.0]],
    ])

    lineSegTensorIntersectionInspector = LineSegTensorIntersectWithLineSegTensorInspector(lineSegsA, lineSegsB)
    intersectingMat = lineSegTensorIntersectionInspector.calcIntersectingMatForTwoLineSegTensors()

    pass
