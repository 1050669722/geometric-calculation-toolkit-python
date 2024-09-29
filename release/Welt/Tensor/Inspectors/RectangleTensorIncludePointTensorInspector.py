# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/25 上午9:05
# @Function:
# @Refer:

import numpy as np

# from typing import Tuple

# from Welt.Tensor.Tensors.PointTensor import PointTensor
# from Welt.Tensor.Tensors.RectangleTensor import RectangleTensor
from Welt.Tensor.Tensors.VectorTensor import VectorTensor
from Welt.Tensor.Tilers.RectangleTensorPointTensorTiler import RectangleTensorPointTensorTiler


class RectangleTensorIncludePointTensorInspector(RectangleTensorPointTensorTiler):
    """
    矩形张量 中的 每一个矩形 是否包含 点张量 中的 每一个点 的 判定器
    """
    def __init__(self, rectangles: np.ndarray, points: np.ndarray):
        """
        判定器初始化
        Args:
            rectangles: 容器 存储 矩形
            points: 容器 存储 点
        """
        # 采用父类初始化方法 初始化
        super(RectangleTensorIncludePointTensorInspector, self).__init__(rectangles, points)


    def __getNeededVecTensor(self):
        """
        获取 所需的6个向量张量
        Returns: 所需的六个向量张量
        """
        VecTensor_DA_P = VectorTensor.generateVectorTensorFromStartsAndEnds(
                            self.rectangleTensorRow.vertexesDA.Xs,
                            self.rectangleTensorRow.vertexesDA.Ys,
                            self.pointTensorColumn.Xs,
                            self.pointTensorColumn.Ys
                        )
        VecTensor_BC_P = VectorTensor.generateVectorTensorFromStartsAndEnds(
                            self.rectangleTensorRow.vertexesBC.Xs,
                            self.rectangleTensorRow.vertexesBC.Ys,
                            self.pointTensorColumn.Xs,
                            self.pointTensorColumn.Ys
                        )

        VecTensor_DA_AB = VectorTensor.generateVectorTensorFromStartsAndEnds(
                            self.rectangleTensorRow.vertexesDA.Xs,
                            self.rectangleTensorRow.vertexesDA.Ys,
                            self.rectangleTensorRow.vertexesAB.Xs,
                            self.rectangleTensorRow.vertexesAB.Ys
                        )
        VecTensor_BC_CD = VectorTensor.generateVectorTensorFromStartsAndEnds(
                            self.rectangleTensorRow.vertexesBC.Xs,
                            self.rectangleTensorRow.vertexesBC.Ys,
                            self.rectangleTensorRow.vertexesCD.Xs,
                            self.rectangleTensorRow.vertexesCD.Ys
                        )

        VecTensor_DA_CD = VectorTensor.generateVectorTensorFromStartsAndEnds(
                            self.rectangleTensorRow.vertexesDA.Xs,
                            self.rectangleTensorRow.vertexesDA.Ys,
                            self.rectangleTensorRow.vertexesCD.Xs,
                            self.rectangleTensorRow.vertexesCD.Ys
                        )
        VecTensor_BC_AB = VectorTensor.generateVectorTensorFromStartsAndEnds(
                            self.rectangleTensorRow.vertexesBC.Xs,
                            self.rectangleTensorRow.vertexesBC.Ys,
                            self.rectangleTensorRow.vertexesAB.Xs,
                            self.rectangleTensorRow.vertexesAB.Ys
                        )

        return VecTensor_DA_P, VecTensor_BC_P, VecTensor_DA_AB, VecTensor_BC_CD, VecTensor_DA_CD, VecTensor_BC_AB


    def calcIncludingMatForRTAndPT(self):
        """
        计算 矩形张量 中的 每一个矩形 是否包含 点张量 中的 每一个点 的 矩阵
        Returns: 这一矩阵
        """
        # 获取 所需的六个向量张量
        VecTensor_DA_P, VecTensor_BC_P, VecTensor_DA_AB, VecTensor_BC_CD, VecTensor_DA_CD, VecTensor_BC_AB = self.__getNeededVecTensor()

        # 判定矩阵0 判定矩阵1
        includingMat0 = VecTensor_DA_AB.crossValue(VecTensor_DA_P) * VecTensor_BC_CD.crossValue(VecTensor_BC_P) >= 0.0
        includingMat1 = VecTensor_DA_CD.crossValue(VecTensor_DA_P) * VecTensor_BC_AB.crossValue(VecTensor_BC_P) >= 0.0

        # 返回 矩形张量 中的 每一个矩形 是否包含 点张量 中的 每一个点 的 矩阵
        return (includingMat0 * includingMat1).astype(bool)


    pass




if __name__ == '__main__':
    rectangles = np.array([
        [[-4.0, 1.0], [-1.0, 1.0], [-1.0, 3.0], [-4.0, 3.0]],
        [[1.0, -1.0], [5.0, -1.0], [5.0, 2.0], [1.0, 2.0]],
    ])

    points = np.array([
        [[10.0, 10.0]],
        [[4.0, 1.0]],
        [[-3.0, 2.0]],
    ])

    rectangleTensorIncludePointTensorInspector = RectangleTensorIncludePointTensorInspector(rectangles, points)
    includingMat = rectangleTensorIncludePointTensorInspector.calcIncludingMatForRTAndPT()

    pass
