# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/06/12 上午10:33
# @Function:
# @Refer:

import numpy as np

from Welt.Tensor.Inspectors.PointTensorAtLineSegTensorInspector import PointTensorAtLineSegTensorInspector


class LineSegTensorIncludeLineSegTensorInspector(object):
    """
    线段张量 中的 线段 与 线段张量 中的 线段 是否重合 的 判定器
    """
    def __init__(self, lineSegsA: np.ndarray, lineSegsB: np.ndarray):
        """
        判定器初始化
        Args:
            lineSegsA: 容器A 存储 线段
            lineSegsB: 容器B 存储 线段
        """
        # 线段张量A
        self.lineSegsA = lineSegsA.astype(np.float64)
        
        # 线段张量B 中的 线段 的 起点
        self.startPointsB = np.expand_dims(lineSegsB[:, 0, :], axis=1).astype(np.float64)
        
        # 线段张量B 中的 线段 的 终点
        self.endPointsB = np.expand_dims(lineSegsB[:, 1, :], axis=1).astype(np.float64)


    def calcIncludingMatForTwoLineSegTensors(self) -> np.ndarray:
        """
        计算 一个线段张量 中的 线段 是否 包含 另一个线段张量 中的 线段 的 矩阵
        Args:
            isLoose: 如真，则在一定阈值内认为包含
        Returns: 这一矩阵
        """
        # 实例化 点张量 是否 位于 线段张量 判定器
        startPointTensorAtLineSegTensorInspector = PointTensorAtLineSegTensorInspector(self.startPointsB, self.lineSegsA)
        endPointTensorAtLineSegTensorInspector = PointTensorAtLineSegTensorInspector(self.endPointsB, self.lineSegsA)

        # 线段张量B 中的 线段 的 起点 与 终点 都位于 线段张量A 中的 线段
        isIncludedMat = startPointTensorAtLineSegTensorInspector.calcIsIncludedMatForPTAndLT() * endPointTensorAtLineSegTensorInspector.calcIsIncludedMatForPTAndLT()

        # 线段张量A 中的 线段 是否 包含 线段张量B 中的 线段
        return isIncludedMat.transpose() # => includingMat

    pass




if __name__ == '__main__':
    # lineSegsA = np.array([
    #     [[0.0, 0.0], [1.0, 1.0]],
    #     [[-1.0, 0.5], [2.0, 0.5]],
    #     [[1.0, 3.0], [2.0, 2.0]],
    #     [[1.5, 2.7], [2.0, 3.0]],
    # ])
    #
    # lineSegsB = np.array([
    #     [[0.0, 0.0], [1.0, 1.0]],
    #     [[1.6, 0.6], [1.8, 0.4]],
    #     [[1.4, 2.5], [1.6, 2.3]],
    #     [[1.0, 3.0], [1.2, 2.8]],
    # ])

    # lineSegsA = np.array([
    #     [[2785369.855655348, 3295860.258957713], [2785369.855655348, 3505342.372608559]],
    #     [[2785369.855655348, 3295860.258957713], [2785369.8556553475, 3450918.290175232]],
    #     [[2785369.855655348, 3295860.258957713], [2862944.8556551756, 3295860.258957715]],
    # ])
    # 
    # lineSegsB = np.array([
    #     [[2785369.855655348, 3295860.258957713], [2785369.855655348, 3505342.372608559]],
    #     [[2785369.855655348, 3295860.258957713], [2785369.8556553475, 3450918.290175232]],
    #     [[2785369.855655348, 3295860.258957713], [2862944.8556551756, 3295860.258957715]],
    # ])
    
    lineSegsA = np.array([
        [[2716894.8556551985, 3521042.3726085415], [2716894.8556551877, 3497854.514704713]],
        [[2716894.8556551985, 3521042.3726085415], [2716894.855655199, 3505342.3726085587]],
        [[2716894.8556551985, 3521042.3726085415], [2814645.135290712, 3521042.372608542]],
    ])

    lineSegsB = np.array([
        [[2716894.8556551985, 3521042.3726085415], [2716894.8556551877, 3497854.514704713]],
        [[2716894.8556551985, 3521042.3726085415], [2716894.855655199, 3505342.3726085587]],
        [[2716894.8556551985, 3521042.3726085415], [2814645.135290712, 3521042.372608542]],
    ])
    
    lineSegTensorIncludeLineSegTensorInspector = LineSegTensorIncludeLineSegTensorInspector(lineSegsA, lineSegsB)
    includingMat = lineSegTensorIncludeLineSegTensorInspector.calcIncludingMatForTwoLineSegTensors()

    pass
