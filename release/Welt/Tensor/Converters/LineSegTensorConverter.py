# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/06/15 下午4:22
# @Function:
# @Refer:

import numpy as np

from typing import List

from Welt.Structs.StructsDoubleParticles.LineSeg.LineSeg import LineSeg


class LineSegTensorConverter(object):
    """
    # TODO: 需要考虑应用场景
    将 图形数据结构 转换为 相应的张量
    """
    @staticmethod
    def convert(lineSegs: List[List[List[float]]]) -> np.ndarray:
        return np.array(lineSegs).astype(np.float64)
    pass




if __name__ == '__main__':
    # lineSegs = [
    #     [[0, 0], [1, 1]],
    #     [[2, 2], [3, 3]],
    #     [[4, 4], [5, 5]],
    # ]

    lineSegs = [
        LineSeg([[0, 0], [1, 1]]),
        LineSeg([[2, 2], [3, 3]]),
        LineSeg([[4, 4], [5, 5]]),
    ]

    lineSegTensorConverter = LineSegTensorConverter()

    res = LineSegTensorConverter.convert(lineSegs)

    pass
