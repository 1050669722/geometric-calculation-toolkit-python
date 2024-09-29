# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/06/17 下午5:09
# @Function:
# @Refer:

import numpy as np

from typing import List

from Welt.Structs.StructsSingleParticle.Point.Point import Point


class PointTensorConverter(object):
    """
    将 图形数据结构 转换为 相应的张量 | (n, 1, 2)
    """
    @staticmethod
    def convert(points: List[List[float]]) -> np.ndarray:
        return np.expand_dims(np.array(points), axis=1).astype(np.float64)
    pass




if __name__ == '__main__':
    points = [
        [1, 1],
        [2, 2],
        [3, 3]
    ]

    res = PointTensorConverter.convert(points)

    pass
