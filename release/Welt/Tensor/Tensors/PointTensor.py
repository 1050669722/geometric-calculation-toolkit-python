# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/25 上午8:21
# @Function:
# @Refer:

import numpy as np

from Welt.Tensor.Tensors.Tensor import Tensor


class PointTensor(Tensor):
    """
    点张量
    存储点信息的张量
    """
    def __init__(self, points: np.ndarray):
        """
        点张量初始化
        Args:
            points: 容器 存储 点
        """
        # 限制数据形状
        assert len(points.shape) == 3
        assert points.shape[2] == 2

        # 超类初始化方法 初始化
        super(PointTensor, self).__init__(points)

        # 更新数据
        PointTensor.update(self)


    def update(self) -> None:
        """
        更新数据
        Returns: None
        """
        # 定义属性张量 横坐标 纵坐标 张量
        self.Xs = self.entities[:, :, 0]
        self.Ys = self.entities[:, :, 1]

    pass




if __name__ == '__main__':
    pointTensor = PointTensor()
