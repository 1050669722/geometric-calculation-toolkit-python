# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/25 上午8:30
# @Function:
# @Refer:

import numpy as np

from Welt.Tensor.Tensors.Tensor import Tensor
from Welt.Tensor.Tensors.PointTensor import PointTensor
from Welt.Tensor.Tensors.LineSegTensor import LineSegTensor


class RectangleTensor(Tensor):
    """
    矩形张量
    存储矩形信息的张量
    """
    def __init__(self, vertexes: np.ndarray):
        """
        矩形张量初始化
        Args:
            vertexes: 容器 存储 矩形
        """
        # 限制数据形状
        assert len(vertexes.shape) == 3
        assert vertexes.shape[2] == 8

        # 超类初始化方法 初始化
        super(RectangleTensor, self).__init__(vertexes)

        # TODO: 断言 每一个四边形都是矩形
        assert self.__checkRects()

        # 更新数据
        RectangleTensor.update(self)


    def __checkRects(self):
        # TODO: 判定 四边形是矩形的 矩阵化 实现
        return True
        pass


    def update(self) -> None:
        """
        更新数据
        Returns:
        """
        # 定义属性张量 边A 边B 边C 边D 张量
        self.edgesALineSeg = LineSegTensor(self.entities[:, :, 0:4])
        self.edgesBLineSeg = LineSegTensor(self.entities[:, :, 2:6])
        self.edgesCLineSeg = LineSegTensor(self.entities[:, :, 4:8])
        self.edgesDLineSeg = LineSegTensor(np.concatenate((self.entities[:, :, 6:8], self.entities[:, :, 0:2]), axis=2))

        # 定义属性张量 顶点DA 顶点AB 顶点BC 顶点CD 张量
        self.vertexesDA = PointTensor(self.entities[:, :, 0:2])
        self.vertexesAB = PointTensor(self.entities[:, :, 2:4])
        self.vertexesBC = PointTensor(self.entities[:, :, 4:6])
        self.vertexesCD = PointTensor(self.entities[:, :, 6:8])

        pass


    def __calcBorderTensor(self) -> None:
        """"""
        pass


    pass




if __name__ == '__main__':
    rectangleTensor = RectangleTensor()
