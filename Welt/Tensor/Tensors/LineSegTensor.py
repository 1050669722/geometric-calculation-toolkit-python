# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/22 上午9:56
# @Function:
# @Refer:

import numpy as np

from Welt.Tensor.Tensors.Tensor import Tensor


class LineSegTensor(Tensor):
    """
    线段张量
    存储线段信息的张量
    """
    def __init__(self, lineSegs: np.ndarray):
        """
        线段张量初始化
        Args:
            lineSegs: 容器 存储 线段
        """
        # 限制数据形状
        assert len(lineSegs.shape) == 3
        assert lineSegs.shape[2] == 4

        # 超类初始化方法 初始化
        super(LineSegTensor, self).__init__(lineSegs)

        # 更新数据
        LineSegTensor.update(self)


    def update(self) -> None:
        """
        更新数据
        Returns: None
        """
        # 定义属性张量 起点横坐标 起点纵坐标 终点横坐标 终点纵坐标 张量
        self.startXs = self.entities[:, :, 0]
        self.startYs = self.entities[:, :, 1]
        self.endXs = self.entities[:, :, 2]
        self.endYs = self.entities[:, :, 3]

        # 计算 边界张量
        self.__calcBorderTensor()

        # 计算 方向向量张量
        self.__calcDirectionVecTensor()

        # 计算 线段长度
        self._calcLengths()


    def __calcBorderTensor(self) -> None:
        """
        计算 边界张量
        Returns: None
        """
        # 获取 xmins, ymins, xmaxs, ymaxs 张量
        xmins = np.min(self.entities[:, :, (0, 2)], axis=2)
        ymins = np.min(self.entities[:, :, (1, 3)], axis=2)
        xmaxs = np.max(self.entities[:, :, (0, 2)], axis=2)
        ymaxs = np.max(self.entities[:, :, (1, 3)], axis=2)

        # 在 axis=2 方向上 扩展维度
        xmins = np.expand_dims(xmins, axis=2)
        ymins = np.expand_dims(ymins, axis=2)
        xmaxs = np.expand_dims(xmaxs, axis=2)
        ymaxs = np.expand_dims(ymaxs, axis=2)

        # 在 axis=2 方向上 拼接 张量 #定义属性张量borderTensor
        self.borderTensors = np.concatenate((xmins, ymins, xmaxs, ymaxs), axis=2)

        # 定义属性张量xmins, ymins, xmaxs, ymaxs
        self.xmins = self.borderTensors[:, :, 0]
        self.ymins = self.borderTensors[:, :, 1]
        self.xmaxs = self.borderTensors[:, :, 2]
        self.ymaxs = self.borderTensors[:, :, 3]


    def __calcDirectionVecTensor(self) -> np.ndarray:
        """
        计算 方向向量张量
        Returns: None
        """
        # 计算 方向向量张量
        self.directionVecTensor = np.concatenate(
            (
                np.expand_dims(self.endXs - self.startXs, axis=2),
                np.expand_dims(self.endYs - self.startYs, axis=2)
             ), axis=2
        )

        # 返回
        return self.directionVecTensor


    def _calcLengths(self) -> np.ndarray:
        """
        计算 长度
        Returns: 长度张量
        """
        # 计算 长度 #TODO: 对于这些容易造成极大数值的运算，要么使用更大容量的数据类型，要么改用不容易产生更大数值的算法
        # self.lengths = np.sqrt(
        #     np.power(self.startXs - self.endXs, 2).astype(np.float64) + \
        #     np.power(self.startYs - self.endYs, 2).astype(np.float64)
        # )

        # 这一做法更加安全
        self.lengths = np.linalg.norm(self.directionVecTensor, axis=2, ord=2)

        # 返回
        return self.lengths

    pass




if __name__ == '__main__':
    from Welt.Tensor.Converters.LineSegTensorConverter import LineSegTensorConverter

    # 线段
    lineSegs = [
        [[0, 0], [1.0, 0.0]],
        [[-1, 0], [-1, -1]],
        [[-2, -2], [-3, -1]],
    ]

    # 转换
    lineSegs = LineSegTensorConverter.convert(lineSegs).reshape((len(lineSegs), -1, 4))

    # 实例化
    lineSegTensor = LineSegTensor(lineSegs)

    # 计算 打印
    print(lineSegTensor.lengths)

    pass
