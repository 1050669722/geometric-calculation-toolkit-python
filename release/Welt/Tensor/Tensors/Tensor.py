# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/22 上午9:53
# @Function:
# @Refer:

import numpy as np


class Tensor(object):
    """
    图形张量
    存储图形信息的张量
    """
    def __init__(self, entities: np.ndarray):
        """
        图形张量初始化
        Args:
            entities: 容器 存储 图形
        """
        # 图形数量
        self.num = len(entities)

        # 图形容器
        self.entities = entities.astype(np.float64)

        # 更新数据
        Tensor.update(self)


    def update(self) -> None:
        """
        更新数据
        Returns: None
        """
        pass

    pass




if __name__ == '__main__':
    tensor = Tensor()
