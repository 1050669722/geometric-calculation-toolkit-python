# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/25 10:44
# @Function：
# @Refer：

import abc
import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.Tensor import Tensor
from Welt.Tensor.Executors.Executor import Executor


class Transformer(Executor):
    """
    变换器
    各种矩阵化变换工具的抽象类
    """
    @abc.abstractmethod
    def __tileRawData(self, dataRow: np.ndarray, dataColumn: np.ndarray) -> Tuple[Tensor, Tensor]:
        """将生数据转换为Transformer可用的数据"""
        pass
    pass




if __name__ == '__main__':
    transformer = Transformer()
