# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/05 10:45
# @Function：
# @Refer：

import abc
import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.Tensor import Tensor
from Welt.Tensor.Executors.Executor import Executor


class Calculator(Executor):
    """
    计算器
    各种矩阵化计算工具的抽象类
    """
    @abc.abstractmethod
    def __convertRawData(self, dataRow: np.ndarray, dataColumn: np.ndarray) -> Tuple[Tensor, Tensor]:
        """将生数据转换为Calculator可用的数据"""
        pass

    pass




if __name__ == '__main__':
    calculator = Calculator()
