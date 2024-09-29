# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/06 10:59
# @Function：
# @Refer：

import abc
import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.Tensor import Tensor
from Welt.Tensor.Executors.Executor import Executor


class Generator(Executor):
    """
    生成器
    各种对象生成工具的抽象类
    """
    @abc.abstractmethod
    def __convertRawData(self, dataRow: np.ndarray, dataColumn: np.ndarray) -> Tuple[Tensor, Tensor]:
        """将生数据转换为Generator可用的数据"""
        pass
    pass
