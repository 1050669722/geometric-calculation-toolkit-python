# -*- coding: utf-8 -*-
# @Author: liuxingbo03
# @Time: 2022/07/06 11:22
# @Function：
# @Refer：

import abc
import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.Tensor import Tensor


class Executor(object):
    """
    执行器
    各种工具抽象类的抽象类
    """
    @abc.abstractmethod
    def __convertRawData(self, dataRow: np.ndarray, dataColumn: np.ndarray) -> Tuple[Tensor, Tensor]:
        """将生数据转换为Executor可用的数据"""
        pass

    pass

