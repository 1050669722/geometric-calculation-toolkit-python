# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/31 上午10:23
# @Function:
# @Refer:

import abc
import numpy as np

from typing import Tuple

from Welt.Tensor.Tensors.Tensor import Tensor
from Welt.Tensor.Executors.Executor import Executor


class Inspector(Executor):
    """
    审查器
    各种矩阵化审查工具的抽象类
    """
    @abc.abstractmethod
    def __tileRawData(self, dataRow: np.ndarray, dataColumn: np.ndarray) -> Tuple[Tensor, Tensor]:
        """将生数据转换为Inspector可用的数据"""
        pass
    pass




if __name__ == '__main__':
    inspector = Inspector()
