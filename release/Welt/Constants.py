# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/23 上午11:11
# @Function:
# @Refer:

# import sys


class Constants(object):
    """"""
    # 阈值
    Threshold = 1e-2

    # 调用栈最大深度
    MaxRecursionLimit = 1000000 #sys.maxsize

    # 包围盒外一点 外部 线度值
    boundingBoxExteriorLength = 100.0

    pass




if __name__ == '__main__':
    constants = Constants()
