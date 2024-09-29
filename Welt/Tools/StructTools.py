# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   StructTools
Description: 
Author:      liu
Date:        1/19/22
---------------------------------------
"""

import sys

if sys.version_info < (3, 8):
    from collections import Iterable
else:
    from collections.abc import Iterable


class StructTools(object):
    @staticmethod
    def modifyActiveDim(oriFunc):
        """
        自动调节维度拨片 --- _activeDim
        for collapseIntoPhasePlane and expandFromPhasePlane
        :param oriFunc:
        :return:
        """
        def dstFunc(*args, **kargs):
            args[0]._activeDim = args[1]
            oriFunc(*args, **kargs)
            args[0]._activeDim = None
        return dstFunc


    @staticmethod
    def runAt2DSpace(oriFunc):
        """
        限制方法仅运行在2D空间中
        :param oriFunc:
        :return:
        """
        def dstFunc(*args, **kargs):
            # self = args[0]
            # if len(self) == 0:
            #     raise ValueError("[ERROR] Invalid length of \"self\": {}".format(len(self)))
            # elif not isinstance(self[0], Iterable):
            #     flag = (len(self) == 2)
            # elif isinstance(self[0], Iterable):
            #     flag = (len(self[0]) == 2)
            # else:
            #     raise TypeError("[ERROR] Invalid type of \"self\": {}".format(type(self)))

            self = args
            tmp = args
            while isinstance(tmp, Iterable):
                self = tmp
                tmp = tmp[0]
            flag = (len(self) == 2)

            if flag:
                results = oriFunc(*args, **kargs)
                return results
            else:
                raise RuntimeError("[ERROR] The function \"{}\" must run at 2D space".format(oriFunc.__name__))

        return dstFunc


    @staticmethod
    def runAt3DSpace(oriFunc):
        """
        限制方法仅运行在2D空间中
        :param oriFunc:
        :return:
        """

        def dstFunc(*args, **kargs):
            # self = args[0]
            # if len(self) == 0:
            #     raise ValueError("[ERROR] Invalid length of \"self\": {}".format(len(self)))
            # elif not isinstance(self[0], Iterable):
            #     flag = (len(self) == 3)
            # elif isinstance(self[0], Iterable):
            #     flag = (len(self[0]) == 3)
            # else:
            #     raise TypeError("[ERROR] Invalid type of \"self\": {}".format(type(self)))

            self = args
            tmp = args
            while isinstance(tmp, Iterable):
                self = tmp
                tmp = tmp[0]
            flag = (len(self) == 3)

            if flag:
                results = oriFunc(*args, **kargs)
                return results
            else:
                raise RuntimeError("[ERROR] The function \"{}\" must run at 3D space".format(oriFunc.__name__))

        return dstFunc
