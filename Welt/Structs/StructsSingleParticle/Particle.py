# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   Particle
Description: 
Author:      liu
Date:        1/19/22
---------------------------------------
"""

import sys
import abc


def abstractproperty(func):
    if sys.version_info > (3, 3):
        return property(abc.abstractmethod(func))
    else:
        return abc.abstractproperty(func)


class Particle(abc.ABC):
    @abc.abstractmethod
    def update(self):
        """更新数据，添加具有意义的属性名，较低一级类实例化，属性赋值"""


    @abstractproperty
    def activeDim(self):
        """定义抽象成员属性"""
        return "Should never get here"


    @activeDim.setter
    def activeDim(self, dim):
        """setter函数"""
        return


    @abc.abstractmethod
    def collapseIntoPhasePlane(self, dim):
        """从3D空间坍缩至2D相平面"""


    @abc.abstractmethod
    def expandFromPhasePlane(self, dim):
        """从2D相平面膨胀至3D空间"""
