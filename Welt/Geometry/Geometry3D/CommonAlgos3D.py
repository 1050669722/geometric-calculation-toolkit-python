# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   CommonAlgos3D.py
Description: 
Author:      liu
Date:        1/21/22
---------------------------------------
"""

from Welt.Geometry.GeneralAlgos import GeneralAlgos


class CommonAlgos3D(object):
    @staticmethod
    def dot(vecA, vecB):
        """
        计算两个向量的点积
        :param vecA: [x, y, z]
        :param vecB: [x, y, z]
        :return: dot product of vecA and vecB
        """
        return GeneralAlgos.dot(vecA, vecB)


    @staticmethod
    def cross(vecA, vecB):
        """
        计算两个向量的叉积
        :param vecA: [x, y, z]
        :param vecB: [x, y, z]
        :return: cross product of vecA and vecB
        """
        return GeneralAlgos.cross(vecA, vecB)
