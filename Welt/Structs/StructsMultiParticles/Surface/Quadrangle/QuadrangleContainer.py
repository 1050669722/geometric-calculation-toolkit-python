# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:   QuadrangleContainer
Description: 
Author:      liu
Date:        1/24/22
---------------------------------------
"""

from typing import List

from Welt.Tools.StructTools import StructTools
from Welt.Structs.StructsSingleParticle.Particle import Particle
from Welt.Structs.StructsMultiParticles.Surface.Quadrangle.Quadrangle import Quadrangle


class QuadrangleContainer(Particle, list):
    def __init__(self, contourList: List[List[List[float]]]): #TODO: 初始化方法重载，应该有很多个重载方法
        super(QuadrangleContainer, self).__init__()

        assert hasattr(contourList, "__getitem__")

        for quadrangleData in contourList:
            self.append(Quadrangle(quadrangleData))

        QuadrangleContainer.update(self)


    def update(self):
        pass


    # def areDimensionNumsIdentical(self) -> bool:
    #     """"""
    #     dimensionNums = set()
    #     for  in self.:
    #         dimensionNums.add(edge.getDimensionNum())
    #     return len(dimensionNums) == 1


    # TODO: 很多接口方法需要更加新的处理


    pass




if __name__ == '__main__':
    pass
