# -*- coding: utf-8 -*-
# @Author: liu
# @Time: 2022/05/10 上午8:48
# @Function：
# @Refer：

# import numpy as np

from typing import List

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import preprocessing

from Welt.Structs.StructsSingleParticle.Point.Point import Point


# 常量
dbscan = "dbscan"
kmeans = "kmeans"

# epsFloor = 0.001
# epsCeil = 1
# epsStride = 0.05

# minSamplesFloor = 2
# minSamplesCeil = 10


class Cluster(object):
    def __init__(self, method=dbscan, eps: float=None, minSamples: int=None, numClusters: int=None) -> None:
        """
        针对点（Point） 聚类器 包含dbscan, kmeans方法 以及 一些简易参数
        Args:
            method: 方法名
            eps: dbscan的 𝜖-邻域距离阈值
            minSamples: dbscan的 样本点要成为核心对象所需要的𝜖-邻域的样本数阈值
            numClusters: kmeans的 聚类类别数量
        """
        # 断言 聚类方法限定
        assert method.lower() in {dbscan, kmeans}
        # 断言 特定方法参数限定
        if method.lower() == dbscan:
            assert not ((eps is None) or (minSamples is None))
        # 断言 特定方法参数限定
        if method.lower() == kmeans:
            assert not (numClusters is None)

        # 参数记录
        self.method = method
        self.eps = eps
        self.minSamples = minSamples
        self.numClusters = numClusters

        # 初始化聚类器
        self.__initCluster()

        pass


    def __initCluster(self) -> None:
        """
        根据方法名 初始化 聚类器
        Returns: None
        """
        if self.method == dbscan:
            self.cluster = DBSCAN(eps=self.eps, min_samples=self.minSamples)
        elif self.method == kmeans:
            self.cluster = KMeans(n_clusters=self.numClusters)


    def __preProcessing(self, points: List[Point]) -> List[Point]:
        """
        预处理
        Args:
            points: 待聚类的点的容器
        Returns: 预处理之后的点的容器
        """
        points = preprocessing.scale(points)
        return points

    
    def loadData(self, points: List[Point]) -> None:
        """
        加载数据
        Args:
            points: 待聚类的点的容器
        Returns: None
        """
        points = self.__preProcessing(points)
        self.data = points


    def fit(self) -> None:
        """
        拟合
        Returns: None
        """
        self.res = self.cluster.fit(self.data)

    pass




if __name__ == '__main__':
    # 点数据
    points = [
        Point([-2.0, -1.0]), Point([-1.0, -2.0]), Point([0.5, 1.0]),
        Point([3.0, 4.0]), Point([4.0, 3.5]), Point([4.0, 4.0]), Point([5.0, 5.0]), Point([4.0, 7.0]),
        Point([10.0, 1.0]), Point([10.0, 2.0]), Point([9.0, 2.2]), Point([10.0, 3.0]),
    ]

    # 聚类器
    cluster = Cluster(method="dbscan", eps=1.0, minSamples=1)
    # cluster = Cluster(method="kmeans", numClusters=4)
    cluster.loadData(points)
    cluster.fit()
    labels = cluster.cluster.labels_

    # 绘图
    from matplotlib import pyplot as plt
    size = (10, 10)
    xytext = (-20, 10)
    textcoords = "offset points"
    fig = plt.figure(figsize=size)
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    plt.scatter(xs, ys, marker='o')
    for idx, point in enumerate(points):
        plt.annotate("class: {}".format(labels[idx]), xy=point, xytext=(-20, 10), textcoords=textcoords,)
    plt.axis("equal")
    plt.show()
