# -*- coding: utf-8 -*-
"""
---------------------------------------
File Name:  setup
Description:
Author:     liu
Date:       2021/12/31
---------------------------------------
"""

from setuptools import setup, find_packages

# import sys
# print("[INFO] Current Python interpreter is at: {}".format(sys.executable))

setup(
    name="Welt",
    version="0.2.0.20220727_alpha",
    keywords=["pip", "Welt"],
    description="A Library for Computational Geometry",
    author="liu",
    packages=find_packages(),
    install_requires=["numpy", "more_itertools", "sklearn"],
)
