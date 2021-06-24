# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:04:52 2021

@author: KWesselkamp
"""

def pytest_addoption(parser):
    parser.addoption("--database", action="store", default="augmented")