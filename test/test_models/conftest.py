# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 14:04:21 2021

@author: KWesselkamp
"""

def pytest_addoption(parser):
    parser.addoption("--model", action="store", default="vanilla")