# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:01:18 2021

@author: KWesselkamp
"""
import pytest

def pytest_addoption(parser):
    parser.addoption('--augmentation', action='store', default=0,
                     help="want to test on the normal or on the augmented dataset?")
    
@pytest.fixture(scope="function")
def aug_opt(request):
    augmentation = request.config.getoption("augmentation")