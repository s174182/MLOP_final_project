# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:55:34 2021

@author: KWesselkamp
"""

import pytest
import os
import torch


#### Run this test with or with out command line parameter --database=augmented
#### If not set the test is performed on the original dataset


modelpath = "../../models"
# annotation_list=torch.load(datapath + 'annotation_list.pt')
# images_list=torch.load(datapath + 'images_list.pt')


class TestClass:
    @pytest.fixture()
    def model_string(self, pytestconfig):
        return pytestconfig.getoption("model")

    def test_getmodel(self, model_string):
        model_opt = model_string
        print(model_opt)
        print(os.listdir(modelpath))
        assert any(model_opt in s for s in os.listdir(modelpath))

    @pytest.fixture()
    def getmodel(self, model_string):
        model_opt = model_string
        for s in os.listdir(modelpath):
            if model_opt in s:
                model = torch.load(modelpath + s)
        return model

    def test_print_name(self, model_string):
        print(f"\ncommand line param (model): {model_string}")

    # def test_print_something(self):
    #    print('Please work')

    # assert model_opt in os.listdir(modelpath)
