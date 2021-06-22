# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:55:34 2021

@author: KWesselkamp
"""
path_to_project ='C:/Users/kwesselkamp/MLOP_final_project/'


import pytest
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import random



#### Run this test with or with out command line parameter --database=augmented 
#### If not set the test is performed on the original dataset


modelpath='C:/Users/kwesselkamp/MLOP_final_project/models'
#annotation_list=torch.load(datapath + 'annotation_list.pt')
#images_list=torch.load(datapath + 'images_list.pt')



class TestClass:  
    
    @pytest.fixture()
    def model_string(self, pytestconfig):
        return pytestconfig.getoption("model")

        
    def test_getmodel(self, model_string):
        model_opt=model_string
        print(model_opt)
        print(os.listdir(modelpath))
        assert (any(model_opt in s for s in os.listdir(modelpath)))
        
    @pytest.fixture()
    def getmodel(self, model_string):
        model_opt=model_string
        for s in os.listdir(modelpath):
            if model_opt in s:
                path_of_model=modelpath +'/' + s
        return path_of_model
    
    
    def test_model_output(self, getmodel):
        ## load the model specified in the path 
        path_to_model = getmodel
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        model.eval()
        
        images_list = torch.load(path_to_project + '/data/processed/test/images_test.pt')
        images_list = random.choices(images_list, k=5)
        for im in images_list:
            pred = model(im)
            assert isinstance(pred, list)
            assert isinstance(pred[0], dict)
            
            ### check if every output box is of size 4 ###
            assert pred[0]['boxes'].size()[1] == 4
            
            ### check if every class of the box is 1
            assert torch.sum(pred[0]['labels']).item() == list(pred[0]['labels'].size())[0]
        
    
        
              
    def test_print_name(self, model_string):
        print(f"\ncommand line param (model): {model_string}")
        
    #def test_print_something(self):
    #    print('Please work')
 
        # assert model_opt in os.listdir(modelpath)
    
    
        