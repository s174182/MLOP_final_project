# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 09:52:16 2021

@author: KWesselkamp
"""
path_to_project = "C:/Users/kwesselkamp/MLOP_final_project/"

import pytest
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import random
import sys

sys.path.append(path_to_project + "src/models")
sys.path.append(path_to_project + "src/data")
import construct_dataset
import utils
from engine import train_one_epoch, evaluate
import tqdm
import numpy as np

#### Run this test with or with out command line parameter --database=augmented
#### If not set the test is performed on the original dataset


modelpath = path_to_project + "models"
# annotation_list=torch.load(datapath + 'annotation_list.pt')
# images_list=torch.load(datapath + 'images_list.pt')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Training on :", device)


### hyperparameters ###
lr = 0.0017
batch_size = 2
train_size = 0.6
num_epochs = 1


annotation_list = torch.load(
    path_to_project + "data/processed/train/annotation_list.pt"
)
images_list = torch.load(path_to_project + "data/processed/train/images_list.pt")
annotation_list = random.choices(annotation_list, k=10)
images_list = random.choices(images_list, k=10)

train_dataset = construct_dataset.constructDataset(
    annotation_list, images_list, transform=None
)

train_len = int(train_size * len(train_dataset))
valid_len = len(train_dataset) - train_len

train_set, validation_set = torch.utils.data.random_split(
    train_dataset, lengths=[train_len, valid_len]
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn,
)  # collate_fn allows you to have images (tensors) of different sizes

validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn,
)  # collate_fn allows you to have images (tensors) of different sizes

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (sheep) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained final layer classification and box regression layers with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
torch.save(model.state_dict(), path_to_project + "models/sheep_vanilla.pth")
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


class TestClass:    
    
    @pytest.fixture()    
    def test_model_validation(self):
        ## load the model specified in the path
        metric_collector = []
        validation_accuracy_old = 0
        validation_AP_accuracy = 0
        for epoch in tqdm.tqdm(range(num_epochs)):
            validation_accuracy_old = validation_AP_accuracy
            # train for one epoch, printing every 5 iterations
            metric_logger = train_one_epoch(
                model, optimizer, train_loader, device, epoch, print_freq=5
            )
            metric_collector.append(metric_logger)
            # update the learning rate
            lr_scheduler.step()
            # Evaluate with validation dataset
            coco_evalu=evaluate(model, validation_loader, device=device)
            validation_AP_accuracy = coco_evalu.coco_eval.get('bbox').stats[0]
            print(type(validation_AP_accuracy))
            # save checkpoint
            if epoch != 0:
                assert ((validation_accuracy_old <= validation_AP_accuracy) or (
                    np.abs(validation_accuracy_old - validation_AP_accuracy)<0.1))
        
=======
            coco_evalu = evaluate(model, validation_loader, device=device)
            validation_AP_accuracy = coco_evalu.coco_eval.get("bbox").stats[0]
            # save checkpoint
            if epoch != 0:
                assert validation_accuracy_old <= validation_AP_accuracy
        return metric_collector

    def test_model_training(self, test_model_validation):
        metric_collector = test_model_validation
        SL=10
        for m in metric_collector:
            SL_old=SL
            TL=(m.loss.total)
            count=(m.loss.count)
            SL=float(TL/count)
            assert ((SL < SL_old) or (np.abs(SL-SL_old)<0.1)) 
                

    # def test_print_something(self):
    #    print('Please work')

    # assert model_opt in os.listdir(modelpath)
