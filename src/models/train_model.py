# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:30:04 2021

@author: KWesselkamp
"""
import argparse

import sys
from os import listdir

import torch
import torchvision


from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
from torch.utils.data import Dataset
import construct_dataset
import transforms as T
import utils





class TrainOREvaluate(object):

    def __init__(self,lr,num_epocs,dataset):
        self.lr = lr
        self.num_epocs = num_epocs
        self.dataset = dataset

    '''
        def pngToPIL(img_file_path: str):
        img_files_list = listdir(img_file_path)
        images_list = []
        for fl in img_files_list:
            img_file_path_sg = img_file_path + '/' + fl
            image = Image.open(img_file_path_sg).convert('RGB')

            images_list.append(image)
        return images_list

    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
    '''


    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)



        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)
        device = 'cpu'
     #   torch.cuda.max_memory_allocated()

        if self.dataset == 'normal':
            annotation_list = torch.load('../../data/processed/train/annotation_list.pt')
            images_list = torch.load('../../data/processed/train/images_list.pt')
        else:
            annotation_list = torch.load('../../data/processed/train/annotation_list_augmented.pt')
            images_list = torch.load('../../data/processed/train/images_list_augmented.pt')



        my_dataset = construct_dataset.costructDataset(annotation_list, images_list,None)



        data_loader = torch.utils.data.DataLoader(
            my_dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)



        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (sheep) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


        model.to(device)



        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr,
                                    momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        metric_collector = []


        for epoch in range(self.num_epocs):
            # train for one epoch, printing every 5 iterations
            metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
            metric_collector.append(metric_logger)
            # update the learning rate
            lr_scheduler.step()
            # Evaluate with validation dataset
       #     evaluate(model, data_loader_validation, device=device)
            # save checlpoint
        torch.save(model.state_dict(),  '../../models/sheep_train.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr',
                        default=0.005,
                        type=int)
    parser.add_argument('-num_epocs',
                        default=8,
                        type=int)
    parser.add_argument('-dataset',
                        default='augmentation',
                        type=str)
    args = parser.parse_args()

    if (args.lr and args.num_epocs and args.dataset):
        trainObj = TrainOREvaluate(args.lr,args.num_epocs,args.dataset)
        trainObj.train()

    else:
        print('Please provide the right arguments for evaluation')
        exit(1)
