# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:30:04 2021
@author: KWesselkamp
"""
import argparse
from torchsummary import summary
import sys
from os import listdir
import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
from torch.utils.data import Dataset
import construct_dataset
import transforms as T
import utils
torch.cuda.empty_cache()

class TrainOREvaluate(object):

    def __init__(self):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.train_size = args.train_size
        self.num_workers = args.num_workers

    def train(self):
      

      

        
        writer = SummaryWriter(log_dir='/content/drive/MyDrive/MLOPS/MLOP_final_project/runs')
        print("Training day and night")

     
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('Training on :', device)

        if self.dataset == 'normal':
            annotation_list = torch.load('/content/drive/MyDrive/MLOPS/MLOP_final_project/data/processed/train/annotation_list.pt')
            images_list = torch.load('/content/drive/MyDrive/MLOPS/MLOP_final_project/data/processed/train/images_list.pt')
        elif self.dataset == 'augmented':

            annotation_list = torch.load('../../data/processed/train/annotation_list_augmented.pt')
            images_list = torch.load('../../data/processed/train/images_list_augmented.pt')
       
        train_dataset = construct_dataset.costructDataset(annotation_list,images_list,transform=None)

        train_len = int(self.train_size * len(train_dataset))
        valid_len = len(train_dataset) - train_len

        train_set, validation_set = torch.utils.data.random_split(train_dataset, lengths=[train_len, valid_len])



        
        
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            collate_fn=utils.collate_fn)
        writer.add_scalar("Num_workers",self.num_workers)
        writer.add_scalar("Batch size",self.batch_size)


        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            collate_fn=utils.collate_fn) # collate_fn allows you to have images (tensors) of different sizes

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        

        #torch.save(model.state_dict(),  '../../models/sheep_vanilla.pth')

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (sheep) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained final layer classification and box regression layers with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.to(device)


        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr,
                                    momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        metric_collector = []


        for epoch in range(self.num_epochs):
            start = time.time()
            # train for one epoch, printing every 5 iterations
            metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=5)
            metric_collector.append(metric_logger)
            TL=(metric_logger.loss.total)
            count=(metric_logger.loss.count)
            SL=float(TL/count)
            writer.add_scalar('loss/train',SL,epoch)
            # update the learning rate
            lr_scheduler.step()
            # Evaluate with validation dataset
            evaluation_result= evaluate(model, validation_loader, device=device)
            # save checlpoint
            test_accuracy=evaluation_result.coco_eval.get('bbox').stats[0]
            writer.add_scalar('loss/test',test_accuracy,epoch)
            end = time.time()
            writer.add_scalar('time/Time_pr_epoch', end-start,epoch)
            torch.save(
                model.state_dict(), "../../models/sheep_train_" + self.dataset + ".pth"
            )
            
        # Creating the test set and testing
        annotation_list = torch.load("../../data/processed/test/annotations_test.pt")
        images_list = torch.load("../../data/processed/test/images_test.pt")

        test_dataset = construct_dataset.constructDataset(
            annotation_list, images_list, transform=None
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=utils.collate_fn,
        )  # collate_fn allows you to have images (tensors) of different sizes

        evaluate(model, test_loader, device=device)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr',
                        default=0.005,
                        type=float)

    parser.add_argument('-num_epochs',
                        default=8,
                        type=int)
    parser.add_argument('-dataset',
                        default='normal',

                        type=str)
    parser.add_argument('-batch_size',
                        default=2,
                        type=int)
    parser.add_argument('-train_size',
                        default=0.9,
                        type=float)
    parser.add_argument('-num_workers',
                        default=4,
                        type=int)
    args = parser.parse_args()


    trainObj = TrainOREvaluate()
    trainObj.train()

