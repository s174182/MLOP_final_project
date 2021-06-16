# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:30:04 2021

@author: KWesselkamp
"""
import argparse
import os
import sys
from os import listdir
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms.functional as TF
# from engine import train_one_epoch, evaluate
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork

from engine import train_one_epoch
from torch.utils.data import Dataset
import construct_dataset
import transforms as T
import utils

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'tkagg')



class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def validation(model, testloader, criterion):
        accuracy = 0
        test_loss = 0
        for images, labels in testloader:
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        return test_loss, accuracy
    def yo(img_file_path: str):
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



    def object_detection_api(boxes, pred_cls, threshold=0.5, rect_th=3, text_size=3, text_th=3):
        img = cv2.imread('C:/Users/maria/Desktop/mathimata/dtu_mlops/project/photo-1588942173353-0c53a1bf9081.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



        for i in range(len(boxes)):
            print(type(boxes[i]))
            print(type(boxes[i][0][0]))
            cv2.rectangle(img, (boxes[i][0][0].astype(int),boxes[i][0][1].astype(int)), (boxes[i][1][0].astype(int), boxes[i][1][1].astype(int)), color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], (boxes[i][0][0].astype(int), boxes[i][0][1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),thickness=text_th)
        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()


    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        annotation_list = torch.load('../../data/processed/annotation_list.pt')
        images_list =TrainOREvaluate.yo('../../data/raw/images')


        my_dataset = construct_dataset.costructDataset(annotation_list, images_list,TrainOREvaluate.get_transform(train=True))



      #  data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=24, shuffle=True,collate_fn=utils.collate_fn)

        data_loader = torch.utils.data.DataLoader(
            my_dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)


        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


        model.to(device)



        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        metric_collector = []
        num_epochs = 5


        for epoch in range(num_epochs):
            # train for one epoch, printing every 5 iterations
            metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
            metric_collector.append(metric_logger)
            # update the learning rate
            lr_scheduler.step()
            # Evaluate with validation dataset
       #     evaluate(model, data_loader_test, device=device)
            # save checlpoint
        torch.save(model.state_dict(),  '../../models/sheep_train.pth')

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Evaluating arguments')
        parser.add_argument('--load_model_from', default='../../models/sheep_train.pth')
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'sheep']


        #Evaluation  here
        #Evaluation  here
        if args.load_model_from:
            print("Evaluation here")

            print(torch.cuda.is_available())

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

            # replace the classifier with a new one, that has
            # num_classes which is user-defined
            num_classes = 2  # 1 class (person) + background
            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)






            model.load_state_dict(torch.load(args.load_model_from))
            model.eval()

            threshold = 0.5

            img = Image.open('C:/Users/maria/Desktop/mathimata/dtu_mlops/project/photo-1588942173353-0c53a1bf9081.jpg').convert('RGB')


            x = TF.to_tensor(img)  # use kornia function to import?

            print(x[0].shape)

            pred = model([x])
            pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
            pred_score = list(pred[0]['scores'].detach().numpy())
            pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
            pred_boxes = pred_boxes[:pred_t + 1]
            pred_class = pred_class[:pred_t + 1]

            TrainOREvaluate.object_detection_api(pred_boxes, pred_class, threshold=0.5, rect_th=3, text_size=3, text_th=3)



        else:
            print("There is no model to evaluate please train first...")


if __name__ == '__main__':
    TrainOREvaluate()