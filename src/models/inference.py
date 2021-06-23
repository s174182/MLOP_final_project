import argparse
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torchvision
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from engine import evaluate
import construct_dataset
import transforms as T
import utils
from os import listdir

import matplotlib
matplotlib.use( 'tkagg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class UseModel(object):

    def __init__(self, model, image):
        if model == 'vanilla':
            self.path_to_model = '../../models/sheep_vanilla.pth'
        elif model == 'normal':
            self.path_to_model = '../../models/sheep_train_normal.pth'
        elif model == 'augmented':
            self.path_to_model = '../../models/sheep_train_augmented.pth'
        else:
            print('Please select a valid dataset')
            exit(1)

        self.path_to_image = '../../data/external/' + image
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def object_detection_api(self, boxes, pred_cls, threshold=0.5, rect_th=3, text_size=3, text_th=3):
        
        img = cv2.imread(self.path_to_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(len(boxes)):
            cv2.rectangle(img, (boxes[i][0][0].astype(int),boxes[i][0][1].astype(int)), (boxes[i][1][0].astype(int), boxes[i][1][1].astype(int)), color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], (boxes[i][0][0].astype(int), boxes[i][0][1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),thickness=text_th)
        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def use(self, threshold):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load(self.path_to_model, map_location=self.device))
        model.eval()

        img = Image.open(self.path_to_image).convert('RGB')

        x = TF.to_tensor(img)  # use kornia function to import?
        x = x.reshape(1, *x.shape)

        COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'sheep']

        pred = model(x)
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(thing) for thing in pred_score if thing > threshold]
        if len(pred_t) == 0:
            pred_boxes = []
            pred_class = []
        else:
            pred_t = pred_t[-1]
            pred_boxes = pred_boxes[:pred_t + 1]
            pred_class = pred_class[:pred_t + 1]

        self.object_detection_api(pred_boxes, pred_class, threshold=threshold, rect_th=3, text_size=3, text_th=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',
                        default='augmented',
                        type=str)
    parser.add_argument('-image',
                        default='figure01.jpeg',
                        type=str)
    parser.add_argument('-threshold',
                        default=0.5,
                        type=float)
    args = parser.parse_args()

    obj_evaluate = UseModel(args.model, args.image)
    obj_evaluate.use(args.threshold)
    