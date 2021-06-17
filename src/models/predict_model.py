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



class TestModels(object):

    def __init__(self, num_epocs, path_to_folder):

        self.path_to_model = path_to_folder
        self.num_epocs = num_epocs


    def modelEvaluation(self):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        num_classes = 2  # 1 class (person) + background

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load(args.load_model_from))
        model.to(device)


        test_images_list = torch.load('../../data/processed/train/images_test.pt')
        test_annotation_list = torch.load('../../data/processed/train/annotations_test.pt')


        my_dataset = construct_dataset.costructDataset(test_annotation_list, test_images_list,None)

        data_loader = torch.utils.data.DataLoader(
            my_dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        for epoch in range(self.num_epocs):
            evaluate(model, data_loader, device=device)



    def object_detection_api(boxes, pred_cls, threshold=0.5, rect_th=3, text_size=3, text_th=3):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        img = cv2.imread('C:/Users/maria/Desktop/mathimata/dtu_mlops/project/maxresdefault.jpg')
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




    def evaluate_model(self):
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

        img = Image.open(
            'C:/Users/maria/Desktop/mathimata/dtu_mlops/project/maxresdefault.jpg').convert('RGB')

        x = TF.to_tensor(img)  # use kornia function to import?


        COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'sheep']

        pred = model([x])
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]

        TestModels.object_detection_api(pred_boxes, pred_class, threshold=0.5, rect_th=3, text_size=3, text_th=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_model_from',
                        default='../../models/sheep_train.pth',
                        type=str)
    parser.add_argument('-num_epocs',
                        default=1,
                        type=int)
    args = parser.parse_args()

    if (args.num_epocs and args.load_model_from):
        obj_evaluate = TestModels(args.num_epocs,args.load_model_from)
        obj_evaluate.evaluate_model()

    else:
        print('Please provide for a valid model path for evaluation')
        exit(1)