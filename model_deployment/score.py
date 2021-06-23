import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from azureml.core.model import Model
from azureml.core import Workspace
from PIL import Image
import cv2
import torchvision.transforms.functional as TF
import numpy as np
from azureml.contrib.services.aml_request import AMLRequest
from azureml.contrib.services.aml_response import AMLResponse
import json

# Called when the service is loaded
def init():
    global model

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Running on :', device)

    model_name = 'sheep_train_augmented'

    model_path = Model.get_model_path(model_name)
    print('Path to model: ', model_path)
    print(torch.__version__)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


def object_detection_api(pil_img, boxes, pred_cls, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    open_cv_image = np.array(pil_img) 
    # Convert RGB to BGR 
    img = open_cv_image[:, :, ::-1].copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):
        cv2.rectangle(img, (boxes[i][0][0].astype(int),boxes[i][0][1].astype(int)), (boxes[i][1][0].astype(int), boxes[i][1][1].astype(int)), color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], (boxes[i][0][0].astype(int), boxes[i][0][1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),thickness=text_th)
    return img

# Called when a request is received
def run(request):
    request = json.loads(request)  
    img = Image.fromarray(np.array(json.loads(request['image']), dtype='uint8'))
    x = TF.to_tensor(img)
    x = x.reshape(1, *x.shape)

    threshold = request['threshold']

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

    result = object_detection_api(img, pred_boxes, pred_class, threshold=threshold, rect_th=3, text_size=3, text_th=3)

    # img_pil = Image.fromarray(img)
    json_result = json.dumps(np.array(result).tolist())
    
    return AMLResponse(json_result, 200)
    