import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from azureml.core.model import Model
from PIL import Image
import cv2
import torchvision.transforms.functional as TF

# Called when the service is loaded
def init():
    global model

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Running on :', device)

    # model.to(device)

    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('sheep_train_augmented')
    print('Path to model: ', model_path)
    
    #Assume model_name is the variable containing name of your model
    # model_name = 'sheep_train_augmented'
    # ws = Run.get_context().experiment.workspace
    # model_obj = Model(ws, model_name)
    # model_path = model_obj.download(exist_ok = True)
    
    # model_path = Model.get_model_path('sheep_train_augmented')
    # model_path = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/mlops-bigboi/code/Users/s202581/model_deployment/sheep_train_augmented.pth'

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


def object_detection_api(path_to_image, boxes, pred_cls, threshold=0.5, rect_th=3, text_size=3, text_th=3):
        
    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):
        cv2.rectangle(img, (boxes[i][0][0].astype(int),boxes[i][0][1].astype(int)), (boxes[i][1][0].astype(int), boxes[i][1][1].astype(int)), color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], (boxes[i][0][0].astype(int), boxes[i][0][1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),thickness=text_th)
    return img

# Called when a request is received
def run(path_to_image, threshold):
    img = Image.open(path_to_image).convert('RGB')

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

    return object_detection_api(path_to_image, pred_boxes, pred_class, threshold=threshold, rect_th=3, text_size=3, text_th=3)

# init()
# img = run('figure05.jpeg',0.6)

# plt.figure(figsize=(20, 30))
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# plt.show()
# plt.savefig('result.jpeg')
    