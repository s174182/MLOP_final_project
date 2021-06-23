# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 17:53:34 2021

@author: KWesselkamp
"""
import torch
import torchdrift
import sys

sys.path.append("../../src/models")
sys.path.append("../../src/data")
import utils
import torchvision
import pytorch_lightning as pl
import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

#%%

### load relevant data, models etc. ###
def create_Dataloaders():
    images_list = torch.load("../../data/processed/test/images_test.pt")
    split_index = round(2 * len(images_list) / 3)
    images_list_train = images_list[:split_index]
    images_list_test = images_list[split_index + 1 :]

    data_loader_train = torch.utils.data.DataLoader(
        images_list_train,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        images_list_test,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )
    return data_loader_train, data_loader_test


def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x.float(), severity=3)


def instantiate_DD():
    feature_extractor = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True
    )
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    in_features = feature_extractor.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    feature_extractor.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    # can be added later
    feature_extractor.load_state_dict(
        torch.load(
            "../../models/sheep_train_augmented.pth", map_location=torch.device("cpu")
        )
    )
    feature_extractor.eval()
    return feature_extractor


def extract_features(dl, opt):
    all_outputs = []
    num_batches = None
    nb = len(dl)

    ### play around with batches (later) ###
    if num_batches is not None:
        nb = min(nb, num_batches)

    for i, b in tqdm.tqdm(zip(range(nb), dl.dataset), total=nb):
        with torch.no_grad():
            if opt == "blurred":
                b = corruption_function(b)
            imp_list = [s.get("scores") for s in feature_extractor(b)]
            rb = torch.mean(torch.stack(imp_list))
            rb = torch.reshape(rb, (-1,))
            all_outputs.append(rb)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_outputs = all_outputs.reshape(-1, 1)
    return all_outputs


if __name__ == "__main__":

    ## create dataloaders ###
    data_loader_train, data_loader_test = create_Dataloaders()

    ## instantiate dd and fd (fasterrcnn)
    feature_extractor = instantiate_DD()
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()

    drift_detection_model = torch.nn.Sequential(feature_extractor, drift_detector)

    ### train the drift_detector ###
    all_outputs = extract_features(data_loader_train, opt="normal")

    ### set 'fit' to the 'training' results ###
    for m in [drift_detector]:
        if hasattr(m, "fit"):
            all_outputs = m.fit(all_outputs)
        else:
            all_outputs = m(all_outputs)

    ### extract features for unblurred /benign data ###
    features = extract_features(data_loader_test, opt="normal")
    ### run the p-test ###
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    print(score, p_val)

    #%%
    # =============================================================================
    #     nb=len(data_loader_test)
    #     features_blurred=[]
    #     for i, b in tqdm.tqdm(zip(range(nb), data_loader_test.dataset), total=nb):
    #
    #         with torch.no_grad():
    #             blurred_b = corruption_function(b)
    #             imp_list=[s.get('scores') for s in feature_extractor(blurred_b)]
    #             rb = torch.mean(torch.stack(imp_list))
    #             rb=torch.reshape(rb, (-1,))
    #             features_blurred.append(rb)
    #
    #     features_blurred=torch.cat(features_blurred, dim=0)
    #     features_blurred = features_blurred.reshape(-1,1)
    # =============================================================================
    features_blurred = extract_features(data_loader_test, opt="blurred")

    ### run the p-test on the blurred data ###
    score_blurred = drift_detector(features_blurred)
    p_val_blurred = drift_detector.compute_p_value(features_blurred)
    print(score_blurred, p_val_blurred)
    #%%
    ### visualizing the two distributions ###

    plt.scatter(
        0.5 * np.random.randn(1, len(features)),
        features.numpy(),
        color="r",
        label="Benign Data",
    )
    plt.scatter(
        0.5 * np.random.randn(1, len(features_blurred)),
        features_blurred.numpy(),
        color="b",
        label="Corrupted Data",
    )
    # circle1=Ellipse((0,torch.mean(features)), 1, torch.std(features), color='r', alpha=0.2)
    # circle2=Ellipse((0,torch.mean(features_blurred)),  1, torch.std(features_blurred), color='b', alpha=0.2)
    plt.legend()
    plt.title("Augmented Model - p-Value={}".format(p_val_blurred))

    # ax.add_patch(circle1)
    # ax.add_patch(circle2)
    plt.savefig("data_drift_augmented.png")
### possible: visualize feature distributions ###
