# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 14:24:36 2021

@author: KWesselkamp
"""
import torch
import tqdm
import torchdrift
import sys, os

sys.path.append("../../src/models")
sys.path.append("../../src/data")
import construct_dataset, utils
import torchvision


annotation_list = torch.load("../../data/processed/test/annotations_test.pt")
images_list = torch.load("../../data/processed/test/images_<test.pt")

my_dataset = construct_dataset.costructDataset(annotation_list, images_list, None)

data_loader = torch.utils.data.DataLoader(
    my_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
)

all_outputs = []
dl = data_loader
num_batches = None
nb = len(dl)
feature_extractor = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True
)
feature_extractor.load_state_dict(torch.load("../../models/sheep_vanilla.pth"))
device = next(feature_extractor.parameters()).device


if num_batches is not None:
    nb = min(nb, num_batches)
for i, b in tqdm.tqdm(zip(range(nb), dl), total=nb):
    if not isinstance(b, torch.Tensor):
        b = b[0]
    with torch.no_grad():
        #### in case of problems - I changed this!
        all_outputs.append(feature_extractor(b.to(device)))
        all_outputs = torch.cat(all_outputs, dim=0)

if __name__ == "main":
    print("Done")
