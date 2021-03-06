# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:59:34 2021

@author: KWesselkamp
"""
import os
import pdb
from os import listdir

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# function which transforms xml-files of annotations to txt
#### SORT DATA BECAUSE THERE'S A CONFUSION BETWEEN ANNOTATION/IMAGES


def XML2Tensor(xmlPath: str):
    bb_coordinates = np.empty((0, 4))
    class_tensor = np.empty((0, 1))
    with open(xmlPath, "r") as fp:
        itr = 0
        for p in fp:
            if "<object>" in p:
                d = [
                    next(fp).split(">")[1].split("<")[0] for _ in range(10)
                ]  # category
                x1 = float(
                    d[-4]
                )  # check if float works too, otherwise change to integer
                y1 = float(d[-3])
                x2 = float(d[-2])
                y2 = float(d[-1])
                if itr == 0:
                    bb_coordinates = np.array([x1, y1, x2, y2])
                    class_tensor = np.array([1])
                    itr = 1
                else:
                    bb_coordinates = np.vstack(
                        (bb_coordinates, np.array([x1, y1, x2, y2]))
                    )
                    class_tensor = np.vstack((class_tensor, np.array([1])))
    return {
        "boxes": torch.from_numpy(bb_coordinates),
        "class": torch.from_numpy(class_tensor),
    }


def JPG2Tensor(img_file_path_sg: str):
    image = Image.open(img_file_path_sg).convert("RGB")
    x = TF.to_tensor(image) / 255.0  # use kornia function to import?
    x.unsqueeze_(0)
    return x


def make_target_tensors():
    annot_filepath = "../../data/raw/annotations"
    img_filepath = "../../data/raw/images"
    txt_files_list = sorted(listdir(annot_filepath))
    annotation_list = []
    for fl in txt_files_list:
        xml_file_path = annot_filepath + "/" + fl
        output_dictionary = XML2Tensor(xml_file_path)
        annotation_list.append(output_dictionary)

    img_files_list = sorted(listdir(img_filepath))
    images_list = []
    for fl in img_files_list:
        img_file_path_sg = img_filepath + "/" + fl
        output_dictionary = JPG2Tensor(img_file_path_sg)
        images_list.append(output_dictionary)

    return annotation_list, images_list
