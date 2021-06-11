# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:59:34 2021

@author: KWesselkamp
"""
from os import listdir
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
# function which transforms xml-files of annotations to txt

def XML2Tensor(xmlPath: str):
    bb_coordinates=np.empty((0,4))
    class_tensor=np.empty((0,1))
    with open(xmlPath, 'r') as fp:
        itr=0
        for p in fp:         
            if '<object>' in p:
                
                d = [next(fp).split('>')[1].split('<')[0] for _ in range(10)] # category
                x1 = float(d[-4]) # check if float works too, otherwise change to integer
                y1 = float(d[-3])
                x2 = float(d[-2])
                y2 = float(d[-1])
                if itr == 0 :
                    bb_coordinates= np.array([x1, y1, x2, y2])
                    class_tensor = np.array([1])
                    itr=1
                else:
                    bb_coordinates = np.vstack((bb_coordinates, np.array([x1, y1, x2, y2])))
                    class_tensor = np.vstack((class_tensor, np.array([1])))
    return {'boxes': torch.from_numpy(bb_coordinates) , 'class': torch.from_numpy(class_tensor)}

def JPG2Tensor(img_file_path_sg: str):
    image = Image.open(img_file_path_sg)
    x = TF.to_tensor(image)/255.0 #use kornia function to import? 
    x.unsqueeze_(0)
    return(x)


def make_target_tensors():
    annot_filepath = '../../data/raw/annotations'
    img_filepath = '../../data/raw/images'
    txt_files_list = listdir(annot_filepath)
    annotation_list = []
    for fl in txt_files_list:
        xml_file_path = annot_filepath + '/' + fl
        output_dictionary = XML2Tensor(xml_file_path)
        annotation_list.append(output_dictionary)
    
    img_files_list = listdir(img_filepath)
    images_list = []
    for fl in img_files_list:
        img_file_path_sg = img_filepath + '/' + fl
        output_dictionary = JPG2Tensor(img_file_path_sg)
        images_list.append(output_dictionary)
    
    return annotation_list, images_list

annotation_list, images_list = make_target_tensors()
print(annotation_list[1])
print(images_list[1])