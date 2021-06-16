# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 11:50:28 2021

@author: KWesselkamp
"""

import kornia.augmentation as K
import kornia
import torch
from make_target_tensors import make_target_tensors
from bb_transforms import createBinImgFromBB
from bb_transforms import retrieveBBfromBinImg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def show(img):
    img= torch.squeeze(img)*255.
    plt.imshow( img.permute(1, 2, 0)) 

def applyVerticalFlip(img, ann):
    bb = createBinImgFromBB(img, ann)
    vflip = kornia.geometry.transform.flips.Vflip()
    imgt= vflip(img.squeeze())
    bbt = vflip(bb)
    annt = retrieveBBfromBinImg(bbt)
    return imgt, annt

def augmentDataset(annotation_list, image_list):
    image_list2 = []
    annotation_list2 = []
    for i in range(len(image_list)):
        ann = annotation_list[i]
        annt=np.empty((0,4))
        if len(list(ann.get('boxes').size())) == 1:
            imgt, anntt = applyVerticalFlip(image_list[i], annotation_list[i])
            image_list2.append(imgt)
            annt = {'boxes': anntt , 'class': ann.get('class')}
            annotation_list2.append(annt)
        else:
            for j in range(ann.get('boxes').size()[0]):
                imgt, anntt = applyVerticalFlip(image_list[i], {'boxes': ann.get('boxes')[j], 'class': ann.get('class')})
                if j==0:
                    annt = anntt
                else:
                    annt = torch.vstack((annt, anntt))
            anntd = {'boxes': anntt , 'class': ann.get('class')}
            image_list2.append(imgt)
            annotation_list2.append(anntd)
    
    image_list_joined = image_list + image_list2
    annotation_list_joined = annotation_list + annotation_list2
    
    return annotation_list_joined, image_list_joined
        
#annotation_list, image_list = make_target_tensors()
#img = image_list[9]
#ann = annotation_list[9]

#output = kornia.filters.gaussian_blur2d(img, (9, 9), (15, 15))

# boundImg = createBinImgFromBB(img, ann)
# aug = K.RandomAffine((-15., 20.), return_transform=True, p=1.)
# out=aug(img)
# image_transformed = aug.inverse(out).numpy()[0]
# # im_squeezes_ = np.squeeze(image_transformed)

# # save image to path: 
#impath='../../data/interim/image1.jpg'
# img= torch.squeeze(img)*255.
#show(img)
# image_transformed = np.moveaxis(image_transformed , 0, -1)
#show(img)
#img_blurry=torch.squeeze(output)*255.
#plt.imshow(img_blurry.permute(1,2,0))
#plt.imshow(boundImg)

#imgt, annt =  applyVerticalFlip(img, {'boxes': ann.get('boxes')[1], 'class': ann.get('class')})
#iml_aug, anl_aug = augmentDataset(image_list, annotation_list)