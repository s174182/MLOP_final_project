# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 11:50:28 2021

@author: KWesselkamp
"""

import os

import kornia
import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from bb_transforms import createBinImgFromBB, retrieveBBfromBinImg
from make_target_tensors import make_target_tensors

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def show(img):
    img = torch.squeeze(img) * 255.0
    plt.imshow(img.permute(1, 2, 0))


def applyVerticalFlip(img, ann):
    bb = createBinImgFromBB(img, ann)
    vflip = kornia.geometry.transform.flips.Vflip()
    imgt = vflip(img.squeeze()).unsqueeze_(0)
    bbt = vflip(bb)
    annt = retrieveBBfromBinImg(bbt)
    return imgt, annt


def applyHorizontalFlip(img, ann):
    bb = createBinImgFromBB(img, ann)
    hflip = kornia.geometry.transform.flips.Hflip()
    imgt = hflip(img.squeeze()).unsqueeze_(0)
    bbt = hflip(bb)
    annt = retrieveBBfromBinImg(bbt)
    return imgt, annt


def applyAffineWarp(img, ann):
    bb = createBinImgFromBB(img, ann)
    bb = bb.unsqueeze_(0).unsqueeze_(0).float()
    # A = torch.tensor([[1.,2.,4.],[1.,5.,0.]], dtype=torch.double).unsqueeze(0)
    A = torch.tensor([[0.7, 0.7, 0.5], [0.0, 0.95, 1.0]]).float().unsqueeze(0)
    imgt = kornia.geometry.transform.imgwarp.warp_affine(
        img, A, ((img.size()[2], img.size()[3])), align_corners=True
    )
    bbt = kornia.geometry.transform.imgwarp.warp_affine(
        bb, A, ((img.size()[2], img.size()[3])), align_corners=True
    )
    bbt = bbt.squeeze()
    annt = retrieveBBfromBinImg(bbt)
    return imgt, annt


def augmentDataset(annotation_list, image_list, opt):
    image_list2 = []
    annotation_list2 = []
    for i in range(len(image_list)):
        ann = annotation_list[i]
        annt = np.empty((0, 4))
        if len(list(ann.get("boxes").size())) == 1:
            imgt, anntt = applyVerticalFlip(image_list[i], annotation_list[i])
            image_list2.append(imgt)
            annt = {"boxes": anntt, "class": ann.get("class")}
            annotation_list2.append(annt)
        else:
            for j in range(ann.get("boxes").size()[0]):
                imgt, anntt = applyVerticalFlip(
                    image_list[i],
                    {"boxes": ann.get("boxes")[j], "class": ann.get("class")},
                )
                if j == 0:
                    annt = anntt
                else:
                    annt = torch.vstack((annt, anntt))
            anntd = {"boxes": anntt, "class": ann.get("class")}
            image_list2.append(imgt)
            annotation_list2.append(anntd)

    image_list_joined = image_list + image_list2
    annotation_list_joined = annotation_list + annotation_list2
    annotation_list3 = []
    image_list3 = []
    for i in range(len(image_list_joined)):
        ann = annotation_list_joined[i]
        annt = np.empty((0, 4))
        if len(list(ann.get("boxes").size())) == 1:
            imgt, anntt = applyAffineWarp(
                image_list_joined[i], annotation_list_joined[i]
            )
            image_list3.append(imgt)
            annt = {"boxes": anntt, "class": ann.get("class")}
            annotation_list3.append(annt)
        else:
            for j in range(ann.get("boxes").size()[0]):
                imgt, anntt = applyAffineWarp(
                    image_list_joined[i],
                    {"boxes": ann.get("boxes")[j], "class": ann.get("class")},
                )
                if j == 0:
                    annt = anntt
                else:
                    annt = torch.vstack((annt, anntt))
            anntd = {"boxes": anntt, "class": ann.get("class")}
            image_list3.append(imgt)
            annotation_list3.append(anntd)

    annotation_list_final_join = annotation_list_joined + annotation_list3
    image_list_final_join = image_list_joined + image_list3

    if opt == "joined":
        return annotation_list_final_join, image_list_final_join
    if opt == "extension":
        return annotation_list2 + annotation_list3, image_list2 + image_list3


# annotation_list, image_list = make_target_tensors()
# img = image_list[1]
# ann = annotation_list[1]

# output = kornia.filters.gaussian_blur2d(img, (9, 9), (15, 15))

# boundImg = createBinImgFromBB(img, ann)
# aug = K.RandomAffine((-15., 20.), return_transform=True, p=1.)
# out=aug(img)
# image_transformed = aug.inverse(out).numpy()[0]
# # im_squeezes_ = np.squeeze(image_transformed)

# # save image to path:
# impath='../../data/interim/image1.jpg'
# img= torch.squeeze(img)*255.
# show(img)
# image_transformed = np.moveaxis(image_transformed , 0, -1)
# show(img)
# img_blurry=torch.squeeze(output)*255.
# plt.imshow(img_blurry.permute(1,2,0))
# plt.imshow(boundImg)
# plt.figure()
# show(img)
# imgt, bbt, annt =  applyAffineWarp(img, ann)

# plt.figure()
# plt.imshow(boundImg)
# plt.figure()
# show(imgt)
# plt.imshow(bbt.squeeze())
# bb_new=createBinImgFromBB(imgt, {'boxes':annt, 'class': 1})
# plt.figure()
# plt.imshow(bb_new)
# annt2 = retrieveBBfromBinImg(bbt)
# iml_aug, anl_aug = augmentDataset(image_list, annotation_list)
