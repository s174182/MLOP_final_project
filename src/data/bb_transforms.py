# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 11:13:31 2021

@author: KWesselkamp
"""
import numpy as np
import torch

def createBinImgFromBB (image, annotation):
    bb_coords=annotation.boxes
    binImg = np.zeros(image.size.numpy())
    binImg[bb_coords[0]:bb_coords[2], bb_coords[1]:bb_coords[3]]=1
    return torch.from_numpy(binImg)


def retrieveBBfromBinImg (image):
    binImg=image.numpy()
    colsum = np.sum(binImg, axis=0)
    rowsum = np.sum(binImg, axis=1)

    y1 = np.nonzero(colsum)[0][0]
    y2 = np.nonzero(colsum)[0][-1] + 1
    x1 = np.nonzero(rowsum)[0][0]
    x2 = np.nonzero(rowsum)[0][-1] + 1
    
    
    return torch.from_numpy(np.array([x1, y1, x2, y2]))

