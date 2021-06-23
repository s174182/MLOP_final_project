import torch
import numpy as np
import pytest


datapath = "../data/processed/train/"

annotation_list = torch.load(datapath + "annotation_list.pt")
images_list = torch.load(datapath + "images_list.pt")


class TestClass:
    """
    def paths(self, aug_opt):
        datapath='../data/processed/train/'
        if aug_opt:
            annotation_list=torch.load(datapath + 'annotation_list_augmented.pt')
            images_list=torch.load(datapath + 'images_list_augmented.pt')
        else :
            annotation_list=torch.load(datapath + 'annotation_list.pt')
            images_list=torch.load(datapath + 'images_list.pt')
        return annotation_list, images_list

    @pytest.mark.parametrize("augmentation",aug_opt)
    """

    def test_images(self):
        #        annotation_list, images_list=TestClass.paths(augmentation)
        length_images = len(images_list)

        for j in range(length_images):

            # test if every item in input list is a tensor
            assert torch.is_tensor(images_list[j])
            # test if every item in input list is of shape [1,3,~,~]
            assert (images_list[j].size()[0] == 1) & (images_list[j].size()[1] == 3)

    def test_annotations(self):
        #        annotation_list, images_list=TestClass.paths(augmentation)
        length_annotations = len(annotation_list)

        for j in range(length_annotations):
            # cla = annotation_list[j].get('class')
            ann = annotation_list[j].get("boxes")

            # test if every bounding box is okay, and if the shape is reasonable
            ann = ann.reshape(-1, 4)
            assert ann.size()[1] == 4

            # x1 must be smaller than x2 and y1 smaller than y2
            assert all(ann[:, 0] < ann[:, 2]) & all(ann[:, 1] < ann[:, 3])


#    def test_dataset_03(self):
