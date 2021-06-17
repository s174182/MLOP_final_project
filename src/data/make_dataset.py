# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import listdir
import numpy as np
import torch
import torchvision
import kaggle
from sklearn.model_selection import train_test_split
import os

from make_target_tensors import make_target_tensors
from data_augmentation import augmentDataset 


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    input_filepath = '../../data/raw'

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('andrewmvd/sheep-detection', path=input_filepath, unzip=True)
    
    #the dataset is in jpg/xml format. images are transformed into a list of tensors, xml- files into a list of dictionaries    
    
    annotation_list, images_list = make_target_tensors()
    
    # augmentDataset can return the joined, or the non-joined version of the transformed data
    
    annotation_list_extension, images_list_extension = augmentDataset(annotation_list, images_list, 'extension')
    
    #split the extension-data into testing and training set 
    
    images_train_extension, images_test, annotations_train_extension, annotations_test = \
                        train_test_split(images_list_extension, annotation_list_extension, test_size=0.33, random_state=42)
    
    # augmented training dataset is created
    images_list_augmented = images_list + images_train_extension
    annotation_list_augmented = annotation_list + annotations_train_extension
    
    
    #saving data into processed-data file 
    train_filepath= '../../data/processed/train'
    test_filepath = '../../data/processed/test'
    os.makedirs(test_filepath,  exist_ok=True), os.makedirs(train_filepath,  exist_ok=True)
    
    torch.save(annotation_list, '../../data/processed/train/annotation_list.pt')
    torch.save(annotation_list_augmented, '../../data/processed/train/annotation_list_augmented.pt')
    
    torch.save(images_list, '../../data/processed/train/images_list.pt')
    torch.save(images_list_augmented, '../../data/processed/train/images_list_augmented.pt')
    
    torch.save(images_test, test_filepath + '/images_test.pt')
    torch.save(annotations_test, test_filepath + '/annotations_test.pt')
    
    # get list of all images in dataset
    #image_files_list = listdir(input_filepath)
    
    #for im in image_files_list:
    #    file_path = input_filepath + '/' + im
    #    x_rgb: torch.tensor = torchvision.io.read_image(file_path)  # CxHxW / torch.uint8
    #    x_rgb = x_rgb.unsqueeze(0)  # BxCxHxW
        
        #resize the images so that they all have the same size
    return annotation_list, images_list  , annotation_list_augmented, images_list_augmented
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    annotation_list, images_list  , annotation_list_augmented, images_list_augmented  = main()
