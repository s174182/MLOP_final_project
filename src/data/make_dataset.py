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

from make_target_tensors import make_target_tensors
from data_augmentation import augmentDataset
from make_target_tensors import makeTargetTensors    


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
    
    #transforming xml-files to txt files (using function in folder)
    # txt-files will be put in the interim-folder
    annotation_list, images_list = make_target_tensors()
    annotation_list, images_list = augmentDataset(annotation_list, images_list)
    
    torch.save(annotation_list, '../../data/processed/annotation_list.pt')
    torch.save(images_list, '../../data/processed/images_list.pt')
    
    # get list of all images in dataset
    #image_files_list = listdir(input_filepath)
    
    #for im in image_files_list:
    #    file_path = input_filepath + '/' + im
    #    x_rgb: torch.tensor = torchvision.io.read_image(file_path)  # CxHxW / torch.uint8
    #    x_rgb = x_rgb.unsqueeze(0)  # BxCxHxW
        
        #resize the images so that they all have the same size
    return annotation_list, images_list   
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    annotation_list, images_list = main()
