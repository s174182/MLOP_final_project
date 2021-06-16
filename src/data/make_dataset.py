# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import kaggle
import torch

from make_target_tensors import makeTargetTensors



def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    input_filepath = '../../data/raw'

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('andrewmvd/sheep-detection', path=input_filepath, unzip=True)
    

    filesTensors=makeTargetTensors()
    #annotation_list -> dictionary(bounding box coordinates , class number)
    annotation_list, images_list=filesTensors.make_target_tensors()


 #   torch.save( dataloader, "../../data/processed/sheep_DataLoader.pth")

    torch.save(annotation_list, '../../data/processed/annotation_list.pt')
    torch.save(images_list, '../../data/processed/images_list.pt')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
