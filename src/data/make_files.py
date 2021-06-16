# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:59:34 2021

@author: KWesselkamp
"""
from os import listdir
import os
# function which transforms xml-files of annotations to txt

def XML2TXT(xmlPath: str):
    txtPath = xmlPath.replace(".xml", ".txt").replace("raw", "interim")   # or csv
    f = open(txtPath, 'w')
    with open(xmlPath, 'r') as fp:
        for p in fp:
            if '<object>' in p:
                d = [next(fp).split('>')[1].split('<')[0] for _ in range(10)] # category
                x1 = int(d[-4]) # bounding box
                y1 = int(d[-3])
                x2 = int(d[-2])
                y2 = int(d[-1])
                int_l=[1, x1, y1, x2-x1, y2-y1]
                f.write(", ".join(str(x) for x in int_l))
                f.write('\n')


def make_txt_files():
    os.mkdir('../../data/interim/annotations')
    input_filepath = '../../data/raw/annotations'
    txt_files_list = listdir(input_filepath)
    
    for fl in txt_files_list:
        xml_file_path = input_filepath + '/' + fl
        XML2TXT(xml_file_path)


