# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:59:34 2021

@author: KWesselkamp
"""

# function which transforms xml-files of annotations to txt

def XML2TXT(xmlPath: str):
    txtPath = xmlPath.replace(".xml", ".txt")   # or csv
    f = open(txtPath, 'w')
    with open(xmlPath, 'r') as fp:
        for p in fp:
            if '<object>' in p:
                d = [next(fp).split('>')[1].split('<')[0] for _ in range(10)] # category
                print(d)
                x1 = int(d[-4]) # bounding box
                y1 = int(d[-3])
                x2 = int(d[-2])
                y2 = int(d[-1])
                int_l=[1, x1, y1, x2-x1, y2-y1]
                f.write(", ".join(str(x) for x in int_l))
                f.write('\n')


XML2TXT('C:/Users/KWesselkamp/MLOP_final_project/data/raw/annotations/sheep7.xml')


