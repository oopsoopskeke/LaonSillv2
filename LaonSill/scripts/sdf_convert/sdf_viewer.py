#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import cv2
import numpy as np
import shutil

LAONSILL_CLIENT_LIB_PATH = os.path.join(os.environ['LAONSILL_HOME'], 'dev/client/lib')
sys.path.append(LAONSILL_CLIENT_LIB_PATH)
from libLaonSillClient import *

import pdb


SDF_TYPE_DATUM = 'DATUM'
SDF_TYPE_ANNO_DATUM = 'ANNO_DATUM'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--sdf-path", type=str, required=True, 
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset-name", type=str, required=False,
                        help="Base learning rate for training with polynomial decay.")

    return parser.parse_args()


def decodeDatum(sdfPath, dataSet):
    header = RetrieveSDFHeader(sdfPath)
    sdfType = header.type

    dataReader = None
    if sdfType == SDF_TYPE_DATUM:
        dataReader = DataReaderDatum(sdfPath)
    elif sdfType == SDF_TYPE_ANNO_DATUM: 
        dataReader = DataReaderAnnoDatum(sdfPath)

    header = dataReader.getHeader()
    print(":: HEADER ::")
    header.info()
    print("::::::::::::")


    labelItemList = header.labelItemList
    labelItemDict = dict()
    for labelItem in labelItemList:
        labelItemDict[labelItem.label] = labelItem


    if dataSet == None:
        dataReader.selectDataSetByIndex(0)
    else:
        dataReader.selectDataSetByName(dataSet)

    key = 0
    while True:
        datum = None
        # pressed 'p' key (stands for 'peek')
        if key == 1048688:       
            datum = dataReader.peekNextData()
        else:
            datum = dataReader.getNextData()

        print(":: DATUM ::")
        datum.info()
        print("::::::::::::")

        if datum.encoded:
            im = cv2.imdecode(np.fromstring(datum.data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        else:
            im = np.fromstring(datum.data, np.uint8).reshape(datum.channels, datum.height,
                    datum.width).transpose(1, 2, 0)

        if sdfType == SDF_TYPE_DATUM:
            strLabel = str(datum.label)
            color = (255, 0, 0)

            if datum.label in labelItemDict:
                strLabel = labelItemDict[datum.label].displayName

            cv2.putText(im, strLabel, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        elif sdfType == SDF_TYPE_ANNO_DATUM:
            for annoGroup in datum.annotation_groups:
                groupLabel = annoGroup.group_label
        
                strLabel = str(groupLabel)
                color = (255, 0, 0)
                if groupLabel in labelItemDict:
                    strLabel = labelItemDict[groupLabel].displayName
                    if labelItemDict[groupLabel].color != None:
                        color = labelItemDict[groupLabel].color

                for annotation in annoGroup.annotations:
                    xmin = int(datum.width * annotation.bbox.xmin)
                    ymin = int(datum.height * annotation.bbox.ymin)
                    xmax = int(datum.width * annotation.bbox.xmax)
                    ymax = int(datum.height * annotation.bbox.ymax)

                    #print("({}, {}, {}, {})".format(xmin, ymin, xmax, ymax))
                    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(im, strLabel, (xmin + 5, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color, 2)


        cv2.imshow('image', im)

        # pressed 'ESC' key
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(key)
        if key == 1048603:
            break





def main():
    """Create the model and start the training."""
    args = get_arguments()

    sdf_path = os.path.expanduser(args.sdf_path)
    dataset_name = args.dataset_name

    assert os.path.exists(sdf_path), "sdf-path not exists: {}".format(sdf_path)


    print("press ESC key to quit")
    print("press Any key but ESC to see Next Datum")
    decodeDatum(sdf_path, dataset_name)
    

if __name__ == '__main__':
    main()



