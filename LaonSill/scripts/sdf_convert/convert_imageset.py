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




def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SDF File Creation Tool")
    parser.add_argument("--dataset", type=str, required=True,
                        help="text file path containing target image list.")
    parser.add_argument("--dataset-name", type=str, required=False,
                        help="")
    parser.add_argument("--resize", type=int, required=False, default=0,
                        help="")
    parser.add_argument("--resize-width", type=int, required=False, default=0,
                        help="")
    parser.add_argument("--resize-height", type=int, required=False, default=0,
                        help="")
    parser.add_argument("--basedir", type=str, required=True, 
                        help="directory path containing image list.")
    parser.add_argument("--outdir", type=str, required=True, 
                        help="directory path where sdf file is created.")
    parser.add_argument("--labelmap", type=str, required=True, 
                        help="whether shuffle the list in dataset or not.")
    parser.add_argument("--shuffle", type=bool, required=False, default=False,
                        help="whether shuffle the list in dataset or not.")

    return parser.parse_args()



def convert_image_set(dataset, dataset_name, resize_width, resize_height, \
        basedir, outdir, labelmap, shuffle):
    param = ConvertImageSetParam()

    imageSet = ImageSet()

    if dataset_name == None:
        imageSet.name = 'noname'
    else:
        imageSet.name = dataset_name

    imageSet.dataSetPath = dataset 

    param.addImageSet(imageSet)

    param.shuffle = shuffle
    param.resizeWidth = resize_width
    param.resizeHeight = resize_height

    param.basePath = basedir
    param.outFilePath = outdir
    param.labelMapFilePath = labelmap

    if os.path.exists(param.outFilePath):
        shutil.rmtree(param.outFilePath)

    param.info()

    ConvertImageSet(param)

    print("resultCode: {}".format(param.resultCode))
    print("resultMsg: {}".format(param.resultMsg))



def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    dataset = os.path.expanduser(args.dataset)
    dataset_name = args.dataset_name

    resize = args.resize
    resize_width = args.resize_width
    resize_height = args.resize_height

    basedir = os.path.expanduser(args.basedir)
    outdir = os.path.expanduser(args.outdir)
    labelmap = os.path.expanduser(args.labelmap)
    shuffle = args.shuffle

    if resize > 0:
        assert resize_width == 0 and resize_height == 0, \
                "resize width and height should be 0 when resize is larger than 0."
        resize_width = resize
        resize_height = resize

    convert_image_set(dataset, dataset_name, resize_width, resize_height, \
            basedir, outdir, labelmap, shuffle)
    

if __name__ == '__main__':
    main()
