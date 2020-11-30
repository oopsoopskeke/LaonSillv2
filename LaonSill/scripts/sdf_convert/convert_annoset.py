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
    parser.add_argument("--basedir", type=str, required=True, 
                        help="directory path containing image list.")
    parser.add_argument("--outdir", type=str, required=True, 
                        help="directory path where sdf file is created.")
    parser.add_argument("--labelmap", type=str, required=True, 
                        help="whether shuffle the list in dataset or not.")
    parser.add_argument("--shuffle", type=bool, required=False, default=False,
                        help="whether shuffle the list in dataset or not.")

    return parser.parse_args()


def convert_anno_set(dataset, basedir, outdir, labelmap, shuffle):
    param = ConvertAnnoSetParam()

    trainImageSet = ImageSet()
    trainImageSet.name = 'noname'
    trainImageSet.dataSetPath = dataset 

    param.addImageSet(trainImageSet)
    param.labelMapFilePath = labelmap 
    param.basePath = basedir
    param.outFilePath = outdir

    if os.path.exists(param.outFilePath):
        shutil.rmtree(param.outFilePath)

    param.info()

    ConvertAnnoSet(param)

    print("resultCode: {}".format(param.resultCode))
    print("resultMsg: {}".format(param.resultMsg))



def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    dataset = os.path.expanduser(args.dataset)
    basedir = os.path.expanduser(args.basedir)
    outdir = os.path.expanduser(args.outdir)
    labelmap = os.path.expanduser(args.labelmap)
    shuffle = args.shuffle

    convert_anno_set(dataset, basedir, outdir, labelmap, shuffle)
    

if __name__ == '__main__':
    main()
