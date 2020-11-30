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
    parser.add_argument("--imagefile", type=str, required=True,
                        help="text file path containing target image list.")
    parser.add_argument("--labelfile", type=str, required=True, 
                        help="directory path containing image list.")
    parser.add_argument("--outdir", type=str, required=True, 
                        help="directory path where sdf file is created.")

    return parser.parse_args()



def convert_mnist_data(imagefile, labelfile, outdir):
    param = ConvertMnistDataParam()

    trainDataSet = MnistDataSet()
    trainDataSet.name = 'train'
    trainDataSet.imageFilePath = imagefile    
    trainDataSet.labelFilePath = labelfile

    param.addDataSet(trainDataSet)
    param.outFilePath = outdir 

    if os.path.exists(param.outFilePath):
        shutil.rmtree(param.outFilePath)

    param.info()

    ConvertMnistData(param)

    print("resultCode: {}".format(param.resultCode))
    print("resultMsg: {}".format(param.resultMsg))



def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    imagefile = os.path.expanduser(args.imagefile)
    labelfile = os.path.expanduser(args.labelfile)
    outdir = os.path.expanduser(args.outdir)

    convert_mnist_data(imagefile, labelfile, outdir)
    

if __name__ == '__main__':
    main()
