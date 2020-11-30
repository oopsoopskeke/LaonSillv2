#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import yaml
import shutil
import pdb
import pudb

LAONSILL_CLIENT_LIB_PATH = os.path.join(os.environ['LAONSILL_HOME'], 'dev/client/lib')
sys.path.append(LAONSILL_CLIENT_LIB_PATH)
from libLaonSillClient import *

DEFAULT_SDF_LIST = "./sdf_list.json"
DEFAULT_CONVERT_PARAM = "./default_convert_param.json"


base_field_set = set(["name", "type", "desc", "build"])
type_value_set = set(["mnist", "imageset", "annoset"])

convert_mnist_data_param = {}
mnist_data_set = {}
convert_image_set_param = {}
image_set = {}
convert_anno_set_param = {}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sdf-list", type=str, default=DEFAULT_SDF_LIST, help="")
    parser.add_argument("--convert-param", type=str, default=DEFAULT_CONVERT_PARAM, help="")

    return parser.parse_args()

def check_base_field(d):
    for base_field in base_field_set:
        if not base_field in d:
            print("key {} is required".format(base_field))
            return False
        if base_field == "type":
            if not d["type"] in type_value_set:
                print("type value {} is invalid".format(d["type"]))
                return False
    return True

def check_mnist_field(d):
    for okey, ovalue in d.iteritems():
        if okey in base_field_set:
            continue
        if not okey in convert_mnist_data_param:
            print("key {} is invalid".format(okey))
            return False
        if type(ovalue) != type(convert_mnist_data_param[okey]):
            print("value type for key {} is invalid".format(okey))
            return False

        if okey == "dataSetList" and len(ovalue) > 0:
            for dataSet in ovalue:
                for ikey, ivalue in dataSet.iteritems():
                    if not ikey in mnist_data_set:
                        print("key {} is invalid".format(ikey))
                        return False
                    if type(ivalue) != type(mnist_data_set[ikey]):
                        print("value type for key {} is invalid".format(ikey))
                        return False

    return True

def check_imageset_field(d):
    for okey, ovalue in d.iteritems():
        if okey in base_field_set:
            continue
        if not okey in convert_image_set_param:
            print("key {} is invalid".format(okey))
            return False
        if type(ovalue) != type(convert_image_set_param[okey]):
            print("value type for key {} is invalid".format(okey))
            return False
        if okey == "imageSetList" and len(ovalue) > 0:
            for imageSet in ovalue:
                for ikey, ivalue in imageSet.iteritems():
                    if not ikey in image_set:
                        print("key {} is invalid".format(ikey))
                        return False
                    if type(ivalue) != type(image_set[ikey]):
                        print("value type for key {} is invalid".format(ikey))
                        return False
    return True

def check_annoset_field(d):
    for okey, ovalue in d.iteritems():
        if okey in base_field_set:
            continue
        if not okey in convert_anno_set_param:
            print("key {} is invalid".format(okey))
            return False
        if type(ovalue) != type(convert_anno_set_param[okey]):
            print("value type for key {} is invalid".format(okey))
            return False
        if okey == "imageSetList" and len(ovalue) > 0:
            for imageSet in ovalue:
                for ikey, ivalue in imageSet.iteritems():
                    if not ikey in image_set:
                        print("key {} is invalid".format(ikey))
                        return False
                    if type(ivalue) != type(image_set[ikey]):
                        print("value type for key {} is invalid".format(ikey))
                        return False
    return True


def print_convert_result(result):
    print("resultCode: {}".format(result.resultCode))
    print("resultMsg: {}".format(result.resultMsg))



# param_dict: 사용자가 json에 정의한 param
# param: ConvertMnistDataParam 객체  
# convert_mnist_data_param: ConvertMnistDataParam에 대한 default value dict 
def handle_mnist(param_dict):
    assert check_mnist_field(param_dict)
    
    convertMnistDataParam = ConvertMnistDataParam()
    for okey, ovalue in convert_mnist_data_param.iteritems():
        if okey in param_dict:
            if okey == "dataSetList" and len(param_dict[okey]) > 0:
                for dataSet in param_dict[okey]:
                    mnistDataSet = MnistDataSet() 
                    for ikey, ivalue in mnist_data_set.iteritems():
                        if ikey in dataSet:
                            if type(dataSet[ikey]) is str: 
                                setattr(mnistDataSet, ikey, os.path.expandvars(dataSet[ikey]))
                            else:
                                setattr(mnistDataSet, ikey, dataSet[ikey])
                        else:
                            setattr(mnistDataSet, ikey, ivalue)
                    convertMnistDataParam.addDataSet(mnistDataSet)
                        
            else:
                if type(param_dict[okey]) is str:
                    setattr(convertMnistDataParam, okey, os.path.expandvars(param_dict[okey]))
                else:
                    setattr(convertMnistDataParam, okey, param_dict[okey])
        else:
            setattr(convertMnistDataParam, okey, ovalue)

    print("ConvertMnistDataParam----------------------------------")
    convertMnistDataParam.info()

    if os.path.exists(convertMnistDataParam.outFilePath):
        shutil.rmtree(convertMnistDataParam.outFilePath)
    ConvertMnistData(convertMnistDataParam)
    print_convert_result(convertMnistDataParam)


def handle_imageset(param_dict):
    assert check_imageset_field(param_dict)

    convertImageSetParam = ConvertImageSetParam()
    for okey, ovalue in convert_image_set_param.iteritems():
        if okey in param_dict:
            if okey == "imageSetList" and len(param_dict[okey]) > 0:
                for image_set in param_dict[okey]:
                    imageSet = ImageSet() 
                    for ikey, ivalue in image_set.iteritems():
                        if ikey in image_set:
                            if type(image_set[ikey]) is str: 
                                setattr(imageSet, ikey, os.path.expandvars(image_set[ikey]))
                            else:
                                setattr(imageSet, ikey, image_set[ikey])
                        else:
                            setattr(imageSet, ikey, ivalue)
                    convertImageSetParam.addImageSet(imageSet)
                        
            else:
                if type(param_dict[okey]) is str:
                    setattr(convertImageSetParam, okey, os.path.expandvars(param_dict[okey]))
                else:
                    setattr(convertImageSetParam, okey, param_dict[okey])
        else:
            setattr(convertImageSetParam, okey, ovalue)

    print("ConvertImageSetParam----------------------------------")
    convertImageSetParam.info()

    if os.path.exists(convertImageSetParam.outFilePath):
        shutil.rmtree(convertImageSetParam.outFilePath)
    ConvertImageSet(convertImageSetParam)
    print_convert_result(convertImageSetParam)

def handle_annoset(param_dict):
    assert check_annoset_field(param_dict)

    convertAnnoSetParam = ConvertAnnoSetParam()
    for okey, ovalue in convert_anno_set_param.iteritems():
        if okey in param_dict:
            if okey == "imageSetList" and len(param_dict[okey]) > 0:
                for image_set in param_dict[okey]:
                    imageSet = ImageSet() 
                    for ikey, ivalue in image_set.iteritems():
                        if ikey in image_set:
                            if type(image_set[ikey]) is str: 
                                setattr(imageSet, ikey, os.path.expandvars(image_set[ikey]))
                            else:
                                setattr(imageSet, ikey, image_set[ikey])
                        else:
                            setattr(imageSet, ikey, ivalue)
                    convertAnnoSetParam.addImageSet(imageSet)
                        
            else:
                if type(param_dict[okey]) is str:
                    setattr(convertAnnoSetParam, okey, os.path.expandvars(param_dict[okey]))
                else:
                    setattr(convertAnnoSetParam, okey, param_dict[okey])
        else:
            setattr(convertAnnoSetParam, okey, ovalue)

    print("ConvertAnnoSetParam----------------------------------")
    convertAnnoSetParam.info()

    if os.path.exists(convertAnnoSetParam.outFilePath):
        shutil.rmtree(convertAnnoSetParam.outFilePath)
    ConvertAnnoSet(convertAnnoSetParam)
    print_convert_result(convertAnnoSetParam)



def main():
    """Create the model and start the training."""
    args = get_arguments()

    sdf_list = os.path.expanduser(args.sdf_list)
    convert_param = os.path.expanduser(args.convert_param)

    global convert_mnist_data_param
    global mnist_data_set
    global convert_image_set_param
    global image_set
    global convert_anno_set_param

    convert_param_file = open(convert_param, "r")
    convert_param_dict = yaml.safe_load(convert_param_file)
    convert_param_file.close()

    convert_mnist_data_param = convert_param_dict["ConvertMnistDataParam"]
    mnist_data_set = convert_param_dict["MnistDataSet"]
    convert_image_set_param = convert_param_dict["ConvertImageSetParam"]
    image_set = convert_param_dict["ImageSet"]
    convert_anno_set_param = convert_param_dict["ConvertAnnoSetParam"]

    #print(convert_mnist_data_param)
    #print(mnist_data_set)
    #print(convert_image_set_param)
    #print(image_set)
    #print(convert_anno_set_param)

    sdf_list_file = open(sdf_list, "r")
    sdf_list_dict = yaml.safe_load(sdf_list_file)
    sdf_list_file.close()

    for name, param in sdf_list_dict.iteritems():
        assert check_base_field(param), "base field for {} is invalid.".format(name)
        if not param["build"]:
            print("skip {} ... ".format(name))
            continue

        type = param["type"]
        if type == "mnist":
            handle_mnist(param)
        elif type == "imageset":
            handle_imageset(param)
        elif type == "annoset":
            handle_annoset(param)


if __name__ == '__main__':
    main()
