#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
import struct
import h5py
import json
import os

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-h5', '--h5_filepath', type=str, default='yolo-voc.weights.h5',
        help='input : h5 file path')
ap.add_argument('-param', '--param_filepath', type=str,
        default='yolo_train.param', help='output : param file path')
ap.add_argument('-json', '--json_filepath', type=str,
        default='../../src/examples/YOLO/yolo_train.json',
        help='network definition json file path')

args = vars(ap.parse_args())

JSON_FILE = args["json_filepath"]
H5_FILE = args["h5_filepath"]
PARAM_FILE = args["param_filepath"]

def json2dict(filename):
    json_file = open(filename, 'r')
    js = json.load(json_file)

    layers = OrderedDict()

    for layer in js["layers"]:
        name = layer["name"]
        type = layer["layer"]
        if type == "Conv":
            W = layer["filterDim.rows"]
            H = layer["filterDim.cols"]
            C = layer["filterDim.channels"]
            F = layer["filterDim.filters"]
            layers["%s_filter" % name] = (F, C, W, H)
            layers["%s_bias" % name] = (F, 1, 1, 1)
        elif "BatchNorm" in type:
            layers["%s_scale" % name] = (1, 1, 1, F)
            layers["%s_bias" % name] = (1, 1, 1, F)
            layers["%s_mean" % name] = (1, 1, 1, F)
            layers["%s_variance" % name] = (1, 1, 1, F)
            layers["%s_variance_correlation" % name] = (1, 1, 1, 1)

    json_file.close()

    return layers

def h52dict(filename):
    h5 = h5py.File(filename, 'r')
    params = OrderedDict()

    no_layer = 1
    for k, v in h5.items():
        layer_type, param_name = k.split('.')[-2:]
        val = v[:]
        name = "conv" + str(no_layer)
        if no_layer > 18: name += "d" # for yolo_train.json
        if layer_type == "bn":
            name = "bn/" + name
            val = val.reshape(1, 1, 1, -1)
            if "bias" in param_name:
                key = "%s_bias" % name
            elif "mean" in param_name:
                key = "%s_mean" % name
            elif "var" in param_name:
                key = "%s_variance" % name
            elif "weight" in param_name:
                key = "%s_scale" % name
        elif layer_type == "conv":
            if "weight" in param_name:
                key = "%s_filter" % name
                no_layer += 1
            elif "bias" in param_name:
                key = "%s_bias" % name
                val = val.reshape(-1, 1, 1, 1)
        params[key] = val

    return params

def convert(param_file, param_name, tensor, shape):
    param_name_len = len(param_name)
    param_file.write(struct.pack("Q", param_name_len))

    for i in range(param_name_len):
        param_file.write(struct.pack("c", str(param_name[i])))

    param_file.write(struct.pack("I", tensor.shape[0]))
    param_file.write(struct.pack("I", tensor.shape[1]))
    param_file.write(struct.pack("I", tensor.shape[2]))
    param_file.write(struct.pack("I", tensor.shape[3]))

    flatten = tensor.flatten()

    elem_count = len(flatten)
    for i in range(elem_count):
        param_file.write(struct.pack("f", flatten[i]))

    log = "param name : %s,\t shape : %s" %(name, tensor.shape)
    if tensor.shape != shape:
        log = "[ERROR] " + log + " =/= " + str(shape)

    return log

params = json2dict(JSON_FILE)
model = h52dict(H5_FILE)

os.system('rm -rf %s' % PARAM_FILE)

param_file = open(PARAM_FILE, 'wb')

param_count = 0
for key in params:
    param_count += 1

print "param count : ", param_count

param_file.write(struct.pack("I", param_count))

for name, shape in params.items():

    if name in model:
        tensor = model[name]
        _log = "[.....] "
    else:
        if "bias" in name:
            tensor = np.zeros(shape, float)
        elif "correlation" in name:
            tensor = np.ones(shape, float)
        else: tensor = np.zeros(shape, float)
        _log = "[N E W] "

    log = convert(param_file, name, tensor, shape)
    if "ERROR" not in log: log = _log + log
    print log

param_file.close()
