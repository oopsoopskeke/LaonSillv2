#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
import struct
import json
import os
from caffe.proto import caffe_pb2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--caffemodel', type=str, default='data/YOLO.caffemodel',
        help='input : caffemodel file path')
ap.add_argument('-p', '--param', type=str, default='data/yolo_caffe.param',
        help='output : param file path')
ap.add_argument('-t', '--prototxt', type=str, default='data/yolo_new.prototxt',
        help='input : network definition prototxt file path')
ap.add_argument('-j', '--json', type=str, default='data/yolo_test_live_new.json',
        help='output : network definition json file path')

args = vars(ap.parse_args())

JSON_FILE = args["json"]
PARAM_FILE = args["param"]
CAFFEMODEL_FILE = args["caffemodel"]
PROTOTXT_FILE = args["prototxt"]

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
            if type == "BatchNorm2":
                layers["%s_variance_correlation" % name] = (1, 1, 1, 1)
            elif type == "BatchNorm3":
                layers["%s_bias_correction" % name] = (1, 1, 1, 1)

    json_file.close()

    return layers

def proto2dict(filename):
    f = open(filename, 'r')
    prototxt = f.read().split('layer')[1:]

    network = OrderedDict()

    for proto in prototxt:
        params = proto.split('\n')
        val = {}
        for param in params:
            if param.count(':') == 0:
                continue
            k, v = param.split(':')
            k, v = k.strip(), v.strip()
            v = v.strip('"')
            try:
                v = float(v)
                if v == int(v): v = int(v)
            except:
                pass

            if k == "name":
                key = v

            if k in val:
                if type(val[k]) != list:
                    vl = [val[k]]
                vl.append(v)
                v = vl

            val[k] = v
        network[key] = val

    f.close()

    return network

def model2dict(model, proto):

    m = caffe_pb2.NetParameter()
    fb = open(model, 'rb')
    m.ParseFromString(fb.read())

    layers = m.layer
    lmap = {}
    for l in layers:
        lmap[l.name] = l

    param = OrderedDict()

    C = 3
    num = 0

    for l_name, layer in proto.items():
        if num == 20: C = 512
        if num == 21: C = 1280
        if "Convolution" in layer["type"]:
            num += 1
            name = str(num) + "d" if num > 18 else str(num)
            W = layer["kernel_size"]
            H = layer["kernel_size"]
            F = layer["num_output"]
            w = np.array(lmap[l_name].blobs[0].data)
            w = w.reshape(F, C, W, H)
            w = w.transpose(0, 1, 2, 3)
            param["conv%s_filter" % str(name)] = w
            if len(lmap[l_name].blobs) == 2:
                b = np.array(lmap[l_name].blobs[1].data)
                b = b.reshape(-1, 1, 1, 1)
                param["conv%s_bias" % str(name)] = b

            C = layer["num_output"]

        if layer["type"] == "BatchNorm":
            scale = np.array(lmap[l_name].blobs[2].data)
            avg = np.array(lmap[l_name].blobs[0].data) / scale
            avg = avg.reshape(1, 1, 1, -1)
            param["bn/conv%s_mean" % str(name)] = avg
            var = np.array(lmap[l_name].blobs[1].data) / scale
            var = var.reshape(1, 1, 1, -1)
            param["bn/conv%s_variance" % str(name)] = var

        if layer["type"] == "Scale":
            gamma = np.array(lmap[l_name].blobs[0].data)
            gamma = gamma.reshape(1, 1, 1, -1)
            param["bn/conv%s_scale" % str(name)] = gamma
            beta = np.array(lmap[l_name].blobs[1].data)
            beta = beta.reshape(1, 1, 1, -1)
            param["bn/conv%s_bias" % str(name)] = beta

    return param

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
proto = proto2dict(PROTOTXT_FILE)
model = model2dict(CAFFEMODEL_FILE, proto)

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

print "The End"
param_file.close()
