#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
import struct

NPZ_FILEPATH = "darknet19.weights.npz"
PARAM_FILEPATH = "darknet_pretrain.param"

npz_dic = np.load(NPZ_FILEPATH)

def convert_key2tensor_name(key):
    splited = key.split('-')
    layer_no = str(int(splited[0]) + 1)

    if "kernel" in key or "biases" in key: 
        name = "conv" + layer_no + "_"
    else:
        name = "bn/conv" + layer_no + "_"

    if "moving_variance" in key:
        name = name + "variance"
    elif "moving_mean" in key:
        name = name + "mean"
    elif "gamma" in key:
        name = name + "scale"
    elif "kernel" in key:
        name = name + "filter"
    elif "biases" in key:
        name = name + "bias"
       
    return name

param_file = open(PARAM_FILEPATH, 'wb')

param_count = 0

for key in npz_dic:
    if "gamma" in key:
        param_count = param_count + 3
    else:
        param_count = param_count + 1

print "param count : ", param_count

param_file.write(struct.pack("I", param_count))
  
for key in npz_dic:
    raw_tensor = npz_dic[key]

    param_name = convert_key2tensor_name(key)
    param_name_len = len(param_name)
    print "param name : ", param_name
    param_file.write(struct.pack("Q", param_name_len))

    for i in range(len(param_name)):
        param_file.write(struct.pack("c", param_name[i]))

    if len(raw_tensor.shape) == 4:
        tensor = np.transpose(raw_tensor, (3, 2, 0, 1))

        param_file.write(struct.pack("I", tensor.shape[0]))
        param_file.write(struct.pack("I", tensor.shape[1]))
        param_file.write(struct.pack("I", tensor.shape[2]))
        param_file.write(struct.pack("I", tensor.shape[3]))

        flatten_tensor = tensor.flatten()

        print tensor.shape

    else:
        if "biases" in key:
            param_file.write(struct.pack("I", raw_tensor.shape[0]))
            param_file.write(struct.pack("I", 1))
            param_file.write(struct.pack("I", 1))
            param_file.write(struct.pack("I", 1))
        else:
            param_file.write(struct.pack("I", 1))
            param_file.write(struct.pack("I", 1))
            param_file.write(struct.pack("I", 1))
            param_file.write(struct.pack("I", raw_tensor.shape[0]))

        flatten_tensor = raw_tensor.flatten()
    
    elem_count = len(flatten_tensor) 
    for i in range(elem_count):
        param_file.write(struct.pack("f", flatten_tensor[i]))

    if "gamma" not in key: 
        continue

    param_name = convert_key2tensor_name(key).replace("scale", "bias")
    param_name_len = len(param_name)
    print "param name : ", param_name
    param_file.write(struct.pack("Q", param_name_len))

    for i in range(len(param_name)):
        param_file.write(struct.pack("c", param_name[i]))

    param_file.write(struct.pack("I", 1))
    param_file.write(struct.pack("I", 1))
    param_file.write(struct.pack("I", 1))
    param_file.write(struct.pack("I", raw_tensor.shape[0]))
    
    elem_count = len(flatten_tensor) 
    for i in range(elem_count):
        param_file.write(struct.pack("f", 0.0))

    param_name = convert_key2tensor_name(key).replace("scale", "variance_correlation")
    param_name_len = len(param_name)
    print "param name : ", param_name
    param_file.write(struct.pack("Q", param_name_len))

    for i in range(len(param_name)):
        param_file.write(struct.pack("c", param_name[i]))

    param_file.write(struct.pack("I", 1))
    param_file.write(struct.pack("I", 1))
    param_file.write(struct.pack("I", 1))
    param_file.write(struct.pack("I", 1))
    param_file.write(struct.pack("f", 1.0))

param_file.close()
