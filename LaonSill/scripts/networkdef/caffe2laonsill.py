#!/usr/bin/python2.7                                                                           
# -*- coding: utf-8 -*-                                                                        
                                                                                               



#(1) string array 내부 문자열에 대해 쌍따옴표 적용
#(3) field 타입(정수형, 실수형)에 따라 number literal 적법하게 적용
#(4) field value가 문자열이 아닌 타입의 경우 쌍따옴표 제거
#(5) 일부 enumeration 문자열 타입 불일치 (xavier -> Xavier, MAX -> Max 등...)
#(6) Solver 정보 추가

#(2) Conv Layer, filterDim.channels 적용


import argparse                                                                                
import os                                                                                      
import sys                                                                                     
import json

import pdb
import pudb

from collections import OrderedDict
                                                                                               

LAYER_PROP_DEF = os.path.join(os.environ['LAONSILL_DEV_HOME'], 'src/prop/layerPropDef.json')
NETWORK_PROP_DEF = os.path.join(os.environ['LAONSILL_DEV_HOME'], 'src/prop/networkPropDef.json')
OBJ_START = '{'
OBJ_END = '}'
LIST_START = '['
LIST_END = ']'
LIST_DELEM = '^'

INNER_ID_START = 10000


skip_layers = set([
])


list_item = set([
    'bottom',
    'top',
    'param',
    'dim',
    'xxxxxxxxxxxxxxxxxxxxxxx'
])

layer_convert_dict = OrderedDict({
    'Data': 'DataInput',
    'Input': 'DataInput',
    'Convolution': 'Conv',
    'Pooling': 'Pooling',
    'InnerProduct': 'FullyConnected',
    'ReLU': 'Relu',
    'Softmax': 'Softmax',
    'BatchNorm': 'BatchNorm3',
    'Scale': 'Scale',
    'Concat': 'Concat',
    'SoftmaxWithLoss': 'SoftmaxWithLoss',
    'Eltwise': 'ElementWise',
    'Dropout': 'DropOut',
    'xxxxxxxxxxxxxxxxxxxxxxx': ''
})

common_dict = OrderedDict({
    'name': 'name',
    'type': 'layer',
    'bottom': 'input',
    'top': 'output',
    'xxxxxxxxxxxxxxxxxxxxxxx': ''
})

convolution_dict = OrderedDict({
    'param' + LIST_DELEM + '0': OrderedDict({
        'lr_mult': 'weightUpdateParam.lr_mult',
        'decay_mult': 'weightUpdateParam.decay_mult'
    }),
    'param' + LIST_DELEM + '1': OrderedDict({
        'lr_mult': 'biasUpdateParam.lr_mult',
        'decay_mult': 'biasUpdateParam.decay_mult'
    }),
    'convolution_param': OrderedDict({
        'num_output': 'filterDim.filters',
        'kernel_size': ['filterDim.rows', 'filterDim.cols'],
        'stride': 'filterDim.stride',
        'pad': 'filterDim.pad',
        'weight_filler': OrderedDict({
            'type': 'weightFiller.type',
            'value': 'weightFiller.value',
            'mean': 'weightFiller.mean',
            'std': 'weightFiller.std'
        }),
        'bias_filler': OrderedDict({
            'type': 'biasFiller.type',
            'value': 'biasFiller.value',
            'mean': 'biasFiller.mean',
            'std': 'biasFiller.std'
        }),
        'bias_term': 'biasTerm',
        'kernel_h': 'filterDim.rows',
        'kernel_w': 'filterDim.cols',
        'pad_h': 'filterDim.pad_h',
        'pad_w': 'filterDim.pad_w'
    })
})

pooling_dict = OrderedDict({
    'pooling_param': OrderedDict({
        'pool': 'poolingType',
        'kernel_size': ['poolDim.rows', 'poolDim.cols'],
        'stride': 'poolDim.stride',
        'pad': 'poolDim.pad',
        'global_pooling' : 'globalPooling'
    })

})

dropout_dict = OrderedDict({
    'dropout_param': OrderedDict({
        'dropout_ratio': 'probability'
    })
})

innerproduct_dict = OrderedDict({
    'param' + LIST_DELEM + '0': OrderedDict({
        'lr_mult': 'weightUpdateParam.lr_mult',
        'decay_mult': 'weightUpdateParam.decay_mult'
    }),
    'param' + LIST_DELEM + '1': OrderedDict({
        'lr_mult': 'biasUpdateParam.lr_mult',
        'decay_mult': 'biasUpdateParam.decay_mult'
    }),
    'inner_product_param': OrderedDict({
        'num_output': 'nOut',
        'weight_filler': OrderedDict({
            'type': 'weightFiller.type',
            'value': 'weightFiller.value',
            'mean': 'weightFiller.mean',
            'std': 'weightFiller.std'
        }),
        'bias_filler': OrderedDict({
            'type': 'biasFiller.type',
            'value': 'biasFiller.value',
            'mean': 'biasFiller.mean',
            'std': 'biasFiller.std'
        })
    })
})

"""
# for nvCaffe batch_norm_layer
batchnorm_dict = OrderedDict({
    'batch_norm_param': OrderedDict({
        'moving_average_fraction': 'movingAverageFraction',
        'eps': 'eps',
        'scale_bias': 'scaleBias',
        'use_global_stats': 'useGlobalStats'
    })
})
"""

# for caffe batch_norm_layer
batchnorm_dict = OrderedDict({
    'batch_norm_param': OrderedDict({
        'moving_average_fraction': 'movingAverageFraction',
        'eps': 'eps',
        'use_global_stats': 'useGlobalStats'
    })
})

scale_dict = OrderedDict({
    'scale_param' : OrderedDict({
        'axis' : 'axis',
        'num_axes' : 'numAxes',
        'filler': OrderedDict({
            'type': 'filler.type',
            'value': 'filler.value',
            'mean': 'filler.mean',
            'std': 'filler.std'
        }),
        'bias_term' : 'biasTerm',
        'bias_filler': OrderedDict({
            'type': 'biasFiller.type',
            'value': 'biasFiller.value',
            'mean': 'biasFiller.mean',
            'std': 'biasFiller.std'
        })
    })
})

softmaxwithloss_dict = OrderedDict({
    'loss_weight': 'lossWeight'
})

softmax_dict = OrderedDict({
    'axis': 'softmaxAxis'
})

eltwise_dict = OrderedDict({
    'eltwise_param': OrderedDict({
        'operation': 'operation',
        'coeff': 'coeff',
        'stable_prod_grad': 'stableProdGrad'
    })
})


layer_prop = OrderedDict()
network_prop = OrderedDict()

filler_param_converter = OrderedDict({
    "constant": "Constant",
    "gaussian": "Gaussian",
    "uniform": "Uniform",
    "xavier": "Xavier",
    "msra": "MSRA"
})
pooling_type_converter = OrderedDict({
    "MAX": "Max",
    "AVE": "Avg"
})
eltwise_op_converter = OrderedDict({
    "PROD": "PROD",
    "SUM": "SUM",
    "MAX": "MAX"
})

solver_dict = OrderedDict({
    "display": "testInterval",
    "base_lr": "baseLearningRate",
    "lr_policy": "lrPolicy",
    "max_iter": "maxIterations",
    "power": "power",
    "momentum": "momentum",
    "weight_decay": "weightDecay",
    "snapshot": "saveInterval"
})

inner_id_dict = OrderedDict()


def out_file(line, depth_level, outfile):
    final = ''
    for _ in range(depth_level):
        final += '\t'
    final += line
    final += '\n'
    outfile.write(final)

def gen_key_line(key):
    return '\"' + key + '\" : ' 


def normalize_prop_value(prop):
    if prop == 'std::string':
        return 'str'
    if prop == 'uint32_t':
        return 'int'
    if prop == 'double':
        return 'float'
    return prop

def parse_layer_prop_def(def_path):
    json_data = open(def_path).read()
    temp = json.loads(json_data)

    # parent layer들이 먼저 나오도록 sort
    layer_prop_def = OrderedDict()
    while True:
        keys = temp.keys()
        for k in keys:
            v = temp[k]
            parent = v['PARENT']
            if parent == '':
                layer_prop_def[k] = v
                temp.pop(k, None)
            else:
                if parent in layer_prop_def:
                    layer_prop_def[k] = v
                    temp.pop(k, None)
        if len(temp.keys()) == 0:
            break

    layer_prop.clear()
    for layer_name in layer_prop_def:
        parent = layer_prop_def[layer_name]['PARENT']
        vars = layer_prop_def[layer_name]['VARS']

        d = None
        # Base 제외 상속받는 레이어가 있는 경우 
        if parent != '' and parent != 'Base':
            assert(parent in layer_prop), "PARENT layer prop should be defined first:" \
                "{}".format(parent)
            d = OrderedDict(layer_prop[parent])
        else:
            d = OrderedDict()

        for var in vars:
            if len(var) == 3:
                d[var[0]] = normalize_prop_value(var[1])
            # var가 obj인 경우 
            elif len(var) == 4:
                pre = var[0]
                prop_fields = var[3]
                for prop_field in prop_fields:
                    field = pre + "." + prop_field[0]
                    d[field] = normalize_prop_value(prop_field[1])
            else:
                assert (False), "invalid length"

        layer_prop[layer_name] = d

def parse_network_prop_def(def_path):
    json_data = open(def_path).read()
    network_prop_def = json.loads(json_data)

    network_prop.clear()
    vars = network_prop_def['VARS']

    for var in vars:
        network_prop[var[0]] = normalize_prop_value(var[1])




# root부터 leaf까지 한 레이어에 대해 전체 key 리스트 목록을 조회
# @param depth_keys: 최종 depth_key가 담기는 list
# cf) temp_keys: 현재의 경로를 담고 있는 depth key
def get_depth_keys(d, parent_keys, depth_keys):
    keys = d.keys()
    key_len = len(keys)

    for idx in range(key_len):
        k = keys[idx]
        v = d[k]

        temp_keys = None
        if parent_keys == None:
            temp_keys = list()
        else:
            temp_keys = list(parent_keys)

        if type(v) is OrderedDict:
            temp_keys.append(k)
            get_depth_keys(v, temp_keys, depth_keys)
        elif type(v) is list and type(v[0]) is OrderedDict :
            for idx in range(len(v)):
                item = v[idx]
                temp_keys.append(k + LIST_DELEM + str(idx))

                # list안 dict 케이스
                if type(item) is OrderedDict:
                    get_depth_keys(item, temp_keys, depth_keys)
                # list안 list 케이스
                elif type(item) is list:
                    assert (False), "not implemented case"
                # list안 기타 item 케이스
                else:
                    depth_keys.append(temp_keys)

                temp_keys.pop()
        else:
            temp_keys.append(k) 
            depth_keys.append(temp_keys)
            

# 여러 단계로 구성된 key가 layer dict의 정의와 일치하는지 테스트
def is_valid_depth_key(d, depth_key):
    depth_key_len = len(depth_key)

    value = None
    for idx in range(depth_key_len):
        key = depth_key[idx]

        if key not in d:
            return False

        if idx < depth_key_len - 1:
            value = d[key]
            if not type(value) is OrderedDict:
                return False
            d = value

    return True

def get_special_key(key):
    splits = key.split(LIST_DELEM)
    if len(splits) == 2:
        return [splits[0], int(splits[1])]
    elif len(splits) == 1:
        return [None, -1]
    else:
        assert (False), "invalid key: {}".format(key)



def retrieve_key_value(layer_dict, obj_dict, depth_key):
    depth_key_len = len(depth_key)

    ret_key = None
    ret_val = None
    for idx in range(depth_key_len):
        key = depth_key[idx]
        special_key, list_index = get_special_key(key)

        ret_key = layer_dict[key]
        if special_key == None:
            ret_val = obj_dict[key]
        else:
            ret_val = obj_dict[special_key][list_index]

        layer_dict = ret_key
        obj_dict = ret_val

    return ret_key, ret_val

def convert_key_value_to_line(layer, key, value, last_line):
    line = gen_key_line(key)

    prop_type = 'str'
    if key != 'layer':
        assert(layer in layer_prop), "layer [{}] is not in layer_prop".format(layer)
        assert(key in layer_prop[layer]), "prop [{}] is not in layer_prop[{}]".format(key, layer)
        prop_type = layer_prop[layer][key]

    if prop_type == 'float':
        value = str(value)
        if not "." in value:
            value += ".0"
        line += value
    elif prop_type == 'int' or prop_type == 'float' or prop_type == 'bool':
        line += str(value)
    elif prop_type == 'std::vector<std::string>':
        line += '['
        value_len = len(value)
        for i in range(value_len):
            line += '\"' + value[i] + '\"'
            if i < value_len -1:
                line += ', '
        line += ']'
    elif prop_type == 'std::vector<bool>':
        line += '['
        value_len = len(value)
        for i in range(value_len):
            line += value[i]
            if i < value_len -1:
                line += ', '
        line += ']'

    else:
        line += '\"' + value + '\"' 

    if not last_line:
        line += ','

    return line
    

def gen_softmax_inner_layer(axis):
    inner_layer_type = 'Softmax'
    inner_layer_id = get_inner_id(inner_layer_type)

    # 생성자로 바로 Dict를 생성하니 순서가 꼬인다.
    inner_layer_obj = OrderedDict()
    inner_layer_obj["name"] = "inner_softmax"
    inner_layer_obj["type"] = "Softmax"
    inner_layer_obj["bottom"] = ["inner_softmax_" + str(inner_layer_id) + "_input"]
    inner_layer_obj["top"] = ["inner_softmax_" + str(inner_layer_id) + "_output"]
    inner_layer_obj["axis"] = axis

    inner_depth_keys = list()
    for key in inner_layer_obj.keys():
        inner_depth_keys.append(list())
        inner_depth_keys[-1].append(key)

    return inner_layer_id, inner_layer_obj, inner_depth_keys



def innerlayer_handler(inner_layer_type, inner_layer_id, inner_layer_obj, inner_depth_keys, \
        depth_level, outfile):

    out_file(gen_key_line('innerLayer'), depth_level, outfile)
    out_file(LIST_START, depth_level, outfile)
    depth_level += 1

    out_file(OBJ_START, depth_level, outfile)
    depth_level += 1

    common_handler(inner_layer_id, inner_layer_obj, inner_depth_keys, depth_level, outfile)
    layer_handler_dict[inner_layer_type](inner_layer_obj, inner_depth_keys, depth_level, outfile)

    depth_level -= 1
    out_file(OBJ_END, depth_level, outfile)

    depth_level -= 1
    out_file(LIST_END, depth_level, outfile)



def common_handler(layer_id, layer_obj, depth_keys, depth_level, outfile):
    wasType = False
    depth_keys_len = len(depth_keys)
    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)

        if not is_valid_depth_key(common_dict, depth_key):
            continue

        key, value = retrieve_key_value(common_dict, layer_obj, depth_key)

        dst_value = None
        # special case: type인 경우 value를 변환해야 함
        # 'type'이 last_line일 수는 없다고 봄 (input 또는 output이 따라옴)
        if depth_key[0] == 'type':
            value = layer_convert_dict[value] 
            wasType = True

        line = convert_key_value_to_line('Base', key, value, last_line) 
        out_file(line, depth_level, outfile)

        if wasType:
            line = convert_key_value_to_line('Base', 'id', layer_id, last_line)
            out_file(line, depth_level, outfile)
            wasType = False

def data_handler(layer_obj, depth_keys, depth_level, outfile):
    pass

def concat_handler(layer_obj, depth_keys, depth_level, outfile):
    pass

def input_handler(layer_obj, depth_keys, depth_level, outfile):
    pass

def convolution_handler(layer_obj, depth_keys, depth_level, outfile):
    paramIdx = 0
    depth_keys_len = len(depth_keys)
    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)

        if is_valid_depth_key(common_dict, depth_key):
            continue

        if depth_key[0] == 'param':
            depth_key[0] += LIST_DELEM + str(paramIdx)
            paramIdx += 1

        assert(is_valid_depth_key(convolution_dict, depth_key)), \
                "invalid depth key in convolution layer: {}".format(depth_key)

        key, value = retrieve_key_value(convolution_dict, layer_obj, depth_key)
        #print("key={}, value={}".format(key, value))

        if type(key) is list:
            key_len = len(key)
            for key_idx in range(key_len):
                k = key[key_idx]
                """
                if key_idx == key_len - 1:
                    last_line = True
                else:
                    last_line = False
                    """

                line = convert_key_value_to_line('Conv', k, value, False)
                out_file(line, depth_level, outfile)
        else:
            if key == "weightFiller.type" or key == "biasFiller.type":
                assert (value in filler_param_converter), "Unsupported filler type: "\
                    "{}".format(value)
                value = filler_param_converter[value]

            line = convert_key_value_to_line('Conv', key, value, last_line)
            out_file(line, depth_level, outfile)


def pooling_handler(layer_obj, depth_keys, depth_level, outfile):
    depth_keys_len = len(depth_keys)
    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)
        if is_valid_depth_key(common_dict, depth_key):
            continue

        assert(is_valid_depth_key(pooling_dict, depth_key)), \
            "invalid depth key in pooling layer: {}".format(depth_key)

        key, value = retrieve_key_value(pooling_dict, layer_obj, depth_key)
        #print("key={}, value={}".format(key, value))

        if type(key) is list:

            key_len = len(key)
            for key_idx in range(key_len):
                k = key[key_idx]
                """
                if key_idx == key_len - 1:
                    last_line = True
                else:
                    last_line = False
                    """
                line = convert_key_value_to_line('Pooling', k, value, False)
                out_file(line, depth_level, outfile)
        else:
            if key == "poolingType":
                assert (value in pooling_type_converter), "Unsupported pooling type: "\
                    "{}".format(value)
                value = pooling_type_converter[value]

            line = convert_key_value_to_line('Pooling', key, value, last_line)
            out_file(line, depth_level, outfile)


def innerproduct_handler(layer_obj, depth_keys, depth_level, outfile):
    paramIdx = 0
    depth_keys_len = len(depth_keys)
    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)
        if is_valid_depth_key(common_dict, depth_key):
            continue

        if depth_key[0] == 'param':
            depth_key[0] += LIST_DELEM + str(paramIdx)
            paramIdx += 1

        assert(is_valid_depth_key(innerproduct_dict, depth_key)), \
            "invalid depth key in innerproduct layer: {}".format(depth_key)

        key, value = retrieve_key_value(innerproduct_dict, layer_obj, depth_key)
        #print("key={}, value={}".format(key, value))

        if type(key) is list:
            key_len = len(key)
            for key_idx in range(key_len):
                k = key[key_idx]
                if key_idx == key_len - 1:
                    last_line = True
                else:
                    last_line = False
                line = convert_key_value_to_line('FullyConnected', k, value, last_line)
                out_file(line, depth_level, outfile)
        else:
            if key == "weightFiller.type" or key == "biasFiller.type":
                assert (value in filler_param_converter), "Unsupported filler type: "\
                    "{}".format(value)
                value = filler_param_converter[value]
            line = convert_key_value_to_line('FullyConnected', key, value, last_line)
            out_file(line, depth_level, outfile)


def batchnorm_handler(layer_obj, depth_keys, depth_level, outfile):
    depth_keys_len = len(depth_keys)
    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)
        if is_valid_depth_key(common_dict, depth_key):
            continue

        assert(is_valid_depth_key(batchnorm_dict, depth_key)), \
            "invalid depth key in batchnorm layer: {}".format(depth_key)

        key, value = retrieve_key_value(batchnorm_dict, layer_obj, depth_key)

        line = convert_key_value_to_line('BatchNorm3', key, value, last_line)
        out_file(line, depth_level, outfile)

def scale_handler(layer_obj, depth_keys, depth_level, outfile):
    depth_keys_len = len(depth_keys)
    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)
        if is_valid_depth_key(common_dict, depth_key):
            continue

        assert(is_valid_depth_key(scale_dict, depth_key)), \
            "invalid depth key in scale layer: {}".format(depth_key)

        key, value = retrieve_key_value(scale_dict, layer_obj, depth_key)

        line = convert_key_value_to_line('Scale', key, value, last_line)
        out_file(line, depth_level, outfile)

def relu_handler(layer_obj, depth_keys, depth_level, outfile):
    print('relu_handler ... ')
    pass

def softmax_handler(layer_obj, depth_keys, depth_level, outfile):
    depth_keys_len = len(depth_keys)
    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)
        if is_valid_depth_key(common_dict, depth_key):
            continue

        assert(is_valid_depth_key(softmax_dict, depth_key)), \
            "invalid depth key in softmax layer: {}".format(depth_key)

        key, value = retrieve_key_value(softmax_dict, layer_obj, depth_key)

        line = convert_key_value_to_line('Softmax', key, value, last_line)
        out_file(line, depth_level, outfile)

def dropout_handler(layer_obj, depth_keys, depth_level, outfile):
    depth_keys_len = len(depth_keys)
    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)
        if is_valid_depth_key(common_dict, depth_key):
            continue

        assert(is_valid_depth_key(dropout_dict, depth_key)), \
            "invalid depth key in dropout layer: {}".format(depth_key)

        key, value = retrieve_key_value(dropout_dict, layer_obj, depth_key)

        line = convert_key_value_to_line('DropOut', key, value, last_line)
        out_file(line, depth_level, outfile)

def eltwise_handler(layer_obj, depth_keys, depth_level, outfile):
    depth_keys_len = len(depth_keys)
    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)
        if is_valid_depth_key(common_dict, depth_key):
            continue

        assert(is_valid_depth_key(eltwise_dict, depth_key)), \
            "invalid depth key in eltwise layer: {}".format(depth_key)

        key, value = retrieve_key_value(eltwise_dict, layer_obj, depth_key)

        if key == "operation":
            assert (value in eltwise_op_converter), "Unsupported eltwise op: "\
                "{}".format(value)
            value = eltwise_op_converter[value]

        # 아래의 layer 이름은 SoooA 기준이다.
        line = convert_key_value_to_line('ElementWise', key, value, last_line)
        out_file(line, depth_level, outfile)


def get_inner_id(layer_type):
    layer_id = -1

    if layer_type in inner_id_dict:
        inner_id_dict[layer_type] += 10
        layer_id = inner_id_dict[layer_type]
    else:
        global INNER_ID_START
        INNER_ID_START += 1000
        inner_id_dict[layer_type] = INNER_ID_START
        layer_id = inner_id_dict[layer_type]

    return layer_id


def softmaxwithloss_handler(layer_obj, depth_keys, depth_level, outfile):
    depth_keys_len = len(depth_keys)
    axis = 2
    foundPropDown = False
    foundSoftmaxAxis = False

    for i in range(depth_keys_len):
        depth_key = depth_keys[i]
        last_line = (i == depth_keys_len - 1)
        if is_valid_depth_key(common_dict, depth_key):
            continue

        assert(is_valid_depth_key(softmaxwithloss_dict, depth_key)), \
            "invalid depth key in softmaxwithloss layer: {}".format(depth_key)

        key, value = retrieve_key_value(softmaxwithloss_dict, layer_obj, depth_key)
        if key == "softmaxAxis":
            axis = int(value)
            foundSoftmaxAxis = True
        elif key == "propDown":
            foundPropDown = True

        line = convert_key_value_to_line('SoftmaxWithLoss', key, value, last_line)
        out_file(line, depth_level, outfile)

    if not foundPropDown:
        line = convert_key_value_to_line('Base', 'propDown', ["true", "false"], False)
        out_file(line, depth_level, outfile)
    if not foundSoftmaxAxis:
        line = convert_key_value_to_line('SoftmaxWithLoss', 'softmaxAxis', axis, False)
        out_file(line, depth_level, outfile)


    inner_layer_id, inner_layer_obj, inner_depth_keys = gen_softmax_inner_layer(axis)
    innerlayer_handler('Softmax', inner_layer_id, inner_layer_obj, inner_depth_keys, \
            depth_level, outfile)







layer_handler_dict = OrderedDict({
    'Data': data_handler,
    'Input': input_handler,
    'Convolution': convolution_handler,
    'Pooling': pooling_handler,
    'InnerProduct': innerproduct_handler,
    'ReLU': relu_handler,
    'Softmax': softmax_handler,
    'BatchNorm': batchnorm_handler,
    'Scale': scale_handler,
    'SoftmaxWithLoss': softmaxwithloss_handler,
    'Concat': concat_handler,
    'Softmax': softmax_handler,
    'Eltwise': eltwise_handler,
    'Dropout': dropout_handler,
    'xxxxxxxxxxxxxxxxxxxxxxx': None
})






                                                                                               
def get_arguments():                                                                           
    """Parse all the arguments provided from the CLI.                                          
                                                                                               
    Returns:                                                                                   
      A list of parsed arguments.                                                              
    """                                                                                        
    parser = argparse.ArgumentParser(description="Caffe protobuf to SoooA network def Converter")  
    parser.add_argument("-i", "--infile", type=str,
                        help="input caffe network definition protobuf file")              
    parser.add_argument("-s", "--solver", type=str, required=False,
                        help="input caffe solver definition protobuf file")          
    parser.add_argument("-o", "--outfile", type=str, 
                        help="output soooa network definition json file")
    parser.add_argument("--layerpropdef", type=str, default=LAYER_PROP_DEF,
                        help="output soooa network definition json file")
    parser.add_argument("--networkpropdef", type=str, default=NETWORK_PROP_DEF,
                        help="output soooa network definition json file")
    return parser.parse_args()                                                                 

def preprocess_line(line):
    if line.startswith('#'):
        print('comment: skip')
        return ''

    splits = line.split('#')
    line = splits[0].strip()

    return line

def preprocess_lines(lines):
    lines_len = len(lines)

    for line_idx in reversed(xrange(lines_len)): 
        line = lines[line_idx].strip()
        line = preprocess_line(line)
        if len(line) < 1:
            del lines[line_idx]
        else:
            lines[line_idx] = line


def get_line_type(line):
    line_type = None
    if line.endswith('{'):
        line_type = 'obj_head'
    elif line == '}':
        line_type = 'obj_tail'
    # 반드시 obj_field보다 먼저 처리해야 함.
    # 그렇지 않을 경우 inline obj가 obj_field로 걸림
    elif '{' in line and ':' in line and '}' in line:
        line_type = 'inline_obj'
    elif ':' in line:
        line_type = 'obj_field'
    else:
        line_type = 'unknown'

    return line_type



# obj head에 대해서만!
# field는 별도 처리 ...
def handle_obj_head(line, obj_stack):
    assert (len(obj_stack) > 0), "obj_stack is empty."
    cur_obj = obj_stack[-1]

    obj_name = line.split('{')[0].strip()
    obj = OrderedDict()

    # obj가 repeated 타입인 경우
    if obj_name in list_item:
        if not obj_name in cur_obj:
            cur_obj[obj_name] = list()
        cur_obj[obj_name].append(obj)

    # obj가 repeated 타입이 아닌 경우 
    else:
        cur_obj[obj_name] = obj

    # 추가한 obj를 stack에 push
    obj_stack.append(obj)


def handle_obj_tail(line, obj_stack):
    assert (len(obj_stack) > 0), "obj_stack is empty."

    obj_stack.pop()



def handle_obj_field(line, obj_stack):
    assert (len(obj_stack) > 0), "obj_stack is empty."
    cur_obj = obj_stack[-1]

    splits = line.split(':')
    assert (len(splits) == 2), "invalid obj_field line: {}".format(line)

    field_name = splits[0].strip()
    field_value = splits[1].strip()
    field_value = field_value.replace('\"', '').replace('\'', '')

    #assert (not field_name in cur_obj), "field name duplicated: {}".format(line)
    # obj가 repeated 타입인 경우
    if field_name in list_item:
        if not field_name in cur_obj:
            cur_obj[field_name] = list()
        cur_obj[field_name].append(field_value)

    # obj가 repeated 타입이 아닌 경우 
    else:
        cur_obj[field_name] = field_value



def handle_inline_obj(line, obj_stack):
    print("not handle inline obj currently... handle it manually: [{}]".format(line))
    pass

def parse_layer(lines, cur_line, unhandled_lines):
    layer_start = False
    layer_end = False
    stack_size = 0

    obj_stack = list()
    layer_dict = OrderedDict()
    obj_stack.append(layer_dict)

    while not layer_end:
        line = lines[cur_line]
        #line = lines[cur_line].strip()
        cur_line += 1

        #print('cur line: [{}]'.format(line))
        #line = preprocess_line(line)
        #print('preprocessed: [{}]'.format(line))

        # ignores empty line
        #if len(line) < 1:
        #    continue

        line_type = get_line_type(line)
        #print('line type: [{}]'.format(line_type))
        assert(line_type != 'unknown'), "unknown line type ...: {}".format(line)

        if not layer_start: 
            if line_type == 'obj_head' and line.startswith('layer'):
                layer_start = True
                layer_dict.clear()
                continue
            elif line_type == 'obj_field':
                print('Network field ... skip')
                continue
            else:
                assert(False), "invalid line->{}".format(line)
        else:
            if line_type == 'obj_tail' and len(obj_stack) == 1:
                layer_start = False
                layer_end = True
                continue
            elif line_type == 'obj_head' and line.startswith('layer'):
                assert(False), "invalid line->{}".format(line)

        # layer obj가 시작하지 않았을 경우 이쪽으로 무조건 내려오지 않음
        # layer obj가 시작했을 경우 layer obj의 끝을 만난 경우가 아니면 항상 내려옴
        # layer obj의 content를 아래에서 처리 

        # layer object head has been filtered already
        # 이 경우 항상 layer inner obj head
        if line_type == 'obj_head':
            handle_obj_head(line, obj_stack)
        # 이 경우 항상 layer inner obj tail
        elif line_type == 'obj_tail':
            handle_obj_tail(line, obj_stack)
        elif line_type == 'obj_field':
            handle_obj_field(line, obj_stack)
        elif line_type == 'inline_obj':
            #handle_inline_obj(line, obj_stack)
            unhandled_lines.append(cur_line - 1)

        #print('------------------------------------')


    assert (len(obj_stack) == 1), "obj stack should only contain layer obj"
    return cur_line, obj_stack[0]


def convert(layer_id, layer_obj, depth_level, outfile, last_layer):
    type = layer_obj['type']

    if type in skip_layers:
        print('Skips [{}] layer ... '.format(type))
        return
    assert (type in layer_handler_dict), "handler for type {} is not implemented".format(type)
    
    out_file(OBJ_START, depth_level, outfile)
    depth_level += 1

    depth_keys = list()
    get_depth_keys(layer_obj, None, depth_keys)

    #for depth_key in depth_keys:
    #    print(depth_key)
    common_handler(layer_id, layer_obj, depth_keys, depth_level, outfile)
    layer_handler_dict[type](layer_obj, depth_keys, depth_level, outfile)


    depth_level -= 1
    if not last_layer:
        out_file(OBJ_END + ',\n', depth_level, outfile)
    else:
        out_file(OBJ_END + '\n', depth_level, outfile)





def parse_solver(solver_path, depth_level, outfile):
    solverfile = open(solver_path, 'rb')
    lines = solverfile.readlines()
    solverfile.close()

    preprocess_lines(lines) 



    last_line = False
    lines_len = len(lines)
    for line_idx in xrange(lines_len):
        line = lines[line_idx]
        if line_idx == lines_len - 1:
            last_line = True

        splits = line.split(':')
        assert (len(splits) == 2), "invalid obj_field line: {}".format(line)

        field_name = splits[0].strip()
        field_value = splits[1].strip()
        value = field_value.replace('\"', '').replace('\'', '')

        if not field_name in solver_dict:
            print("unknown solver field {} ... skip ...".format(field_name))
            continue
        
        key = solver_dict[field_name]




        line = '\"' + key + '\" : ' 

        prop_type = 'str'
        if key != 'layer':
            assert(key in network_prop), "solver field [{}] is not in network_prop".format(key)
            prop_type = network_prop[key]

        if prop_type == 'float':
            value = str(value)
            if not "." in value:
                value += ".0"
            line += value
        elif prop_type == 'int' or prop_type == 'float' or prop_type == 'bool':
            line += str(value)
        elif prop_type == 'std::vector<std::string>':
            line += '['
            value_len = len(value)
            for i in range(value_len):
                line += '\"' + value[i] + '\"'
                if i < value_len -1:
                    line += ', '
            line += ']'
        else:
            line += '\"' + value + '\"' 

        if not last_line:
            line += ','

        out_file(line, depth_level, outfile)


                                                                                               
                                                                                               
def main():                                                                                    
    """Create the model and start the training."""                                             
    args = get_arguments()                                                                     

    infile_path = args.infile
    solver_path = args.solver
    outfile_path = args.outfile 
    layerpropdef_path = args.layerpropdef
    networkpropdef_path = args.networkpropdef

    has_solver = (solver_path != None)


    #f = open('inception_v3.prototxt', 'rb')
    infile = open(infile_path, 'rb')
    lines = infile.readlines()
    infile.close()
    
    parse_layer_prop_def(layerpropdef_path)

    #if os.path.exists(outfile_path):
    #    os.remove(outfile_path)
    outfile = open(outfile_path, 'w')
    depth_level = 0

    out_file(OBJ_START, depth_level, outfile)
    depth_level += 1
    out_file(gen_key_line('layers'), depth_level, outfile)
    out_file(LIST_START, depth_level, outfile)
    depth_level += 1

    unhandled_lines = list()

    preprocess_lines(lines)
    num_lines = len(lines)

    cur_line = 0
    layer_id = 0

    first_layer = True
    last_layer = False
    while cur_line < num_lines:
        cur_line, layer_obj = parse_layer(lines, cur_line, unhandled_lines)
        #print(layer_obj)

        if first_layer:
            if 'Input' != layer_obj['type'] or 'Data' != layer_obj['type']:
                layer_id += 10
            first_layer = False

        if cur_line == num_lines:
            last_layer = True

        convert(layer_id, layer_obj, depth_level, outfile, last_layer)
        layer_id += 10

    
    depth_level -= 1
    out_file(LIST_END + ",\n", depth_level, outfile)


    out_file(gen_key_line('configs'), depth_level, outfile)
    out_file(OBJ_START, depth_level, outfile)
    depth_level += 1



    if has_solver:
        parse_network_prop_def(networkpropdef_path)
        parse_solver(solver_path, depth_level, outfile) 






    depth_level -= 1
    out_file(OBJ_END, depth_level, outfile)

    depth_level -= 1
    out_file(OBJ_END, depth_level, outfile)

    outfile.close()

    print('unhandled lines:')
    for ln in unhandled_lines:
        print(ln)





                                                                                               
if __name__ == '__main__':                                                                     
    main()         
