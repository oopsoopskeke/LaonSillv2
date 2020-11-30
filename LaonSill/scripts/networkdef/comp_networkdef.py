#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""
2개의 network def 파일에 대해 어떤 차이점이 있는 지 비교하는 스크립트
"""

import argparse
import os
import sys
import json

#DEFAULT_DEF1_PATH = "$TOOLBOX_HOME/soosemi/networkdef/vgg16_train.json"
#DEFAULT_DEF2_PATH = "$TOOLBOX_HOME/soosemi/networkdef/vgg16_union.json"

both_union_case = False



def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="")
    """
    parser.add_argument("--def1-path", type=str, default=DEFAULT_DEF1_PATH, 
            help="")
    parser.add_argument("--def2-path", type=str, default=DEFAULT_DEF2_PATH, 
            help="")
    """
    parser.add_argument("--def1-path", type=str, required=True, 
            help="specify first network def file to compare.")
    parser.add_argument("--def2-path", type=str, required=True, 
            help="specify second network def file to compare.")

    return parser.parse_args()

def build_layers_dict(layers):
    layers_dict = dict()
    for layer in layers:
        layer_name = layer["name"]

        if layer_name not in layers_dict:
            layers_dict[layer_name] = list()

        layers_dict[layer_name].append(layer)

    return layers_dict

def comp_value(key, value1, value2):
    assert type(value1) == type(value2),\
            "'{}': type({}) of value1({}) is inconsistent with type({}) of value2({})".format(
                key, type(value1), value1, type(value2), value2)

    value_type = type(value1)
    if value_type is dict:
        comp_dict(key, value1, value2)
    elif value_type is list:
        comp_list(key, value1, value2)
    else:
        if value1 != value2:
            print("\t#'{}': value1 '{}' is inconsistent with value2 '{}'".format(
                key, value1, value2))


def comp_dict(key, dict1, dict2):
    for key1, value1 in dict1.iteritems():
        if key1 not in dict2:
            global both_union_case
            if both_union_case or key1 != "activation":
                print("\t#'{}': no '{}' key in dict2".format(key, key1))
        else:
            comp_value(key1, value1, dict2[key1])

def comp_list(key, list1, list2):
    if len(list1) != len(list2):
        print("\t#'{}': len({}) of list1({}) is inconsistent with len({}) of list2({})".format(
            key, len(list1), list1, len(list2), list2))
        return

    list1_sort = list1.sort()
    list2_sort = list2.sort()
    for idx in xrange(len(list1)):
        comp_value(key, list1[idx], list2[idx])



def comp_layers(layers1, type1, layers2, type2):
    print("::: Comparing Layers :::")
    layers_dict1 = build_layers_dict(layers1) 
    layers_dict2 = build_layers_dict(layers2) 

    for layer_name1, layer_dict1 in layers_dict1.iteritems():
        print("---> Comparing Layer {} ... ".format(layer_name1))

        if type1 == "union" and type2 != "union":
            for layer_dict1_item in layer_dict1:
                # 1에 2 대비 맞지 않는 activation 레이어가 있는 경우 무시.
                if "activation" in layer_dict1_item:
                    if ((type2 == "train" and layer_dict1_item["activation"] == "TestActivation") or
                            (type2 == "test" and layer_dict1_item["activation"] ==
                                "TrainActivation")):
                                print("\t({} for {} ... ignore ...)".format(
                                    layer_dict1_item["activation"], type2))
                                continue

                if layer_name1 not in layers_dict2:
                    print("\t# no '{}' layer in layer2".format(layer_name1))
                else:
                    assert len(layers_dict2[layer_name1]) == 1
                    comp_dict(layer_name1, layer_dict1_item, layers_dict2[layer_name1][0])
                    break

        elif type1 != "union" and type2 == "union":
            assert len(layer_dict1) == 1

            if layer_name1 not in layers_dict2:
                print("\t# no '{}' layer in layer2".format(layer_name1))
            else:
                layer_dict2 = layers_dict2[layer_name1]

                for layer_dict2_item in layer_dict2: 

                    if "activation" in layer_dict2_item:
                        if ((type1 == "train" and layer_dict2_item["activation"] == "TestActivation") or
                            (type1 == "test" and layer_dict2_item["activation"] == "TrainActivation")):
                                print("\t({} for {} ... ignore ...)".format(
                                    layer_dict2_item["activation"], type1))
                                continue

                    comp_dict(layer_name1, layer_dict1[0], layer_dict2_item)
                    break

        else:
            if layer_name1 not in layers_dict2:
                print("\t# no '{}' layer in layer2".format(layer_name1))
            else:
                layer_dict2 = layers_dict2[layer_name1]
                assert len(layer_dict1) == len(layer_dict2)

                for idx in xrange(len(layer_dict1)):
                    comp_dict(layer_name1, layer_dict1[idx], layer_dict2[idx])



    """
    for layer_name1, layer_dict1 in layers_dict1.iteritems():
        print("---> Comparing Layer {} ... ".format(layer_name1))
        if layer_name1 not in layers_dict2:
            print("\t# no '{}' layer in layer2".format(layer_name1))
        else:
            layer_dict2 = layers_dict2[layer_name1]

            if "activation" in layer_dict2:
                if ((type1 == "train" and layer_dict2["activation"] == "TestActivation") or
                    (type1 == "test" and layer_dict2["activation"] == "TrainActivation")):
                        print("\t({} for {} ... ignore ...)".format(
                            layer_dict2["activation"], type1))
                        continue

            comp_dict(layer_name1, layer_dict1, layer_dict2)
    """


def comp_configs(configs1, configs2):
    print("::: Comparing Configs :::")
    comp_dict("configs", configs1, configs2)

def comp_def(def1, type1, def2, type2):
    layers1 = def1["layers"]
    layers2 = def2["layers"]

    configs1 = def1["configs"]
    configs2 = def2["configs"]
    
    comp_layers(layers1, type1,  layers2, type2)
    comp_configs(configs1, configs2)

def read_def_type(def_dict):
    useCompositeModel = False
    status = "Train"

    if "configs" in def_dict: 
        if "useCompositeModel" in def_dict["configs"]:
            useCompositeModel = def_dict["configs"]["useCompositeModel"]

        if "status" in def_dict["configs"]:
            status = def_dict["configs"]["status"]

    if useCompositeModel:
        return "union"
    else:
        return status.lower()




def main():
    """Create the model and start the training."""
    args = get_arguments()

    def1_path = os.path.expandvars(args.def1_path)
    def2_path = os.path.expandvars(args.def2_path)

    def1_dict = json.loads(open(def1_path).read())
    def2_dict = json.loads(open(def2_path).read())

    def1_type = read_def_type(def1_dict)
    def2_type = read_def_type(def2_dict)

    print("def1_type = {}, def2_type = {}".format(def1_type, def2_type))

    assert (def1_type == def2_type) or\
            def1_type == "union" or\
            def2_type == "union",\
            "invalid def type config: def1_type={}, def2_type={}".format(
                def1_type, def2_type)

    global both_union_case
    if def1_type == "union" and def2_type == "union":
        both_union_case = True




    
    print("\n\n")
    print("1) Comparing <{}> with <{}>.".format(def1_path, def2_path))
    comp_def(def1_dict, def1_type, def2_dict, def2_type)

    print("\n\n")
    print("2) Comparing <{}> with <{}>.".format(def2_path, def1_path))
    comp_def(def2_dict, def2_type, def1_dict, def1_type)



if __name__ == '__main__':
    main()
