#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import re
import math

from collections import OrderedDict

import pdb

DESCRIPTION = "Visualize LaonSill Training Process"



# Train 과정의 measure등을 log 파싱을 통해 그래프로 시각화
# 현재 average loss에 대해서만 그래프 그리고 있음 

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--parse-type", choices={'txt', 'csv'}, required=True, 
                        help="")
    parser.add_argument("--infile", type=str, required=True, 
                        help="input file which is one of txt and csv.")
    parser.add_argument("--outfile", type=str,
                        help="output intermediate csv file when txt is selected as a parse type.")
    parser.add_argument("--title", type=str, required=False, default=DESCRIPTION,
                        help="")
    parser.add_argument("--bound-upper", type=float, required=False, 
                        help="")
    parser.add_argument("--bound-lower", type=float, required=False, default=0.0,
                        help="")

    return parser.parse_args()


def get_candidate_logs(infile):
    infile = os.path.abspath(infile)
    split = os.path.split(infile)
    path = split[0] 
    basename = split[1]
    #print("path={}, basename={}".format(path, basename))

    prefixed = [filename for filename in os.listdir(path) if filename.startswith(basename)]
    prefixed.sort(key=alphanum_key)

    candidate_list = list()
    for prefix in prefixed:
        try:
            int(prefix.split(basename + ".")[-1])
            candidate_list.append(os.path.join(path, prefix))
        except ValueError:
            pass

    if os.path.isfile(infile):
        candidate_list.append(infile)

    return candidate_list


def parse_line(line):
    key = re.search('average loss\[(.+?)\]', line)
    assert key 
    key = key.group(1)
    value = float(line.split(":")[-1])
    return key, value



def load_data(infile, parse_type, bound_upper, bound_lower):
    infile_list = get_candidate_logs(infile)
    assert len(infile_list) > 0

    loss_dict = OrderedDict()
    loss_index = list()
    for infile_path in infile_list:
        # infile should exists
        assert os.path.isfile(infile_path), "infile not exists or not a file"
        assert '.{}'.format(parse_type) in infile_path, \
                "infile should be .{} file.".format(parse_type)

        if len(loss_dict) == 0:
            loss_index.append(0)
        else:
            loss_index.append(len(loss_dict.itervalues().next()))

        if parse_type == 'txt':
            logfile = open(infile_path, "r")
            loglines = logfile.readlines()
            logfile.close()

            for line in loglines:
                if 'average loss' in line:
                    key, value = parse_line(line) 
                    if bound_upper != None and value > bound_upper:
                        value = bound_upper
                    if value < bound_lower:
                        value = bound_lower

                    if key in loss_dict:
                        loss_dict[key].append(value)
                    else:
                        values = list()
                        values.append(value)
                        loss_dict[key] = values

        elif parse_type == 'csv':
            csvfile = open(infile_path, "r")
            csvlines = csvfile.readlines()
            csvfile.close()

            is_header = True
            keys = list()
            for line in csvlines:
                if is_header:
                    keys = line.split(",")
                    for key in keys:
                        values = list()
                        loss_dict[key] = values
                    is_header = False
                else:
                    values = line.split(",")
                    assert len(keys) == len(values)

                    for idx in xrange(len(keys)):
                        loss_dict[keys[idx]].append(values[idx])

    return loss_dict, loss_index

def out_csv_file(loss_list, outfile_path):
    keys = loss_dict.keys()
    num_entities = len(keys)

    outfile = open(outfile_path, "w")
    len_values = len(loss_list[0])

    # write header line
    for idx in xrange(num_entities):
        if idx < num_entities - 1:
            outfile.write("{},".format(keys[idx]))
        else:
            outfile.write("{}\n".format(keys[idx]))

    # write value lines
    for idx in xrange(len_values):
        for key_idx in xrange(num_entities):
            if key_idx < num_entities - 1:
                outfile.write("{},".format(loss_list[key_idx][idx]))
            else:
                outfile.write("{}\n".format(loss_list[key_idx][idx]))

    outfile.close()


def draw_graph(title, keys, loss_list, loss_index):
    num_entities = len(keys)

    fig = plt.figure()
    fig.suptitle(title)

    major = int(math.ceil(math.sqrt(num_entities)))
    minor = 0
    for idx in xrange(major):
        if major * (idx + 1) >= num_entities:
            minor = idx + 1
            break

    graphs = list()
    for idx in xrange(num_entities):
        graphs.append(fig.add_subplot(major, minor, idx + 1))
        graphs[idx].set_ylabel(keys[idx])
        graphs[idx].plot(loss_list[idx], 'r')

    plt.show()





def main():
    args = get_arguments()

    parse_type = args.parse_type
    infile_path = args.infile
    outfile_path = args.outfile
    title = args.title
    bound_upper = args.bound_upper
    bound_lower = args.bound_lower

    support_outfile = False
    if parse_type == 'txt':
        if outfile_path != None:
            #assert os.path.exists(outfile_path), "outfile not exists."
            support_outfile = True
    elif parse_type == 'csv':
        assert outfile_path == None, "outfile not supported."

    loss_dict, loss_index = load_data(infile_path, parse_type, bound_upper, bound_lower)

    keys = loss_dict.keys()
    num_entities = len(keys)
    loss_list = list()
    for _, values in loss_dict.iteritems():
        loss_list.append(values)

    if support_outfile:
        out_csv_file(loss_list, outfile_path)

    draw_graph(title, keys, loss_list, loss_index)



def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]



if __name__ == '__main__':
    main()


