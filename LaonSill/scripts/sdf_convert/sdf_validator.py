#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import cv2
import numpy as np
import shutil
import hashlib
import time
import pudb

LAONSILL_CLIENT_LIB_PATH = os.path.join(os.environ['LAONSILL_HOME'], 'dev/client/lib')
sys.path.append(LAONSILL_CLIENT_LIB_PATH)
from libLaonSillClient import *


SDF_PATH = "$LAONSILL_HOME/data/sdf"
SUMMARY_FILE = "summary.txt"

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--mode", choices={"SUM", "VAL"}, default="VAL", help="")
    parser.add_argument("--sdf-path", type=str, default=SDF_PATH, help="")
    parser.add_argument("--summary-file", type=str, default=SUMMARY_FILE, help="")

    return parser.parse_args()

def hash_bytestr_iter(bytesiter, hasher, ashexstr=False):
    for block in bytesiter:
        hasher.update(block)
    return (hasher.hexdigest() if ashexstr else hasher.digest())

def file_as_blockiter(afile, blocksize=65536):
    size = os.fstat(afile.fileno()).st_size
    processed = 0
    with afile:
        block = afile.read(blocksize)
        while len(block) > 0:
            yield block
            processed += len(block)
            print("processed ({} / {})".format(processed, size)) 
            block = afile.read(blocksize)

def summary(sdfPath, summaryFilePath):
    sdfDirs = [dir for dir in os.listdir(sdfPath) \
            if os.path.isdir(os.path.join(sdfPath, dir)) and \
            os.path.exists(os.path.join(sdfPath, dir, "data.sdf"))]
    print(sdfDirs)

    summaryFile = open(summaryFilePath, "w")    
    for sdfDir in sdfDirs:
        start_time = time.time()
        name = sdfDir
        path = os.path.join(sdfPath, sdfDir, "data.sdf")
        size = os.stat(path).st_size
        # 아래의 방법은 큰 파일에 대해 메모리 문제가 있어 그 다음의 방법으로 대체
        #checksum = hashlib.md5(open(path, "rb").read()).hexdigest()
        checksum = hash_bytestr_iter(file_as_blockiter(open(path, 'rb'), 1073741824), hashlib.md5(), True) 

        print("name->{}, size->{}, checksum->{}, time->{}".format(\
                name, size, checksum, time.time() - start_time))
        summaryFile.write("{},{},{}\n".format(name, size, checksum))
    summaryFile.close()



def validate(sdfBasePath, summaryFilePath):
    sdfDirs = [dir for dir in os.listdir(sdfBasePath) \
            if os.path.isdir(os.path.join(sdfBasePath, dir)) and \
            os.path.exists(os.path.join(sdfBasePath, dir, "data.sdf"))]

    summarySizeDict = None
    summaryHashDict = None
    if os.path.exists(summaryFilePath):
        summaryFile = open(summaryFilePath, "r")
        lines = summaryFile.readlines()
        summaryFile.close()

        summarySizeDict = dict()
        summaryHashDict = dict()
        for line in lines:
            line = line.strip()
            parts = line.split(",")
            assert len(parts) == 3, "invalid summary line: {}".format(line)
            summarySizeDict[parts[0]] = int(parts[1])
            summaryHashDict[parts[0]] = parts[2]


    for sdfDir in sdfDirs:
        print("----------> validate {}".format(sdfDir))
        sdfPath = os.path.join(sdfBasePath, sdfDir)
        sdfDataPath = os.path.join(sdfPath, "data.sdf")

        print("\t> validate file size for {}".format(sdfDir))
        size = os.stat(sdfDataPath).st_size
        assert size == summarySizeDict[sdfDir], \
                "size inconsistent for file: {}".format(sdfDir)
        print("\t...ok.")

        print("\t> validate file md5 checksum for {}".format(sdfDir))
        checksum = hash_bytestr_iter(file_as_blockiter(open(sdfDataPath, 'rb'), 1073741824), \
                hashlib.md5(), True) 
        assert checksum == summaryHashDict[sdfDir], \
                "checksum inconsistent for file: {}".format(sdfDir)
        print("\t...ok.")

        header = RetrieveSDFHeader(sdfPath)
        sdfType = header.type

        dataReader = None
        if sdfType == 'DATUM':
            dataReader = DataReaderDatum(sdfPath)
        elif sdfType == 'ANNO_DATUM':
            dataReader = DataReaderAnnoDatum(sdfPath)
        header = dataReader.getHeader()

        print("\t> validate header info consistence for {}".format(sdfDir))
        assert (len(header.names) == header.numSets) and \
                (len(header.setSizes) == header.numSets) and \
                (len(header.setStartPos) == header.numSets), \
            "invalid header. numSets and header meta data are inconsistent."
        print("\t...ok.")

        print("\t> validate set sizes for {}".format(sdfDir))
        for i in range(header.numSets):
            assert header.setSizes[i] > 0, \
                "one of data set size in SDF is 0."
        print("\t...ok.")

        print("\t> validate images for {}".format(sdfDir))
        for dataSetIdx in xrange(header.numSets):
            dataReader.selectDataSetByIndex(dataSetIdx)
            for imageIdx in xrange(header.setSizes[dataSetIdx]):
                dataReader.getNextData()
        print("\t...ok.")





def main():
    """Create the model and start the training."""
    args = get_arguments()

    mode = args.mode
    sdf_path = os.path.expandvars(args.sdf_path)
    summary_file = os.path.expanduser(args.summary_file)
    assert os.path.exists(sdf_path), "sdf-path not exists: {}".format(sdf_path)

    if mode == "SUM":
        summary(sdf_path, summary_file)
    elif mode == "VAL":
        validate(sdf_path, summary_file)
    

if __name__ == '__main__':
    main()



