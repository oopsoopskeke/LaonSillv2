#!/usr/bin/python
#
# FIXME: should check error but .... it is a very trivial program & I'm too lazy :)
#
import os
from shutil import copyfile

sourceMetaFilePath="/data/VOCdevkit/pascal_voc.txt"
targetMetaFilePath="/data/VOCdevkit/pascal_voc_multilabel_only.txt"

def preprocess():
    keywords = []

    sourceFile = open(sourceMetaFilePath, 'r')
    targetFile = open(targetMetaFilePath, 'w')

    lines = sourceFile.readlines()

    boxCount = 0
    for line in lines:
        if boxCount == 0:
            strLine = line[0:-1]
            split = strLine.split(' ')
            boxCount = int(split[1])
            imageFilePath = split[0]
            targetFile.write('%s ' % imageFilePath)
        else:
            boxCount = boxCount - 1
            strLine = line[0:-1]
            split = strLine.split(' ')

            targetFile.write('%s' % split[4])
            if boxCount == 0:
                targetFile.write('\n')
            else:
                targetFile.write(',')

    sourceFile.close()
    targetFile.close()
preprocess()
