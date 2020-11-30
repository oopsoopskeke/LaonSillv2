#!/usr/bin/python
#
# FIXME: should check error but .... it is a very trivial program & I'm too lazy :)
#
import os
from shutil import copyfile

espDirPath = "/data/ESP-ImageSet"
espLabelDirPath = "/data/ESP-ImageSet/LABELS"
espImageDirPath = "/data/ESP-ImageSet/images"
espKeywordsFilePath = "/data/ESP-ImageSet/top1000keywords.txt"

def preprocess():
    keywords = []

    fileKeyword = open(espKeywordsFilePath, 'w')
    fileNameList = os.listdir(espLabelDirPath)
    for fileName in fileNameList:
        labelFile = open(espLabelDirPath + "/" + fileName, 'r')
        lines = labelFile.readlines()

        for line in lines:
            keyword = line[0:-1]
            if keyword not in keywords:
                keywords.append(keyword)

        labelFile.close()

    for keyword in keywords:
        fileKeyword.write(keyword + '\n')

    fileKeyword.close()

    folderIdx = 0

    imageNameList = os.listdir(espImageDirPath)
    for imageName in imageNameList:
        folderPath = espDirPath + "/%.6d" % folderIdx
        os.mkdir(folderPath)
        copyfile(espImageDirPath + "/" + imageName, folderPath + "/" + imageName)
        copyfile(espLabelDirPath + "/" + imageName + ".desc", folderPath + "/keywords.txt")
        folderIdx = folderIdx + 1

preprocess()
