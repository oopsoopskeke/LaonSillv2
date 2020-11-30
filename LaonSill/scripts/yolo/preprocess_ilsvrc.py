#!/usr/bin/python

import os

ILSVRC_ROOT_PATH = "/data/ilsvrc12_train/"
ILSVRC_CLASS_FILENAME = "map_clsloc.txt"
ILSVRC_META_FILENAME = "ilsvrc.txt"

classDic = dict()

classFilePath = os.path.join(ILSVRC_ROOT_PATH, ILSVRC_CLASS_FILENAME)
f = open(classFilePath, "r")
lines = f.readlines()

for line in lines:
    elems = line.split(' ') 
    folderName = elems[0]
    classID = elems[1]
    classDic[folderName] = classID

f.close()

metaFilePath = os.path.join(ILSVRC_ROOT_PATH, ILSVRC_META_FILENAME)
f = open(metaFilePath, "w")

isFirst = True

for folderName in classDic:
    classID = classDic[folderName]
    dirName = os.path.join(ILSVRC_ROOT_PATH, 'train', folderName)

    fileNames = os.listdir(dirName)
    for fileName in fileNames:
        if "JPEG" in fileName:
            if isFirst:
                isFirst = False
            else:
                f.write('\n')
            filePath = dirName + "/" + fileName
            f.write("%s %d" % (filePath, int(classID) - 1))

f.close()
