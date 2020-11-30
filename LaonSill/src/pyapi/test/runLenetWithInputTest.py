#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from LaonSill.ClientAPI import *

rows = 28
cols = 28

# (1) LaonSill Client 핸들을 생성한다.
print "create handle"
handle = ClientHandle()
ret = handle.createHandle()

# (2) 세션을 얻는다.
print "get session"
ret = handle.getSession()

# (3) 네트워크를 생성한다.
NETWORK_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "LeNet",
        #"lenet_test_live.json")
        "lenet_union.json")

print "create network. network filepath : ", NETWORK_FILEPATH
ret = handle.createNetworkFromFile(NETWORK_FILEPATH)

# (4) 네트워크를 빌드한다.
print "build network (epoch=1)"
ret = handle.buildNetwork(1)

# (5) 이미지 분류 루프
TESTIMAGE_BASE_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "client", "test",
        "mnist")
for i in range(9):
    imagePath = os.path.join(TESTIMAGE_BASE_FILEPATH, "img_%d.jpg" % (i + 1))

    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]

    resized_img = cv2.resize(img, (rows, cols))
    converted_img = np.array(resized_img).astype('f')

    ret, label_index = handle.runClassificationWithInput(1, rows, cols,
            converted_img.flatten(), 3)  # 3 for lenet

    print "label_index : ", label_index
        
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

# (6) 자원을 정리합니다.
print "cleanup resources"
handle.destroyNetwork()
handle.releaseSession()

