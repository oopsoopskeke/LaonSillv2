#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from LaonSill.ClientAPI import *

rows = 224
cols = 224

base_network = 0   # for VGG16
#base_network = 1    # for INCEPTION_V3
#base_network = 2   # for RESNET50


# (1) LaonSill Client 핸들을 생성한다.
print "create handle"
handle = ClientHandle()
ret = handle.createHandle()

# (2) 세션을 얻는다.
print "get session"
ret = handle.getSession()

# (3) 네트워크를 생성한다.
if base_network == 0:
    NETWORK_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "VGG16",
        "vgg16_test_live.json")
elif base_network == 1:
    NETWORK_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "Inception",
        "inception_v3_test_live.json")
    rows = 299
    cols = 299
elif base_network == 2:
    NETWORK_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "ResNet",
        "resnet50_test_live.json")
else:
    print 'wrong base network'
    handle.releaseSession()
    os.exit(0)

print "create network. network filepath : ", NETWORK_FILEPATH
ret = handle.createNetworkFromFile(NETWORK_FILEPATH)

# (4) 네트워크를 빌드한다.
print "build network (epoch=1)"
ret = handle.buildNetwork(1)

# (5) 이미지 분류 루프
TESTIMAGE_BASE_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "client", "test")
for i in range(7):
    imagePath = os.path.join(TESTIMAGE_BASE_FILEPATH, "%d.jpg" % (i + 1))

    img = cv2.imread(imagePath)
    height, width = img.shape[:2]

    resized_img = cv2.resize(img, (rows, cols))
    converted_img = np.array(resized_img).astype('f')

    baseNetworkType = base_network     # 0 for VGG16, 1 for Inception, 2 for ResNet
    ret, result = handle.runClassificationWithInput(3, rows, cols,
            converted_img.flatten(), baseNetworkType)

    print "result : ", result
        
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

# (6) 자원을 정리합니다.
print "cleanup resources"
handle.destroyNetwork()
handle.releaseSession()

