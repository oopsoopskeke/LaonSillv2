#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from LaonSill.ClientAPI import *

res = 512
#res = 416

#base_network = 0   # for SSD
base_network = 1   # for FRCNN     
#base_network = 2   # for YOLO

# (1) LaonSill Client 핸들을 생성한다.
print "create handle"
handle = ClientHandle()
ret = handle.createHandle()

# (2) 세션을 얻는다.
print "get session"
ret = handle.getSession()

# (3) 네트워크를 생성한다.
if base_network == 0:
    NETWORK_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "SSD",
        "ssd_512_infer_live.json")
elif base_network == 1:
    NETWORK_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "frcnn",
        "frcnn_test_live.json")
elif base_network == 2:
    NETWORK_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "YOLO",
        "yolo_test_live.json")
else:
    print 'wrong base network'
    handle.releaseSession()
    os.exit(0)

print "create network. network filepath : ", NETWORK_FILEPATH
handle.createNetworkFromFile(NETWORK_FILEPATH)

# (4) 네트워크를 빌드한다.
print "build network (epoch=1)"
handle.buildNetwork(1)

# (5) 오브젝트 디텍션 루프
TESTIMAGE_BASE_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "client", "test")
for i in range(19):
    imagePath = os.path.join(TESTIMAGE_BASE_FILEPATH, "%d.jpg" % (i + 1))

    img = cv2.imread(imagePath)
    height, width = img.shape[:2]

    resized_img = cv2.resize(img, (res, res))
    converted_img = np.array(resized_img).astype('f')

    ret, bboxes = handle.runObjectDetectionWithInput(3, res, res, converted_img.flatten(), 
            20, base_network)

    for bbox in bboxes:
        print "bbox : (", bbox[1], ", ",  bbox[0], ", ", bbox[3], ", ", bbox[2], ")"
        print "score : ", bbox[4], ", labelIndex : ", bbox[5]

        left = max(int(bbox[1] * width / res), 0)
        top = max(int(bbox[0] * height / res), 0)
        right = min(int(bbox[3] * width / res), width -1)
        bottom = min(int(bbox[2] * height / res), width - 1)

        cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 0), 2)
        
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

# (6) 자원을 정리합니다.
print "cleanup resources"
handle.destroyNetwork()
handle.releaseSession()
