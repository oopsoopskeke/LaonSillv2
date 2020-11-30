#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from LaonSill.ClientAPI import *
import os
import threading
import time
import cv2

def runFirstNetworkThread(handle1):
    print "sleep 15 seconds.."
    time.sleep(15)

    print "run ad-hoc network"
    TESTIMAGE_BASE_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "client",
            "test")

    for i in range(9):
        imagePath = os.path.join(TESTIMAGE_BASE_FILEPATH, "%d.jpg" % (i + 1))

        img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        height, width = img.shape[:2]
        converted_img = np.array(img).astype('f')

        ret, bboxes = handle1.runObjectDetectionWithInput(3, height, width,
                converted_img.flatten(), 20, 1, useAdhocRun=True)  # 3 for lenet

        print "ret : ", ret
        print "bboxes : ", bboxes
            
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyWindow('image')

# (1) LaonSill Client 핸들을 생성한다.
print "create handle"
handle1 = ClientHandle()
ret = handle1.createHandle()

handle2 = ClientHandle()
ret = handle2.createHandle()

# (2) 세션을 얻는다.
print "get session"
ret = handle1.getSession()
ret = handle2.getSession()

# (3) 네트워크를 생성한다.
NETWORK_FILEPATH2 = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "frcnn",
        "frcnn_union.json")
print "create network. network filepath : ", NETWORK_FILEPATH2
handle2.createNetworkFromFile(NETWORK_FILEPATH2)

handle1.setNetworkID(handle2.networkID.value)

# (4) 네트워크를 빌드한다.
print "build network (epoch=0)"
handle2.buildNetwork(0)

# (5) 1번 네트워크를 학습하는 쓰레드를 실행한다. 
run1stNetworkThread = threading.Thread(target=runFirstNetworkThread, args=(handle1,))
run1stNetworkThread.start()

# (6) 메인 쓰레드에서는 네트워크2를 학습한다.
print "run second network"
handle2.runNetwork(False)

# (7) 자원을 정리합니다.
print "cleanup resources"
handle2.destroyNetwork()

# (8) measure Thread 종료를 기다린다.
run1stNetworkThread.join()

handle1.releaseSession()
handle2.releaseSession()
