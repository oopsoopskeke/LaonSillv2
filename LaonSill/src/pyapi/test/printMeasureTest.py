#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from LaonSill.ClientAPI import *
import os
import threading
import time


def printMeasureThread(networkID):
    print "network ID : ", networkID
    print "[Measure Thread] create handle"
    handle = ClientHandle()
    ret = handle.createHandle()

    print "[Measure Thread] get session"
    ret = handle.getSession()

    ret, itemNames = handle.getMeasureItemName(networkID)
    print "[Measure Thread] item names : ", itemNames

    print "[Measure Thread] (1) forward search "
    for i in range(5):
        ret, startIterNum, curIterNum, totalIterNum, measures = \
            handle.getMeasures(networkID, len(itemNames), True, i * 10, 10)
        for j in range(len(measures)):
            print "[", (startIterNum + j), "] : ", measures[j]
        print "progress : %d/%d" % (curIterNum, totalIterNum)
        print ""

    print "[Measure Thread] (2) backward search "
    for i in range(5):
        ret, startIterNum, curIterNum, totalIterNum, measures = \
            handle.getMeasures(networkID, len(itemNames), False, -1, 10)
        for j in range(len(measures)):
            print "[", (startIterNum + j), "] : ", measures[j]
        print "progress : %d/%d" % (curIterNum, totalIterNum)
        print ""

    print "[Measure Thread] release session"
    handle.releaseSession()

# (1) LaonSill Client 핸들을 생성한다.
print "create handle"
handle = ClientHandle()
ret = handle.createHandle()

# (2) 세션을 얻는다.
print "get session"
ret = handle.getSession()

# (3) 네트워크를 생성한다.
NETWORK_FILEPATH = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "LeNet",
        "lenet_train.json")
print "create network. network filepath : ", NETWORK_FILEPATH
handle.createNetworkFromFile(NETWORK_FILEPATH)

# (4) 네트워크를 빌드한다.
print "build network (epoch=8)"
handle.buildNetwork(8)

# (5) measure Thread를 실행한다.
measureThread = threading.Thread(target=printMeasureThread, args=(handle.networkID.value,))
measureThread.start()

# (6) 메인 쓰레드에서는 네트워크를 학습한다.
handle.runNetwork(False)

# (7) measure Thread 종료를 기다린다.
measureThread.join()

# (8) 자원을 정리합니다.
print "cleanup resources"
handle.destroyNetwork()
handle.releaseSession()
