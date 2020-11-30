#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from LaonSill.ClientAPI import *
import os
import threading
import time

def runFirstNetworkThread(handle1):
    print "run first network"
    handle1.runNetwork(False)

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
NETWORK_FILEPATH1 = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "LeNet",
        "lenet_train.json")
NETWORK_FILEPATH2 = os.path.join(os.environ["LAONSILL_SOURCE_PATH"], "examples", "SSD",
        "ssd_512_train.json")

print "create network1. network filepath : ", NETWORK_FILEPATH1
print "create network2. network filepath : ", NETWORK_FILEPATH2
handle1.createNetworkFromFile(NETWORK_FILEPATH1)
handle2.createNetworkFromFile(NETWORK_FILEPATH2)

# (4) 네트워크를 빌드한다.
print "build network (epoch=0)"
handle1.buildNetwork(0)
handle2.buildNetwork(0)

# (5) 1번 네트워크를 학습하는 쓰레드를 실행한다. 
run1stNetworkThread = threading.Thread(target=runFirstNetworkThread, args=(handle1,))
run1stNetworkThread.start()

# (6) 메인 쓰레드에서는 네트워크2를 학습한다.
print "run second network"
handle2.runNetwork(False)

# (7) measure Thread 종료를 기다린다.
run1stNetworkThread.join()

# (8) 자원을 정리합니다.
print "cleanup resources"
handle1.destroyNetwork()
handle2.destroyNetwork()
handle1.releaseSession()
handle2.releaseSession()
