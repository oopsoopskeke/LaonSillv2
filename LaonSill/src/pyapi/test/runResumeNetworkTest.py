#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from LaonSill.ClientAPI import *
import os
import threading
import time

def runStopNetworkThread(networkID):
    handle = ClientHandle()
    ret = handle.createHandle()
    ret = handle.getSession()

    print "sleep 5 seconds"
    time.sleep(5)

    print "stop network"
    handle.stopNetwork(networkID)

    print "release session"
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
print "build network (epoch=0)"
handle.buildNetwork(0)

networkID = handle.getNetworkID()

# (5) 1번 네트워크를 학습하는 쓰레드를 실행한다. 
runStopNetworkThread = threading.Thread(target=runStopNetworkThread, args=(networkID,))
runStopNetworkThread.start()

# (6) 메인 쓰레드에서는 네트워크2를 학습한다.
print "train network"
handle.runNetwork(False)

# (7) stop network thread 종료를 기다린다.
runStopNetworkThread.join()

# (8) 학습이 중단된 네트워크를 정리한다.
print "destroy network"
handle.destroyNetwork()

# (9) 학습을 재개한다.
print "resume network"
handle.createResumeNetwork(networkID, keepHistory=1)

# (10) resume-network를 빌드한다.
print "build resume-network"
handle.buildNetwork(0)

# (10) resume-network을 학습한다.
print "run resume-network"
handle.runNetwork(False)

# (11) resume-network를 정리한다.
print "destory network"
handle.destroyNetwork()

# (12) 세션 자원을 정리합니다.
print "cleanup resources"
handle.releaseSession()
