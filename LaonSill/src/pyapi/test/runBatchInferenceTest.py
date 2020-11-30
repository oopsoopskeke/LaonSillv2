#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from LaonSill.ClientAPI import *
import os
import threading
import time

def runGetNetworkProgressThread(networkID):
    handle = ClientHandle()
    ret = handle.createHandle()
    ret = handle.getSession()

    print "sleep 3 seconds"
    time.sleep(3)


    while True:
        ret, curIter, totalIter = handle.getNetworkProgress(networkID)
        print "progress : %d/%d" % (curIter + 1, totalIter)

        if curIter + 1 == totalIter:
            break

    ret, result = handle.getNetworkResult(networkID)
    print "network result : ", result

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
        "lenet_batch_inference.json")

print "create network. network filepath : ", NETWORK_FILEPATH
handle.createNetworkFromFile(NETWORK_FILEPATH)

# (4) 네트워크를 빌드한다.
print "build network (epoch=0)"
handle.buildNetwork(0)

networkID = handle.getNetworkID()

# (5) 1번 네트워크를 학습하는 쓰레드를 실행한다. 
getNetworkProgressThread = threading.Thread(target=runGetNetworkProgressThread, args=(networkID,))
getNetworkProgressThread.start()

# (6) 메인 쓰레드에서는 네트워크를 inference 한다.
print "train network"
handle.runNetwork(True)

# (7) get network progress thread 종료를 기다린다.
getNetworkProgressThread.join()

# (8) 학습이 중단된 네트워크를 정리한다.
print "destroy network"
handle.destroyNetwork()

# (9) 세션 자원을 정리합니다.
print "cleanup resources"
handle.releaseSession()
