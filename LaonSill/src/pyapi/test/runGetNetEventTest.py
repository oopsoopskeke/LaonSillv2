#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from LaonSill.ClientAPI import *
import os
import time

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

# (5) 학습한다.
print "train network"
handle.runNetwork(False)

# (6) 네트워크 이벤트 정보를 얻는다.
print "get network events"
ret, events = handle.getNetworkEvent(handle.getNetworkID())
print "events : ", events

print "get network event messages"
ret, msgs = handle.getNetworkEventMsg(handle.getNetworkID())
print "event msgs : ", msgs

# (7) 학습이 중단된 네트워크를 정리한다.
print "destroy network"
handle.destroyNetwork()

# (8) 세션 자원을 정리합니다.
print "cleanup resources"
handle.releaseSession()
