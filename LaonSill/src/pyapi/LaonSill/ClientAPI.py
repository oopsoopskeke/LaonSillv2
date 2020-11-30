# -*- coding: utf-8 -*-

from ctypes import *
libLaonSill = CDLL('libLaonSillClient.so.1.0.1')

MAX_MESAURE_ITEMCOUNT=20
MAX_MEASURE_ITEMNAMELEN=64

# libLaonSillClient에서 제공하는 함수들을 준비한다.
funcTestYo = libLaonSill.testYo
funcTestYo.argtypes = [c_int, c_char_p, c_float]

funcGetSession = libLaonSill.getSession
funcGetSession.argtypes = [POINTER(c_int), c_char_p, c_int, POINTER(c_int), c_char_p]

funcReleaseSession = libLaonSill.releaseSession
funcReleaseSession.argtypes = [c_int, c_char_p, POINTER(c_int)]

funcCreateNetwork = libLaonSill.createNetwork
funcCreateNetwork.argtypes = [c_int, c_int, c_char_p, c_char_p, c_char_p]

funcCreateNetworkFromFile = libLaonSill.createNetworkFromFile
funcCreateNetworkFromFile.argtypes = [c_int, c_int, c_char_p, c_char_p, c_char_p]

funcCreateResumeNetwork = libLaonSill.createResumeNetwork
funcCreateResumeNetwork.argtypes = [c_int, c_int, c_char_p, c_char_p, c_int, c_char_p]

funcStopNetwork = libLaonSill.stopNetwork
funcStopNetwork.argtypes = [c_int, c_int, c_char_p, c_char_p]

funcDestroyNetwork = libLaonSill.destroyNetwork
funcDestroyNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p]

funcBuildNetwork = libLaonSill.buildNetwork
funcBuildNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]

funcResetNetwork = libLaonSill.resetNetwork
funcResetNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p]

funcRunNetwork = libLaonSill.runNetwork
funcRunNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]

funcRunNetworkMiniBatch = libLaonSill.runNetworkMiniBatch
funcRunNetworkMiniBatch.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int]

funcSaveNetwork = libLaonSill.saveNetwork
funcSaveNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_char_p]

funcLoadNetwork = libLaonSill.loadNetwork
funcLoadNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_char_p]


class BoundingBox(Structure):
    _fields_ = [("top", c_float), ("left", c_float), ("bottom", c_float), ("right", c_float),
                ("confidence", c_float), ("class_id", c_int)]

funcGetObjectDetection = libLaonSill.getObjectDetection
funcGetObjectDetection.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int,
                                   c_int, POINTER(c_float), c_void_p, c_int, c_int]


funcRunObjectDetectionWithInput = libLaonSill.runObjectDetectionWithInput
funcRunObjectDetectionWithInput.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int,
                                   c_int, POINTER(c_float), c_void_p, c_int, c_int, c_int]


funcRunClassificationWithInput = libLaonSill.runClassificationWithInput
funcRunClassificationWithInput.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int,
                                   c_int, POINTER(c_float), c_int, c_int, c_int, 
                                   POINTER(c_int), POINTER(c_int), POINTER(c_float)]

funcGetMeasureItemName = libLaonSill.getMeasureItemName
funcGetMeasureItemName.argtypes = [c_int, c_char_p, c_char_p, c_int,
    POINTER(POINTER(c_char)), POINTER(c_int)]

funcGetMeasures = libLaonSill.getMeasures
funcGetMeasures.argtypes = [c_int, c_char_p, c_char_p, c_int, c_int, c_int, POINTER(c_int),
    POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_float)]

funcGetNetworkProgress = libLaonSill.getNetworkProgress
funcGetNetworkProgress.argtypes = [c_int, c_char_p, c_char_p, POINTER(c_int), POINTER(c_int)]

funcGetNetworkResult = libLaonSill.getNetworkResult
funcGetNetworkResult.argtypes = [c_int, c_char_p, c_char_p, c_int, POINTER(c_int),
    POINTER(POINTER(c_char)), POINTER(c_float)]

class NetEvent(Structure):
    _fields_ = [("eventType", c_int), ("eventTime", c_char * 20), ("layerID", c_int),
        ("message", c_char * 2048)]

funcGetNetworkEvent = libLaonSill.getNetworkEvent
funcGetNetworkEvent.argtypes = [c_int, c_char_p, c_char_p, c_int, c_void_p]

funcGetNetworkEventMsg = libLaonSill.getNetworkEventMsg
funcGetNetworkEventMsg.argtypes = [c_int, c_char_p, c_char_p, c_int, c_int,
    POINTER(POINTER(c_char)), POINTER(c_int)]

funcCheckNetworkDef = libLaonSill.checkNetworkDef
funcCheckNetworkDef.argtypes = [c_int, c_char_p, c_char_p, c_int, POINTER(c_int),
    POINTER(c_int), POINTER(c_int), c_char_p]

class ClientHandle:
    def __init__(self):
        self.hasSession = c_int(0)
        self.bufLen = 1452      # see LaonSill MessageHeader.h source
        self.sockFD = c_int(-1)
        self.serverHostName = c_char_p("localhost")
        self.serverPortNum = c_int(20088)

        self.networkID = create_string_buffer(37)   # uuid size : 32 + 4 + 1
        self.isCreated = c_int(0)

    def createHandle(self, serverHostName = "localhost", serverPortNum=20088):
        self.serverHostName = c_char_p(serverHostName)
        self.serverPortNum = c_int(serverPortNum)
        self.buffer = create_string_buffer(self.bufLen)
        return 0    # success

    def getSession(self):
        ret = funcGetSession(byref(self.hasSession), self.serverHostName,
            self.serverPortNum, byref(self.sockFD), self.buffer)
        return ret

    def setNetworkID(self, networkIDVal):
        self.networkID = c_char_p(networkIDVal)
        self.isCreated = c_int(1)

    def getNetworkID(self):
        return self.networkID.value

    def setSession(self, networkID):
        self.networkID = c_char_p(networkID)
        self.isCreated = c_int(1)

    def releaseSession(self):
        ret = funcReleaseSession(self.sockFD, self.buffer, byref(self.hasSession))
        return ret

    def createNetwork(self, networkDef): 
        ret = funcCreateNetwork(self.sockFD, self.hasSession, self.buffer,
                c_char_p(networkDef), self.networkID)
        self.isCreated = c_int(1)
        return ret

    def createNetworkFromFile(self, filePathInServer):
        ret = funcCreateNetworkFromFile(self.sockFD, self.hasSession, self.buffer,
                c_char_p(filePathInServer), self.networkID)
        self.isCreated = c_int(1)
        return ret

    def createResumeNetwork(self, networkID, keepHistory=0):
        ret = funcCreateResumeNetwork(self.sockFD, self.hasSession, self.buffer,
                c_char_p(networkID), c_int(keepHistory), self.networkID)
        self.isCreated = c_int(1)
        return ret

    def stopNetwork(self, networkID):
        # XXX: 다른 세션에서 네트워크를 끌 수 있다면 이제 isCreated라는 플래그는 의미가
        #       없어진다. 추후에 정리하자.
        ret = funcStopNetwork(self.sockFD, self.hasSession, self.buffer, c_char_p(networkID))
        self.isCreated = c_int(0)
        return ret

    def destroyNetwork(self):
        ret = funcDestroyNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID)
        self.isCreated = c_int(0)
        return ret

    def buildNetwork(self, epochs):
        ret = funcBuildNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID,
                c_int(epochs))
        return ret

    def resetNetwork(self):
        ret = funcResetNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID)
        return ret

    def runNetwork(self, inference):
        ret = funcRunNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID,
                c_int(inference))
        return ret

    def runNetworkMiniBatch(self, inference, miniBatchIdx):
        ret = funcRunNetworkMiniBatch(self.sockFD, self.buffer, self.isCreated,
                self.networkID, c_int(inference), c_int(miniBatchIdx))
        return ret

    def saveNetwork(self, filePath):
        ret = funcSaveNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID,
                c_char_p(filePath))
        return ret

    def loadNetwork(self, filePath):
        ret = funcLoadNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID,
                c_char_p(filePath))
        return ret

    def getObjectDetection(self, channel, height, width, imageData, maxBoxCount,
            coordRelative):
        imageDataArray = (c_float * len(imageData))(*imageData)
        bboxArray = (BoundingBox * maxBoxCount)()

        ret = funcGetObjectDetection(self.sockFD, self.buffer, self.isCreated, 
                self.networkID, c_int(channel), c_int(height), c_int(width),
                imageDataArray, bboxArray, c_int(maxBoxCount), c_int(coordRelative))

        result_box = []
        for bbox in bboxArray:
            if bbox.confidence > 0.000001:
                result_box.append([bbox.top, bbox.left, bbox.bottom, bbox.right,
                        bbox.confidence, bbox.class_id])
        return ret, result_box

    def runObjectDetectionWithInput(self, channel, height, width, imageData, maxBoxCount,
            networkType, useAdhocRun=False):
        imageDataArray = (c_float * len(imageData))(*imageData)
        bboxArray = (BoundingBox * maxBoxCount)()

        if useAdhocRun == True:
            useAdhocVal = 1
        else:
            useAdhocVal = 0

        ret = funcRunObjectDetectionWithInput(self.sockFD, self.buffer, self.isCreated, 
                self.networkID, c_int(channel), c_int(height), c_int(width),
                imageDataArray, bboxArray, c_int(maxBoxCount), c_int(networkType),
                c_int(useAdhocVal))

        result_box = []
        for bbox in bboxArray:
            if bbox.confidence > 0.000001:
                result_box.append([bbox.top, bbox.left, bbox.bottom, bbox.right,
                        bbox.confidence, bbox.class_id])
        return ret, result_box

    def runClassificationWithInput(self, channel, height, width, imageData, networkType,
            maxResultCount=5, useAdhocRun=False):

        sortedLabelIndexes = (c_int * maxResultCount)()
        sortedScores = (c_float * maxResultCount)()

        imageDataArray = (c_float * len(imageData))(*imageData)
        result = []
        result_count = c_int(-1)

        if useAdhocRun == True:
            useAdhocVal = 1
        else:
            useAdhocVal = 0

        ret = funcRunClassificationWithInput(self.sockFD, self.buffer, self.isCreated, 
                self.networkID, c_int(channel), c_int(height), c_int(width),
                imageDataArray, c_int(networkType), c_int(maxResultCount), 
                c_int(useAdhocVal), byref(result_count), sortedLabelIndexes, sortedScores)

        for i in range(result_count.value):
            result.append([sortedLabelIndexes[i], sortedScores[i]])

        return ret, result

    def getMeasureItemName(self, networkID):
        itemCount = c_int(-1)        
        itemNameArray = (POINTER(c_char) * MAX_MESAURE_ITEMCOUNT)()
        for i in range(MAX_MESAURE_ITEMCOUNT):
            itemNameArray[i] = create_string_buffer(MAX_MEASURE_ITEMNAMELEN)
        
        ret = funcGetMeasureItemName(self.sockFD, self.buffer, c_char_p(networkID),
                c_int(MAX_MESAURE_ITEMCOUNT), itemNameArray, byref(itemCount))

        if ret != 0:
            return ret, []

        result = []
        for i in range(itemCount.value):
            value = cast(itemNameArray[i], c_char_p).value
            result.append(value);

        return ret, result

    def getMeasures(self, networkID, itemCount, forwardSearch, start, count):
        assert itemCount > 0

        startIterNum = c_int(-1)
        dataCount = c_int(-1)
        curIterCount = c_int(-1)
        totalIterCount = c_int(-1)
        measureArray = (c_float * (itemCount * count))()
       
        ret = funcGetMeasures(self.sockFD, self.buffer, c_char_p(networkID),
                c_int(int(forwardSearch)), c_int(start), c_int(count), byref(startIterNum),
                byref(dataCount), byref(curIterCount), byref(totalIterCount), measureArray)

        if ret != 0:
            ret, -1, []

        result = []

        for i in range(dataCount.value / itemCount):
            curr_result = []
            for j in range(itemCount):
                index = i * itemCount + j
                curr_result.append(measureArray[index])
            result.append(curr_result)

        return ret, startIterNum.value, curIterCount.value, totalIterCount.value, result

    def getNetworkEvent(self, networkID, maxEventCount=1000):
        netEvents = (NetEvent * maxEventCount)()

        ret = funcGetNetworkEvent(self.sockFD, self.buffer, c_char_p(networkID),
                c_int(maxEventCount), netEvents)

        if ret != 0:
            return ret, []

        result = []
        for netEvent in netEvents:  
            if netEvent.eventType >= 0:
                event = [netEvent.eventType, netEvent.eventTime, netEvent.layerID,\
                        netEvent.message]
                result.append(event)

        return ret, result

    def getNetworkEventMsg(self, networkID, maxEventCount=1000, maxMessageLen=2048):
        eventCount = c_int(-1)        
        eventMsgs = (POINTER(c_char) * maxEventCount)()
        for i in range(maxEventCount):
            eventMsgs[i] = create_string_buffer(maxMessageLen)
        
        ret = funcGetNetworkEventMsg(self.sockFD, self.buffer, c_char_p(networkID),
                c_int(maxEventCount), c_int(maxMessageLen), eventMsgs, byref(eventCount))

        if ret != 0:
            return ret, []

        result = []
        for i in range(eventCount.value):
            value = cast(eventMsgs[i], c_char_p).value
            result.append(value);

        return ret, result

    def checkNetworkDef(self, networkDef, maxMessageLen=2048):
        resultCode = c_int(-1)
        gpuMBSize = c_int(-1)
        layerID = c_int(-1)
        msg = create_string_buffer(maxMessageLen) 

        ret = funcCheckNetworkDef(self.sockFD, self.buffer, c_char_p(networkDef),
                c_int(maxMessageLen), byref(resultCode), byref(gpuMBSize), byref(layerID),
                msg)

        return ret, resultCode.value, gpuMBSize.value, layerID.value, msg.value

    def getNetworkProgress(self, networkID): 
        curIterCount = c_int(-1)
        totalIterCount = c_int(-1)

        ret = funcGetNetworkProgress(self.sockFD, self.buffer, c_char_p(networkID),
                byref(curIterCount), byref(totalIterCount))

        return ret, curIterCount.value, totalIterCount.value

    def getNetworkResult(self, networkID, maxResultCount=20):
        resultCount = c_int(-1)
        itemResultArray = (c_float * maxResultCount)()
        itemNameArray = (POINTER(c_char) * MAX_MESAURE_ITEMCOUNT)()
        for i in range(maxResultCount):
            itemNameArray[i] = create_string_buffer(MAX_MEASURE_ITEMNAMELEN)

        ret = funcGetNetworkResult(self.sockFD, self.buffer, c_char_p(networkID),
                c_int(maxResultCount), byref(resultCount), itemNameArray, itemResultArray)

        result = []
        for i in range(resultCount.value):
            item = [cast(itemNameArray[i], c_char_p).value, itemResultArray[i]]
            result.append(item)

        return ret, result

