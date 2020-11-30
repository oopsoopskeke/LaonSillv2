#!/usr/bin/env python

"""decodeHotCode.py: """

import json
import sys
import os
import struct

def checkEventProperty(hotCodeDic, hotCode, propertyName):
    if not propertyName in hotCodeDic[hotCode]:
        print "ERROR: hotCode %s does not have %s property" % (hotCode, propertyName)
        exit(-1)

# XXX:  we only considers Linux 64bit platform.

# XXX: python does not support long double format..
typeFmtDic = {\
    "UINT8" : "B", "INT8" : "b",\
    "UINT16" : "H", "INT16" : "h",\
    "UINT32" : "I", "INT32" : "i",\
    "UINT64" : "Q", "INT64" : "q",\
    "BOOL" : "?", "FLOAT" : "f",\
    "DOUBLE" : "d", "LONGDOUBLE" : "d",\
}

def getValueSize(typeStr):
    if typeStr in ["UINT8", "INT8"]:
        return 1
    elif typeStr in ["UINT16", "INT16"]:
        return 2
    elif typeStr in ["UINT32", "INT32"]:
        return 4
    elif typeStr in ["UINT64", "INT64"]:
        return 8
    elif typeStr in ["FLOAT"]:
        return 4
    elif typeStr in ["DOUBLE"]:
        return 8
    elif typeStr in ["LONGDOUBLE"]:
        return 16
    elif typeStr in ["BOOL"]:
        return 1
    return 0

def decodeFile(srcFileName, hotCodeDic):
    try:
        srcFile = open(srcFileName, 'rb')
        eventCount = 0
        print '[ decode ' + srcFileName + ' starts ]'
        print "================================================"

        while True:
            chunk = srcFile.read(4)
            if chunk == '':
                break

            codeId = struct.unpack('i', chunk)[0]
            if codeId == 0:
                chunk = srcFile.read(4)
                failCount = struct.unpack('i', chunk)[0]
                print "================================================"
                print " - event count=%d" % eventCount
                print " - fail count=%d" % failCount
                print "================================================\n"
                break

            eventCount = eventCount + 1
            hotCode = str(codeId)
            if hotCode not in hotCodeDic:
                print 'ERROR: hotcode (%s) is not defined in hotCodeDic' % hotCode
                exit(-1)

            paramList = hotCodeDic[hotCode]['ARGS']
            paramSize = 0
            t = ()
            for param in paramList:
                arrayString = ""
                foundNull = False

                if param in typeFmtDic:
                    paramSize = int(getValueSize(param))
                    fmt = typeFmtDic[param]
                elif "CHAR" in param:
                    # XXX: needs error-check
                    arrayCount = int(param.replace(")", "$").replace("(", "$").split("$")[1])
                    paramSize = arrayCount
                    fmt = '%ds' % paramSize
                else:
                    print "ERROR: invalid hotCode type(%s) for hotCode(%s)" % (param, hotCode)
                    exit(-1)

                chunk = srcFile.read(paramSize)
                if chunk == '':
                    print 'ERROR: data is truncated'
                    exit(-1)

                # change fmt if there is '\0' middle of chunk
                if "CHAR" in param:
                    nullOffset = 0
                    for char in chunk:
                        if char == '\0':
                            fmt = '%ds' % nullOffset
                            foundNull = True
                            break
                        nullOffset = nullOffset + 1
              
                if foundNull == True:
                    t = t + struct.unpack(fmt, chunk[:nullOffset])
                else:
                    t = t + struct.unpack(fmt, chunk)
            print hotCodeDic[hotCode]['FMT'] % t

    except Exception as e:
        print str(e)
        exit(-1)

    finally:
        srcFile.close()

def printUsage():
    print "USAGE: ./decodeHotLog hotCodeDefFilePath hotLogTopDir pid"
    print "USAGE: ./decodeHotLog hotCodeDefFilePath hotLogTopDir pid tid"
    exit(0)

# (1) parsing argument
try:
    if len(sys.argv) < 4:
        printUsage()
    elif len(sys.argv) == 4:
        defFilePath = sys.argv[1]
        hotLogDir = sys.argv[2]
        pid = int(sys.argv[3])
        tid = -1
    elif len(sys.argv) == 5:
        defFilePath = sys.argv[1]    
        hotLogDir = sys.argv[2]
        pid = int(sys.argv[3])
        tid = int(sys.argv[4])
    else:
        printUsage()
except Exception as e:
    print str(e)
    exit(-1)

# (2) loading hotCodeDef json file into hotCodeDic
try:
    jsonFile = open(defFilePath, 'r')
    hotCodeDic = json.load(jsonFile)

    for hotCode in hotCodeDic:
        checkEventProperty(hotCodeDic, hotCode, "FMT")
        checkEventProperty(hotCodeDic, hotCode, "ARGS")

except Exception as e:
    print str(e)
    exit(-1)

finally:
    jsonFile.close()

# (3) search target file(s)
# XXX: deep in depth
try:
    targetFilePathList = []
    if tid == -1:
        pidFilePrefixName = 'hot.%d.' % pid
        pidFilePrefixNameLen = len(pidFilePrefixName)
        for searchPath, searchDirs, searchFiles in os.walk(hotLogDir):
            if searchFiles:
                for searchFile in searchFiles:
                    if pidFilePrefixName == searchFile[:pidFilePrefixNameLen]:
                        targetFilePathList.append(os.path.join(searchPath, searchFile))
    else:
        pidTidFileName = 'hot.%d.%d' % (pid, tid)
        for searchPath, searchDirs, searchFiles in os.walk(hotLogDir):
            if searchFiles:
                for searchFile in searchFiles:
                    if pidTidFileName == searchFile:
                        targetFilePathList.append(os.path.join(searchPath, searchFile))

except Exception as e:
    print str(e)
    exit(-1)

# (4) decode file(s)
for targetFilePath in targetFilePathList:
    print 'target file : ' + targetFilePath
    decodeFile(targetFilePath, hotCodeDic)
