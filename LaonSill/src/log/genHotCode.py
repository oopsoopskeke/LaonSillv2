#!/usr/bin/env python

"""genHotCode.py: """

import json;

def checkEventProperty(hotCodeDic, hotCode, propertyName):
    if not propertyName in hotCodeDic[hotCode]:
        print "ERROR: hotCode %s does not have %s property" % (hotCode, propertyName)
        exit(-1)

# XXX:  we only considers Linux 64bit platform.
def getValueSize(typeStr):
    if typeStr in ["uint8_t", "int8_t"]:
        return 1
    elif typeStr in ["uint16_t", "int16_t"]:
        return 2
    elif typeStr in ["uint32_t", "int32_t"]:
        return 4
    elif typeStr in ["uint64_t", "int64_t"]:
        return 8
    elif typeStr in ["float"]:
        return 4
    elif typeStr in ["double"]:
        return 8
    elif typeStr in ["long double"]:
        return 16
    elif typeStr in ["bool"]:
        return 1
    return 0

# (1) load hotCodeDef.json
try:
    jsonFile = open('hotCodeDef.json', 'r')
    hotCodeDic = json.load(jsonFile)

except Exception as e:
    print str(e)
    exit(-1)

finally:
    jsonFile.close()

# (2) check hotCodeDef syntax
for hotCode in hotCodeDic:
    checkEventProperty(hotCodeDic, hotCode, "FMT")
    checkEventProperty(hotCodeDic, hotCode, "ARGS")

# (3) generate header file
headerTopSentences = [\
"/**",\
" * @file HotCode.h",\
" * @author moonhoen lee",\
" * @brief hot code module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
"#ifndef HOTCODE_H_",\
"#define HOTCODE_H_",\
"",\
"#include <stdint.h>",\
"#include <vector>",\
"#include <string>",\
"",\
'#include "common.h"',\
'#include "HotLog.h"',\
"",\
"class HotLog;",\
"",\
"class HotCode {",\
"public:",\
"    HotCode() {}",\
"    virtual ~HotCode() {}",\
"",\
]

headerBottomSentences = [\
"};",\
"",\
"#endif /* HOTCODE_H_ */"]

typeDic = {\
    "UINT8" : "uint8_t", "INT8" : "int8_t",\
    "UINT16" : "uint16_t", "INT16" : "int16_t",\
    "UINT32" : "uint32_t", "INT32" : "int32_t",\
    "UINT64" : "uint64_t", "INT64" : "int64_t",\
    "BOOL" : "bool", "FLOAT" : "float",\
    "DOUBLE" : "double", "LONGDOUBLE" : "long double",\
}

try:
    headerFile = open('HotCode.h', 'w+')

    # (1) write head of header
    for line in headerTopSentences:
        headerFile.write(line + "\n")
    
    # (2) write hot code function
    for hotCode in hotCodeDic:
        # (2-1) parse parameter
        argArray = hotCodeDic[hotCode]["ARGS"]

        # (2-2) fill eventDic
        paramList = []
        paramIdx = 0
        paramSize = 0
        for param in argArray:
            arrayString = ""
            typeString = ""

            if param in typeDic:
                typeString = typeDic[param]
                paramSize = int(getValueSize(typeString))
            elif "CHAR" in param:
                # XXX: needs error-check
                arrayCount = int(param.replace(")", "$").replace("(", "$").split("$")[1])
                typeString = "char"
                paramSize = arrayCount
            else:
                print "ERROR: invalid hotCode type(%s) for hotCode(%s)" % (param, hotCode)
                exit(-1)

            paramList.append((paramIdx, typeString, paramSize))
            paramIdx = paramIdx + 1
   
        # (2-3) write function
        headerFile.write('    static void HOT_LOG%s(' % hotCode)
        isFirst = True
        totalLen = 4        # size of hotcode(integer type)
        for paramTuple in paramList:
            if isFirst == True:
                isFirst = False
            else:
                headerFile.write(', ')

            if paramTuple[1] == 'char':
                headerFile.write('const char v%d[%d]' % (paramTuple[0], paramTuple[2]))
            else:
                headerFile.write('%s v%d' % (paramTuple[1], paramTuple[0]))

            totalLen = totalLen + int(paramTuple[2])

        headerFile.write(');\n')
        
    # (3) write bottom of header
    for line in headerBottomSentences:
        headerFile.write(line + "\n")

except Exception as e:
    print str(e)
    exit(-1)

finally:
    headerFile.close()

# (4) generate source file
sourceTopSentences = [\
"/**",\
" * @file HotCode.cpp",\
" * @author moonhoen lee",\
" * @brief hot code module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
'#include "HotCode.h"',\
"",\
"using namespace std;",\
"",\
""]

try:
    sourceFile = open('HotCode.cpp', 'w+')

    for line in sourceTopSentences:
        sourceFile.write(line + "\n")

    # (2) write hot code function
    for hotCode in hotCodeDic:
        # (2-1) parse parameter
        argArray = hotCodeDic[hotCode]["ARGS"]

        # (2-2) fill eventDic
        paramList = []
        paramIdx = 0
        paramSize = 0
        for param in argArray:
            arrayString = ""
            typeString = ""

            if param in typeDic:
                typeString = typeDic[param]
                paramSize = int(getValueSize(typeString))
            elif "CHAR" in param:
                # XXX: needs error-check
                arrayCount = int(param.replace(")", "$").replace("(", "$").split("$")[1])
                typeString = "char"
                paramSize = arrayCount
            else:
                print "ERROR: invalid hotCode type(%s) for hotCode(%s)" % (param, hotCode)
                exit(-1)

            paramList.append((paramIdx, typeString, paramSize))
            paramIdx = paramIdx + 1
   
        # (2-3) write function
        sourceFile.write('void HotCode::HOT_LOG%s(' % hotCode)
        isFirst = True
        totalLen = 4        # size of hotcode(integer type)
        for paramTuple in paramList:
            if isFirst == True:
                isFirst = False
            else:
                sourceFile.write(', ')

            if paramTuple[1] == 'char':
                sourceFile.write('const char v%d[%d]' % (paramTuple[0], paramTuple[2]))
            else:
                sourceFile.write('%s v%d' % (paramTuple[1], paramTuple[0]))

            totalLen = totalLen + int(paramTuple[2])

        sourceFile.write(') {\n')
        sourceFile.write('    HotLogContext* context = HotLog::getHotCodeContext();\n\n')
        sourceFile.write('    if (!context->checkMem(%dUL)) { return; }\n\n' % totalLen)

        sourceFile.write('    int eventCode = %s;\n' % hotCode)
        sourceFile.write('    context->writeMem((char*)&eventCode, 0UL, 4UL);\n') 
        paramOffset = 4
        for paramTuple in paramList:
            if paramTuple[1] == 'char':
                sourceFile.write('    context->writeMem((char*)v%d, %dUL, %dUL);\n'\
                    % (paramTuple[0], paramOffset, paramTuple[2]))
            else:
                sourceFile.write('    context->writeMem((char*)&v%d, %dUL, %dUL);\n'\
                    % (paramTuple[0], paramOffset, paramTuple[2]))
            paramOffset = paramOffset + int(paramTuple[2])

        sourceFile.write('\n    context->updateMem(%dUL);\n}\n\n' % totalLen)

except Exception as e:
    print str(e)
    exit(-1)

finally:
    sourceFile.close()
