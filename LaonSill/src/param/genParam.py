#!/usr/bin/env python

"""genParam.py: """

import json;

def checkParamProperty(paramDic, param, propertyName):
    if not propertyName in paramDic[param]:
        print "ERROR: param %s does not have %s property" % (param, propertyName)
        exit(-1)

# deprecated function..
def getRangeValue(paramName, typeStr, value):
    intList = ["uint8_t", "int8_t", "uint16_t", "int16_t",\
              "uint32_t", "int32_t", "uint64_t", "int64_t"]
    realList = ["float", "double", "long double"]
   
    try:
        if typeStr in intList:
            return int(value)
        elif typeStr in realList:
            return real(value)
        else:
            print "ERROR: only integer or real type has range value.\
                param=%s, type=%s, range=%s" % (paramName, typeStr, value)
            exit(-1)

    except Exception as e:
        print "ERROR: range value parsing failed. param=%s, type=%s, range=%s"\
            % (paramName, typeStr, value)
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

# (1) load paramDef.json
try:
    jsonFile = open('paramDef.json', 'r')
    paramDic = json.load(jsonFile)

except Exception as e:
    print str(e)
    exit(-1)

finally:
    jsonFile.close()

# (2) check paramDef syntax
for param in paramDic:
    checkParamProperty(paramDic, param, "DESC")
    checkParamProperty(paramDic, param, "MANDATORY")
    checkParamProperty(paramDic, param, "MUTABLE")
    checkParamProperty(paramDic, param, "SCOPE")
    checkParamProperty(paramDic, param, "TYPE")
    checkParamProperty(paramDic, param, "DEFAULT")

if not "SESS_COUNT" in paramDic:
    print "ERROR: SESS_COUNT parameter does not exist"
    exit(-1)

# (3) generate header file
headerTopSentences = [\
"/**",\
" * @file Param.h",\
" * @author moonhoen lee",\
" * @brief parameter mgmt module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
"#ifndef PARAM_H_",\
"#define PARAM_H_",\
"",\
"#include <stdint.h>",\
"#include <vector>",\
"#include <string>",\
"",\
'#include "common.h"',\
'#include "ParamDef.h"',\
'#include "ParamMgmt.h"',\
"",\
"class ParamMgmt;",\
"",\
"#define SPARAM(n)              Param::_##n",\
"",\
"class Param {",\
"public:",\
"    Param() {}",\
"    virtual ~Param() {}",\
"",\
]

headerBottomSentences = [\
"    static void    fillParamDefMap(std::map<std::string, ParamDef*>& paramDefMap);\n",\
"};",\
"",\
"#endif /* PARAM_H_ */"]

typeDic = {\
    "UINT8" : "uint8_t", "INT8" : "int8_t",\
    "UINT16" : "uint16_t", "INT16" : "int16_t",\
    "UINT32" : "uint32_t", "INT32" : "int32_t",\
    "UINT64" : "uint64_t", "INT64" : "int64_t",\
    "BOOL" : "bool", "FLOAT" : "float",\
    "DOUBLE" : "double", "LONGDOUBLE" : "long double",\
}

try:
    headerFile = open('Param.h', 'w+')

    for line in headerTopSentences:
        headerFile.write(line + "\n")

    for param in paramDic:
        # (1) parse parameter
        desc = paramDic[param]["DESC"]
        mandatory = paramDic[param]["MANDATORY"]
        mutable = paramDic[param]["MUTABLE"]

        sessScope = False
        if paramDic[param]["SCOPE"] == "SESSION":
            sessScope = True

        arrayString = ""
        typeString = ""
        if paramDic[param]["TYPE"] in typeDic:
            typeString = typeDic[paramDic[param]["TYPE"]]
        elif "CHAR" in paramDic[param]["TYPE"]:
            # XXX: needs error-check
            arrayCount = (int)(paramDic[param]["TYPE"].replace(")", "$")\
                                .replace("(", "$").split("$")[1])
            typeString = "char"
            arrayString = "[%d]" % arrayCount
        else:
            print "ERROR: invalid param type(%s) for param(%s)" %\
                (paramDic[param]["TYPE"], param)
            exit(-1)

        defaultValue = paramDic[param]["DEFAULT"]

        # (2) generate parameter comment
        headerFile.write('    // PARAM NAME : %s\n' % param)

        # (3) generate system-scope variable
        if sessScope == False:
            if mutable == True:
                headerFile.write("    static volatile %s _%s%s;\n"\
                    % (typeString, param, arrayString))
            else:
                headerFile.write("    static %s _%s%s;\n"\
                    % (typeString, param, arrayString))
            
        # (4) generate sess-scope variables
        if sessScope == True:
            if mutable == True:
                headerFile.write("    static volatile thread_local %s _%s%s;\n"\
                    % (typeString, param, arrayString))
            else:
                headerFile.write("    static thread_local %s _%s%s;\n"\
                    % (typeString, param, arrayString))

        headerFile.write('\n')

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
" * @file Param.cpp",\
" * @author moonhoen lee",\
" * @brief parameter mgmt module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
'#include "Param.h"',\
'#include "MemoryMgmt.h"',\
"",\
"using namespace std;",\
"",\
""]

paramDefList = []

try:
    sourceFile = open('Param.cpp', 'w+')

    for line in sourceTopSentences:
        sourceFile.write(line + "\n")

    for param in paramDic:
        # (1) parse parameter
        desc = paramDic[param]["DESC"]
        mandatory = paramDic[param]["MANDATORY"]
        mutable = paramDic[param]["MUTABLE"]

        sessScope = False
        if paramDic[param]["SCOPE"] == "SESSION":
            sessScope = True

        typeString = ""
        arrayString = ""
        typeName = paramDic[param]["TYPE"]
        if paramDic[param]["TYPE"] in typeDic:
            typeString = typeDic[paramDic[param]["TYPE"]]
            valueLen = getValueSize(typeString)
        elif "CHAR" in paramDic[param]["TYPE"]:
            # XXX: needs error-check
            arrayCount = (int)(paramDic[param]["TYPE"].replace(")", "$")\
                                .replace("(", "$").split("$")[1])
            typeString = "char"
            arrayString = "[%d]" % arrayCount
            valueLen = arrayCount
        else:
            print "ERROR: invalid param type(%s) for param(%s)" %\
                (paramDic[param]["TYPE"], param)
            exit(-1)

        defaultValue = paramDic[param]["DEFAULT"]

        # (2) generate parameter comment
        sourceFile.write('// PARAM NAME : %s\n' % param)

        if "char" in typeString:
            defaultValueStr = '{"%s"}' % str(defaultValue)
        elif 'bool' in typeString:
            defaultValueStr = '%s' % str(defaultValue).lower()
        else:
            defaultValueStr = '%s' % str(defaultValue)

        # (3) generate system-scope variable
        if sessScope == False:
            if mutable == True:
                sourceFile.write("volatile %s Param::_%s%s = %s;\n\n"\
                    % (typeString, param, arrayString, defaultValueStr))
            else:
                sourceFile.write("%s Param::_%s%s = %s;\n\n"\
                    % (typeString, param, arrayString, defaultValueStr))

        # (4) generate sess-scope variables
        if sessScope == True:
            if mutable == True:
                sourceFile.write("volatile thread_local %s Param::_%s%s = %s;\n\n"\
                    % (typeString, param, arrayString, defaultValueStr))
            else:
                sourceFile.write("thread_local %s Param::_%s%s = %s;\n\n"\
                    % (typeString, param, arrayString, defaultValueStr))

        paramDefList.append((param, desc, str(defaultValue), typeName, mandatory,\
            mutable, sessScope, valueLen))


    # (12) prepare fillParamDefMap func() 
    sourceFile.write("void Param::fillParamDefMap(map<string, ParamDef*>& paramDefMap) {\n")
    for paramDef in paramDefList:
        if "CHAR" in typeString:
            sourceFile.write(\
                '    SNEW_ONCE(paramDefMap["%s"], ParamDef, "%s", \
                \n        "%s", "%s", %s, %s, %s, (void*)Param::_%s, %s);\n'\
                % (str(paramDef[0]), str(paramDef[1]), str(paramDef[2]), str(paramDef[3]),\
                str(paramDef[4]).lower(), str(paramDef[5]).lower(), str(paramDef[6]).lower(),\
                str(paramDef[0]), str(paramDef[7])))
        else:
            sourceFile.write(\
                '    SNEW_ONCE(paramDefMap["%s"], ParamDef, "%s", \
                \n        "%s", "%s", %s, %s, %s, (void*)&Param::_%s, %s);\n'\
                % (str(paramDef[0]), str(paramDef[1]), str(paramDef[2]), str(paramDef[3]),\
                str(paramDef[4]).lower(), str(paramDef[5]).lower(), str(paramDef[6]).lower(),\
                str(paramDef[0]), str(paramDef[7])))
            
    sourceFile.write("}\n\n")
    
except Exception as e:
    print str(e)
    exit(-1)
finally:
    sourceFile.close()
