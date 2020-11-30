#!/usr/bin/env python

"""genNetworkProp.py: """

import json

def checkParamProperty(propDic, prop, propertyName):
    if not propertyName in propDic[prop]:
        print "ERROR: prop %s does not have %s property" % (prop, propertyName)
        exit(-1)

# (1) load enumDef.json
try:
    jsonFile = open('enumDef.json', 'r')
    enumDic = json.load(jsonFile)

except Exception as e:
    print str(e)
    exit(-1)

finally:
    jsonFile.close()

# (3) generate header file
headerTopSentences = [\
"/**",\
" * @file EnumDef.h",\
" * @author moonhoen lee",\
" * @brief enumeration type module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
"#ifndef ENUMDEF_H_",\
"#define ENUMDEF_H_",\
"",\
'#include <string>',\
"",\
]

headerClassDefSentences = [\
"class EnumDef {",\
"public : ",\
"    EnumDef() {}",\
"    virtual ~EnumDef() {}",\
]

headerBottomSentences = [\
"};",\
"",\
"#endif /* ENUMDEF_H_ */",\
]

sourceTopSentences = [\
"/**",\
" * @file EnumDef.cpp",\
" * @author moonhoen lee",\
" * @brief enumeration type module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
'#include "EnumDef.h"',\
'#include "SysLog.h"',\
"",\
"using namespace std;",\
"",\
]

try:
    headerFile = open('EnumDef.h', 'w+')
    sourceFile = open('EnumDef.cpp', 'w+')

    # write top sentences
    for line in headerTopSentences:
        headerFile.write(line + "\n")

    for line in sourceTopSentences:
        sourceFile.write(line + "\n")

    enumMap = enumDic["CONF"]
    for enumData in enumMap:
        name = enumData["NAME"]
        enumList = enumData["ENUM"]
        headerFile.write("typedef enum %s_s {\n" % name)

        isFirst = True
        for enum in enumList:
            if isFirst:
                isFirst = False
            else:
                headerFile.write(',\n')
            headerFile.write("    %s" % enum)
        headerFile.write('\n} %s;\n\n' % name)

    for line in headerClassDefSentences:
        headerFile.write(line + "\n")

    headerFile.write("    static bool isEnumType(std::string type);\n")
    headerFile.write("    static int convertEnumValue(std::string value);\n")

    sourceFile.write("bool EnumDef::isEnumType(string type) {\n")
    isFirst = True
    for enumData in enumMap:
        name = enumData["NAME"]
        if isFirst:
            sourceFile.write('    if (type == "%s") {\n' % name)
            isFirst = False
        else:
            sourceFile.write('    } else if (type == "%s") {\n' % name)
        sourceFile.write('        return true;\n') 
    sourceFile.write('    } else {\n')
    sourceFile.write('        return false;\n')
    sourceFile.write('    }\n}\n\n')

    sourceFile.write("int EnumDef::convertEnumValue(string value) {\n")
    isFirst = True
    for enumData in enumMap:
        name = enumData["NAME"]
        enumList = enumData["ENUM"]
        for enum in enumList:
            if isFirst:
                sourceFile.write('    if (value == "%s") {\n' % enum)
                isFirst = False
            else:
                sourceFile.write('    } else if (value == "%s") {\n' % enum)
            sourceFile.write('        return (int)%s::%s;\n' % (name, enum)) 
    sourceFile.write('    } else {\n')
    sourceFile.write('        SASSERT(false, "invalid enum value. value=%s", value.c_str());\n')
    sourceFile.write('        return -1;\n')
    sourceFile.write('    }\n}\n\n')

    for line in headerBottomSentences:
        headerFile.write(line + "\n")

except Exception as e:
    print str(e)
    exit(-1)

finally:
    headerFile.close()
