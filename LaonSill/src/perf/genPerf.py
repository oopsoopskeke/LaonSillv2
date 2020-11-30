#!/usr/bin/env python

"""genPerf.py: """

import json;

def checkParamProperty(perfDic, perf, propertyName):
    if not propertyName in perfDic[perf]:
        print "ERROR: perf %s does not have %s property" % (perf, propertyName)
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

# (1) load perfDef.json
try:
    jsonFile = open('perfDef.json', 'r')
    perfDic = json.load(jsonFile)

except Exception as e:
    print str(e)
    exit(-1)

finally:
    jsonFile.close()

# (2) check perfDef syntax
for perf in perfDic:
    checkParamProperty(perfDic, perf, "DESC")
    checkParamProperty(perfDic, perf, "SCOPE")
    checkParamProperty(perfDic, perf, "USETIME")
    checkParamProperty(perfDic, perf, "USEAVGTIME")
    checkParamProperty(perfDic, perf, "USEMAXTIME")
    checkParamProperty(perfDic, perf, "ARGS")

# (3) generate header file
headerTopSentences = [\
"/**",\
" * @file PerfList.h",\
" * @author moonhoen lee",\
" * @brief performance list module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
"#ifndef PERFLIST_H_",\
"#define PERFLIST_H_",\
"",\
"#include <stdint.h>",\
"#include <time.h>",\
"",\
"#include <vector>",\
"#include <map>",\
"#include <string>",\
"",\
'#include "common.h"',\
'#include "PerfArgDef.h"',\
'#include "PerfDef.h"',\
'#include "Param.h"',\
'#include "SysLog.h"',\
"",\
"class PerfList {",\
"public:",\
"    PerfList() {}",\
"    virtual ~PerfList() {}",\
"",\
]

headerBottomSentences = [\
"    static void    fillPerfDefMap(std::map<std::string, PerfDef*>& perfDefMap);\n",\
"};",\
"",\
"#endif /* PERFLIST_H_ */"]

typeDic = {\
    "UINT8" : "uint8_t", "INT8" : "int8_t","UINT16" : "uint16_t", "INT16" : "int16_t",\
    "UINT32" : "uint32_t", "INT32" : "int32_t","UINT64" : "uint64_t", "INT64" : "int64_t",\
    "FLOAT" : "float", "DOUBLE" : "double", "LONGDOUBLE" : "long double"\
}

try:
    headerFile = open('PerfList.h', 'w+')

    for line in headerTopSentences:
        headerFile.write(line + "\n")

    for perf in perfDic:
        # (1) parse performance def
        desc = perfDic[perf]["DESC"]

        jobScope = False
        if perfDic[perf]["SCOPE"] == "JOB":
            jobScope = True

        useTime = perfDic[perf]["USETIME"]

        if useTime == False:
            useAvgTime = False
            useMaxTime = False
        else:
            useAvgTime = perfDic[perf]["USEAVGTIME"]
            useMaxTime = perfDic[perf]["USEMAXTIME"]

        perfArgs = perfDic[perf]["ARGS"]
        newArgList = []
        for perfArg in perfArgs:
            if perfArg[1] in typeDic:
                typeString = typeDic[perfArg[1]]
            else:
                print "ERROR: invalid args type(%s) for perf(%s)" %\
                    (perfDic[perf]["TYPE"], perf)
                exit(-1)
            newArgList.append((perfArg[0], typeString, perfArg[2])) 

        # (2) generate performance comment
        headerFile.write('    // PERF NAME : %s\n' % perf)

        # (3) generate variables
        if jobScope == True:
            volStr = "thread_local"
        else:
            volStr = "volatile"

        headerFile.write("    static %s long _%sCount;\n" % (volStr, perf))
        if useTime == True:
            headerFile.write("    static %s double _%sTime;\n" % (volStr, perf))
        if useAvgTime == True:
            headerFile.write("    static %s double _%sAvgTime;\n" % (volStr, perf))
        if useMaxTime == True:
            headerFile.write("    static %s double _%sMaxTime;\n" % (volStr, perf))

        for newArg in newArgList:
            headerFile.write("    static %s %s _%s_%s;\n" %\
                (volStr, newArg[1], perf, newArg[0]))

        # (4) generate functions
        headerFile.write("    static void mark%s(" % perf)
        isFirst = True
        for newArg in newArgList:
            if isFirst == True:
                isFirst = False 
            else:
                headerFile.write(", ") 
            headerFile.write("%s %s" % (newArg[1], newArg[0]))
        headerFile.write(");\n")

        headerFile.write("    static void start%s(struct timespec* startTime);\n" % perf)

        headerFile.write("    static void end%s(struct timespec startTime" % perf)
        for newArg in newArgList:
            headerFile.write(", %s %s" % (newArg[1], newArg[0]))
        headerFile.write(");\n")

        headerFile.write("    static void clear%s();\n" % perf)

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
" * @file PerfList.cpp",\
" * @author moonhoen lee",\
" * @brief performance list module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
'#include "PerfList.h"',\
'#include "MemoryMgmt.h"',\
"",\
"using namespace std;",\
"",\
""]

perfDefList = []

try:
    sourceFile = open('PerfList.cpp', 'w+')

    for line in sourceTopSentences:
        sourceFile.write(line + "\n")

    for perf in perfDic:
        # (1) parse performance def
        desc = perfDic[perf]["DESC"]

        jobScope = False
        if perfDic[perf]["SCOPE"] == "JOB":
            jobScope = True

        useTime = perfDic[perf]["USETIME"]

        if useTime == False:
            useAvgTime = False
            useMaxTime = False
        else:
            useAvgTime = perfDic[perf]["USEAVGTIME"]
            useMaxTime = perfDic[perf]["USEMAXTIME"]

        perfArgs = perfDic[perf]["ARGS"]
        newArgList = []
        for perfArg in perfArgs:
            if perfArg[1] in typeDic:
                typeString = typeDic[perfArg[1]]
            else:
                print "ERROR: invalid args type(%s) for perf(%s)" %\
                    (perfDic[perf]["TYPE"], perf)
                exit(-1)
            newArgList.append((perfArg[0], typeString, perfArg[2])) 

        # (2) generate performance comment
        sourceFile.write('// PERF NAME : %s\n' % perf)

        # (3) generate variables
        if jobScope == True:
            volStr = "thread_local"
        else:
            volStr = "volatile"

        sourceFile.write("%s long PerfList::_%sCount = 0L;\n" % (volStr, perf))
        if useTime == True:
            sourceFile.write("%s double PerfList::_%sTime = 0;\n" % (volStr, perf))
        if useAvgTime == True:
            sourceFile.write("%s double PerfList::_%sAvgTime = 0;\n" % (volStr, perf))
        if useMaxTime == True:
            sourceFile.write("%s double PerfList::_%sMaxTime = 0;\n" % (volStr, perf))

        for newArg in newArgList:
            sourceFile.write("%s %s PerfList::_%s_%s = 0;\n" %\
                (volStr, newArg[1], perf, newArg[0]))

        # (4) generate functions
        # (4-1) mark function
        sourceFile.write("void PerfList::mark%s(" % perf)
        isFirst = True
        for newArg in newArgList:
            if isFirst == True:
                isFirst = False 
            else:
                sourceFile.write(", ") 
            sourceFile.write("%s %s" % (newArg[1], newArg[0]))
        sourceFile.write(") {\n")
        if useTime == True:
            sourceFile.write('    SASSERT(false, "you should use SPERF_START() or SPERF_END(). perf name=%s",')
            sourceFile.write('"%s");\n' % perf) 
        else:
            sourceFile.write('    PerfList::_%sCount += 1L;\n' % perf)
            for newArg in newArgList:
                sourceFile.write("    PerfList::_%s_%s = %s;\n" % (perf, newArg[0], newArg[0]));
        sourceFile.write('}\n\n')

        # (4-2) start function
        sourceFile.write("void PerfList::start%s(struct timespec* startTime) {\n" % perf)

        if useTime == False:
            sourceFile.write('    SASSERT(false, "you should use SPERF_MARK(). perf name=%s",')
            sourceFile.write('"%s");\n' % perf)
        else:
            if jobScope == True:
                sourceFile.write('    if (SPARAM(JOBSCOPE_CLOCKTYPE) == 0)\n')
                sourceFile.write('        clock_gettime(CLOCK_THREAD_CPUTIME_ID, startTime);\n')
                sourceFile.write('    else if (SPARAM(JOBSCOPE_CLOCKTYPE) == 1)\n')
                sourceFile.write('        clock_gettime(CLOCK_MONOTONIC, startTime);\n')
                sourceFile.write('    else if (SPARAM(JOBSCOPE_CLOCKTYPE) == 2)\n')
                sourceFile.write('        clock_gettime(CLOCK_MONOTONIC_COARSE, startTime);\n')
                sourceFile.write('    else\n')
                sourceFile.write('        SASSERT(false, "invalid clock type. clock type=%d"')
                sourceFile.write(', (int)SPARAM(JOBSCOPE_CLOCKTYPE));\n')
            else:
                sourceFile.write('    clock_gettime(CLOCK_REALTIME, startTime);\n')
        sourceFile.write('}\n\n')

        # (4-3) end function
        sourceFile.write("void PerfList::end%s(struct timespec startTime" % perf)
        for newArg in newArgList:
            sourceFile.write(", %s %s" % (newArg[1], newArg[0]))
        sourceFile.write(") {\n")

        if useTime == False:
            sourceFile.write('    SASSERT(false, "you should use SPERF_MARK(). perf name=%s",')
            sourceFile.write('"%s");\n' % perf)
        else:
            sourceFile.write('    struct timespec endTime;\n')
            if jobScope == True:
                sourceFile.write('    if (SPARAM(JOBSCOPE_CLOCKTYPE) == 0)\n')
                sourceFile.write('        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &endTime);\n')
                sourceFile.write('    else if (SPARAM(JOBSCOPE_CLOCKTYPE) == 1)\n')
                sourceFile.write('        clock_gettime(CLOCK_MONOTONIC, &endTime);\n')
                sourceFile.write('    else if (SPARAM(JOBSCOPE_CLOCKTYPE) == 2)\n')
                sourceFile.write('        clock_gettime(CLOCK_MONOTONIC_COARSE, &endTime);\n')
                sourceFile.write('    else\n')
                sourceFile.write('        SASSERT(false, "invalid clock type. clock type=%d"')
                sourceFile.write(', (int)SPARAM(JOBSCOPE_CLOCKTYPE));\n')
            else:
                sourceFile.write('    clock_gettime(CLOCK_REALTIME, &endTime);\n')

            sourceFile.write('    PerfList::_%sCount += 1L;\n' % perf)
            sourceFile.write('    double elapsed = (endTime.tv_sec - startTime.tv_sec)') 
            sourceFile.write('\n        + (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;\n')
            sourceFile.write('    PerfList::_%sTime += elapsed;\n' % perf)

            if useAvgTime == True:
                sourceFile.write('    PerfList::_%sAvgTime = PerfList::_%sTime /\
(double)PerfList::_%sCount;\n' % (perf, perf, perf))

            if useMaxTime == True:
                sourceFile.write('    if (elapsed > PerfList::_%sMaxTime)\n' % perf)
                sourceFile.write('        PerfList::_%sMaxTime = elapsed;\n' % perf)

            for newArg in newArgList:
                sourceFile.write("    PerfList::_%s_%s += %s;\n" % (perf, newArg[0], newArg[0]));

        sourceFile.write('}\n\n')

        # (4-4) clear function
        sourceFile.write("void PerfList::clear%s() {\n" % perf)
        sourceFile.write('    PerfList::_%sCount = 0L;\n' % perf)
        if useTime == True:
            sourceFile.write('    PerfList::_%sTime = 0.0;\n' % perf)
            if useAvgTime == True:
                sourceFile.write('    PerfList::_%sAvgTime = 0.0;\n' % perf)
            if useMaxTime == True:
                sourceFile.write('    PerfList::_%sMaxTime = 0.0;\n' % perf)
        for newArg in newArgList:
            sourceFile.write("    PerfList::_%s_%s = 0;\n" % (perf, newArg[0]))
        sourceFile.write('}\n\n')

        sourceFile.write('\n')
        perfDefList.append((perf, desc, jobScope, useTime, useAvgTime, useMaxTime, newArgList))

    # (12) prepare fillPerfDefMap func() 
    sourceFile.write("void PerfList::fillPerfDefMap(map<string, PerfDef*>& perfDefMap) {\n")

    isFirst = True
    for perfDef in perfDefList:
        if isFirst == True:
            isFirst = False
        else:
            sourceFile.write('\n')
        sourceFile.write('    PerfDef* perfDef%s = NULL;\n' % str(perfDef[0]))
        sourceFile.write('    SNEW_ONCE(perfDef%s, PerfDef, "%s", %s, %s, %s, %s,\
\n        (void*)&PerfList::_%sCount'\
            % (str(perfDef[0]), str(perfDef[1]), str(perfDef[2]).lower(),\
                str(perfDef[3]).lower(), str(perfDef[4]).lower(),\
                str(perfDef[5]).lower(), str(perfDef[0])))
        if perfDef[3] == True:  # useTime
            sourceFile.write(', (void*)&PerfList::_%sTime,' % str(perfDef[0]))
        else:
            sourceFile.write(', NULL,')

        if perfDef[4] == True:  # useAvgTime
            sourceFile.write('\n        (void*)&PerfList::_%sAvgTime,' % str(perfDef[0]))
        else:
            sourceFile.write('\n        NULL,')
       
        if perfDef[5] == True:  # useMaxTime
            sourceFile.write('\n        (void*)&PerfList::_%sMaxTime,' % str(perfDef[0]))
        else:
            sourceFile.write('\n        NULL,')
        sourceFile.write('%d);\n' % len(perfDef[6]))
        
        for newArg in perfDef[6]:
            sourceFile.write('    { PerfArgDef* newPerfArgDef;\n')
            sourceFile.write('    SNEW_ONCE(newPerfArgDef, PerfArgDef, "%s", "%s", "%s",\
\n        (void*)&PerfList::_%s_%s, %d); \n'\
                % (str(newArg[0]), str(newArg[1]), str(newArg[2]),\
str(perfDef[0]), str(newArg[0]), getValueSize(str(newArg[1]))))
            sourceFile.write('    perfDef%s->addArgs(newPerfArgDef); }\n' % str(perfDef[0]))
        sourceFile.write('    perfDefMap["%s"] = perfDef%s;\n'\
            % (str(perfDef[0]), str(perfDef[0])))
            
    sourceFile.write("}\n\n")
    
except Exception as e:
    print str(e)
    exit(-1)
finally:
    sourceFile.close()
