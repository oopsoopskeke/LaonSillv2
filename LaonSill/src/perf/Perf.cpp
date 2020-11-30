/**
 * @file Perf.cpp
 * @date 2016-11-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Perf.h"
#include "SysLog.h"

using namespace std;

map<string, PerfDef*> Perf::perfDefMap;

void Perf::init() {
    PerfList::fillPerfDefMap(Perf::perfDefMap);
}

bool Perf::isPerfExist(string perfName) {
    if (Perf::perfDefMap.find(perfName) == Perf::perfDefMap.end())
        return false;
    return true;
}

char* Perf::getDesc(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];
    return perfDef->desc;
}

bool Perf::isJobScope(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];
    return perfDef->jobScope;
}

bool Perf::isUseTime(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];
    return perfDef->useTime;
}

bool Perf::isUseAvgTime(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];
    return perfDef->useAvgTime;
}

bool Perf::isUseMaxTime(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];
    return perfDef->useMaxTime;
}

long Perf::getCount(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];

    long perfCount;
    memcpy((void*)&perfCount, perfDef->countPtr, sizeof(long));

    return perfCount;
}

double Perf::getTime(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];

    double perfTime; 
    memcpy((void*)&perfTime, perfDef->timePtr, sizeof(double));

    return perfTime;
}

double Perf::getAvgTime(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];

    double perfAvgTime; 
    memcpy((void*)&perfAvgTime, perfDef->avgTimePtr, sizeof(double));

    return perfAvgTime;
}

double Perf::getMaxTime(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];

    double perfMaxTime; 
    memcpy((void*)&perfMaxTime, perfDef->maxTimePtr, sizeof(double));

    return perfMaxTime;
}

int Perf::getArgCount(string perfName) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    PerfDef* perfDef = Perf::perfDefMap[perfName];

    return perfDef->argCount;
}

void Perf::getArgValue(string perfName, int argIndex, void* value) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    SASSUME(argIndex < Perf::getArgCount(perfName),
        "perfName=%s, argIdx=%d", perfName.c_str(), argIndex);

    PerfDef* perfDef = Perf::perfDefMap[perfName];
    PerfArgDef* perfArgDef = perfDef->argArray[argIndex];

    memcpy(value, perfArgDef->valuePtr, perfArgDef->valueLen);
}

char* Perf::getArgDesc(std::string perfName, int argIndex) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    SASSUME(argIndex < Perf::getArgCount(perfName),
        "perfName=%s, argIdx=%d", perfName.c_str(), argIndex);

    PerfDef* perfDef = Perf::perfDefMap[perfName];
    PerfArgDef* perfArgDef = perfDef->argArray[argIndex];

    return perfArgDef->desc;
}

char* Perf::getArgName(std::string perfName, int argIndex) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    SASSUME(argIndex < Perf::getArgCount(perfName),
        "perfName=%s, argIdx=%d", perfName.c_str(), argIndex);

    PerfDef* perfDef = Perf::perfDefMap[perfName];
    PerfArgDef* perfArgDef = perfDef->argArray[argIndex];

    return perfArgDef->argName;
}

char* Perf::getArgTypeName(string perfName, int argIndex) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    SASSUME(argIndex < Perf::getArgCount(perfName),
        "perfName=%s, argIdx=%d", perfName.c_str(), argIndex);

    PerfDef* perfDef = Perf::perfDefMap[perfName];
    PerfArgDef* perfArgDef = perfDef->argArray[argIndex];

    return perfArgDef->typeName;
}

PerfArgDef::PerfArgType Perf::getArgType(string perfName, int argIndex) {
    SASSUME(Perf::isPerfExist(perfName), "perfName=%s", perfName.c_str());
    SASSUME(argIndex < Perf::getArgCount(perfName),
        "perfName=%s, argIdx=%d", perfName.c_str(), argIndex);

    PerfDef* perfDef = Perf::perfDefMap[perfName];
    PerfArgDef* perfArgDef = perfDef->argArray[argIndex];

    if (strcmp(perfArgDef->typeName, "UINT8") == 0)
        return PerfArgDef::UINT8;
    else if (strcmp(perfArgDef->typeName, "INT8") == 0)
        return PerfArgDef::INT8;
    else if (strcmp(perfArgDef->typeName, "UINT16") == 0)
        return PerfArgDef::UINT16;
    else if (strcmp(perfArgDef->typeName, "INT16") == 0)
        return PerfArgDef::INT16;
    else if (strcmp(perfArgDef->typeName, "UINT32") == 0)
        return PerfArgDef::UINT32;
    else if (strcmp(perfArgDef->typeName, "INT32") == 0)
        return PerfArgDef::INT32;
    else if (strcmp(perfArgDef->typeName, "UINT64") == 0)
        return PerfArgDef::UINT64;
    else if (strcmp(perfArgDef->typeName, "INT64") == 0)
        return PerfArgDef::INT64;

    else if (strcmp(perfArgDef->typeName, "FLOAT") == 0)
        return PerfArgDef::FLOAT;
    else if (strcmp(perfArgDef->typeName, "DOUBLE") == 0)
        return PerfArgDef::DOUBLE;
    else if (strcmp(perfArgDef->typeName, "LONGDOUBLE") == 0)
        return PerfArgDef::LONGDOUBLE;

    else {
        SASSERT(false, "typeName=%s", perfArgDef->typeName);
        return PerfArgDef::MAX;      // meaningless
    }
}
