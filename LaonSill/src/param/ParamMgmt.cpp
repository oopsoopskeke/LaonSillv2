/**
 * @file ParamMgmt.cpp
 * @date 2016-10-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <assert.h>
#include <string.h>

#include "ParamMgmt.h"
#include "Param.h"
#include "SysLog.h"
#include "MemoryMgmt.h"

using namespace std;

map<string, ParamDef*> ParamMgmt::paramDefMap;

void ParamMgmt::initParamDefMap() {
    Param::fillParamDefMap(paramDefMap);
}

void ParamMgmt::cleanupParamDefMap() {
    map<string, ParamDef*>::iterator iter;
    for (iter = ParamMgmt::paramDefMap.begin();
        iter != ParamMgmt::paramDefMap.end(); iter++) {
       
        ParamDef* paramDef = (ParamDef*)iter->second;
        SDELETE(paramDef);
        iter->second = NULL;
    }
}

bool ParamMgmt::isParamExist(string paramName) {
    if (ParamMgmt::paramDefMap.find(paramName) == ParamMgmt::paramDefMap.end())
        return false;
    return true;
}

void ParamMgmt::getValue(string paramName, void* value) {
    SASSUME(ParamMgmt::isParamExist(paramName), "paramName=%s", paramName.c_str());
    ParamDef* paramDef = ParamMgmt::paramDefMap[paramName];
    memcpy(value, paramDef->valuePt, paramDef->valueLen);
}

void ParamMgmt::setValue(string paramName, void* value) {
    SASSUME(ParamMgmt::isParamExist(paramName), "paramName=%s", paramName.c_str());
    ParamDef* paramDef = ParamMgmt::paramDefMap[paramName];
    memcpy(paramDef->valuePt, value, paramDef->valueLen);
}

char* ParamMgmt::getDesc(string paramName) {
    SASSUME(ParamMgmt::isParamExist(paramName), "paramName=%s", paramName.c_str());
    ParamDef* paramDef = ParamMgmt::paramDefMap[paramName];
    return paramDef->desc;
}

bool ParamMgmt::isMandatory(string paramName) {
    SASSUME(ParamMgmt::isParamExist(paramName), "paramName=%s", paramName.c_str());
    ParamDef* paramDef = ParamMgmt::paramDefMap[paramName];
    return paramDef->isMandatory;
}

bool ParamMgmt::isMutable(string paramName) {
    SASSUME(ParamMgmt::isParamExist(paramName), "paramName=%s", paramName.c_str());
    ParamDef* paramDef = ParamMgmt::paramDefMap[paramName];
    return paramDef->isMutable;
}

bool ParamMgmt::isSessScope(string paramName) {
    SASSUME(ParamMgmt::isParamExist(paramName), "paramName=%s", paramName.c_str());
    ParamDef* paramDef = ParamMgmt::paramDefMap[paramName];
    return paramDef->isSessScope;
}

char* ParamMgmt::getTypeName(string paramName) {
    SASSUME(ParamMgmt::isParamExist(paramName), "paramName=%s", paramName.c_str());
    ParamDef* paramDef = ParamMgmt::paramDefMap[paramName];
    return paramDef->typeName;
}

ParamMgmt::ParamType ParamMgmt::getParamType(string paramName) {
    SASSUME(ParamMgmt::isParamExist(paramName), "paramName=%s", paramName.c_str());
    ParamDef* paramDef = ParamMgmt::paramDefMap[paramName];

    if (strcmp(paramDef->typeName, "UINT8") == 0)
        return ParamMgmt::UINT8;
    else if (strcmp(paramDef->typeName, "INT8") == 0)
        return ParamMgmt::INT8;
    else if (strcmp(paramDef->typeName, "UINT16") == 0)
        return ParamMgmt::UINT16;
    else if (strcmp(paramDef->typeName, "INT16") == 0)
        return ParamMgmt::INT16;
    else if (strcmp(paramDef->typeName, "UINT32") == 0)
        return ParamMgmt::UINT32;
    else if (strcmp(paramDef->typeName, "INT32") == 0)
        return ParamMgmt::INT32;
    else if (strcmp(paramDef->typeName, "UINT64") == 0)
        return ParamMgmt::UINT64;
    else if (strcmp(paramDef->typeName, "INT64") == 0)
        return ParamMgmt::INT64;

    else if (strcmp(paramDef->typeName, "BOOL") == 0)
        return ParamMgmt::INT32;

    else if (strcmp(paramDef->typeName, "FLOAT") == 0)
        return ParamMgmt::FLOAT;
    else if (strcmp(paramDef->typeName, "DOUBLE") == 0)
        return ParamMgmt::DOUBLE;
    else if (strcmp(paramDef->typeName, "LONGDOUBLE") == 0)
        return ParamMgmt::LONGDOUBLE;

    else if (strncmp(paramDef->typeName, "CHAR", 4) == 0)
        return ParamMgmt::STRING;

    else {
        SASSERT(false, "typeName=%s", paramDef->typeName);
        return ParamMgmt::MAX;      // meaningless
    }
}

char* ParamMgmt::getDefaultValue(string paramName) {
    SASSUME(ParamMgmt::isParamExist(paramName), "paramName=%s", paramName.c_str());
    ParamDef* paramDef = ParamMgmt::paramDefMap[paramName];
    return paramDef->defaultValue;
}

class ParamMgmt;
