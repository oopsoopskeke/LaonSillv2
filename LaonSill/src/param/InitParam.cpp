/**
 * @file InitParam.cpp
 * @date 2016-10-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

#include <string>
#include <vector>

#include "InitParam.h"
#include "Param.h"
#include "ParamMgmt.h"
#include "SysLog.h"

using namespace std;

void InitParam::init() {
    // XXX: mandatory 체크 필요
    // (1) paramDefMap을 초기화 한다.
    ParamMgmt::initParamDefMap();

    // (2) 초기화 파라미터를 초기화 파일(laonsill.conf)로 부터 읽어 들여서 초기화 한다.
    InitParam::load();
}

void InitParam::destroy() {
    ParamMgmt::cleanupParamDefMap();
}

bool InitParam::isWhiteSpace(char c) {
    if ((c == ' ') || (c == '\t') || (c == '\r') || (c == '\n'))
        return true;
    return false;
}

/**
 * XXX: 리눅스 64bit 환경에 대해서만 고려하였다.
 */
const int DECIMAL_BASE = 10;
void InitParam::setParam(const char* paramName, const char* paramValue) {
    string strParamName = string(paramName);
    SASSERT(ParamMgmt::isParamExist(strParamName), "");

    ParamMgmt::ParamType paramType = ParamMgmt::getParamType(strParamName);

    switch (paramType) {
    // XXX: endptr를 고려해서 더 세련된 에러 처리를 하자
    case ParamMgmt::UINT8: {
        uint8_t value = (uint8_t)strtol(paramValue, NULL, DECIMAL_BASE);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }
    case ParamMgmt::INT8: {
        int8_t value = (int8_t)strtol(paramValue, NULL, DECIMAL_BASE);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }
    case ParamMgmt::UINT16: {
        uint16_t value = (uint16_t)strtol(paramValue, NULL, DECIMAL_BASE);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }
    case ParamMgmt::INT16: {
        int16_t value = (int16_t)strtol(paramValue, NULL, DECIMAL_BASE);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }
    case ParamMgmt::UINT32: {
        uint32_t value = (uint32_t)strtol(paramValue, NULL, DECIMAL_BASE);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }
    case ParamMgmt::INT32: {
        int32_t value = (int32_t)strtol(paramValue, NULL, DECIMAL_BASE);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }
    case ParamMgmt::UINT64: {
        uint64_t value = (uint64_t)strtoll(paramValue, NULL, DECIMAL_BASE);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }
    case ParamMgmt::INT64: {
        int64_t value = (int64_t)strtoll(paramValue, NULL, DECIMAL_BASE);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }

    case ParamMgmt::FLOAT: {
        float value = strtof(paramValue, NULL);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }
    case ParamMgmt::DOUBLE: {
        double value = strtod(paramValue, NULL);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }
    case ParamMgmt::LONGDOUBLE: {
        long double value = strtold(paramValue, NULL);
        ParamMgmt::setValue(strParamName, &value);
        break;
    }

    case ParamMgmt::BOOL: {
        bool value;
        if (strcmp(paramValue, "TRUE") == 0) {
            value = true;
        } else if (strcmp(paramValue, "FALSE") == 0) {
            value = false;
        } else {
            SASSERT(false, "invalid value");
        }
        ParamMgmt::setValue(strParamName, &value);
        break;
    }

    case ParamMgmt::STRING:
        // XXX: needs string length check
        ParamMgmt::setValue(strParamName, (void*)paramValue);
        break;
    

    default:
        SASSERT(false, "invalid type!!!");
    }
}

void InitParam::loadParam(const char* line, int len) {
    bool    foundParamName = false;
    bool    foundEqual = false;

    char    paramName[128];
    int     paramNameOffset = 0;

    int     i;

    // (1) get paramName
    for (i = 0; i < len; i++) {

        // found comment
        if (line[i] == '#') {
            SASSERT(!foundParamName, "");
            break;
        }

        if (!foundParamName && !isWhiteSpace(line[i])) {
            foundParamName = true;
            paramName[paramNameOffset] = line[i];
            paramNameOffset++;
        } else if (foundParamName) {
            if (line[i] == '=') {
                paramName[paramNameOffset] = '\0';
                foundEqual = true;
                i++;
                break;
            } else if (isWhiteSpace(line[i])) {
                paramName[paramNameOffset] = '\0';
                i++;
                break;
            } else {
                paramName[paramNameOffset] = line[i];
                paramNameOffset++;
            }
        }
    }

    // (2) expect equal
    if (!foundEqual) {
        for (; i < len; i++) {
            if (line[i] == '=') {
                foundEqual = true;
                i++;
                break;
            } else {
                SASSERT(!isWhiteSpace(line[i]), "");
            }
        }
    }

    // (3) get param value
    bool    doubleQuotesFound = false;
    char    paramValue[128];
    int     paramValueOffset = 0;
    for (; i < len; i++) {
        if (isWhiteSpace(line[i])) {
            if (doubleQuotesFound) {
                paramValue[paramValueOffset] = line[i];
                paramValueOffset++;
            } else if (paramValueOffset > 0) {
                paramValue[paramValueOffset ] ='\0';
                break;
            }
        } else if (line[i] == '"') {
            if (doubleQuotesFound) {
                paramValue[paramValueOffset] = line[i];
                paramValueOffset++;
                paramValue[paramValueOffset] = '\0';
                break;
            } else {
                paramValue[paramValueOffset] = line[i];
                paramValueOffset++;
                doubleQuotesFound = true;
            }
        } else if (line[i] == '#') {
            if (!doubleQuotesFound) {
                paramValue[paramValueOffset] = '\0';
                break;
            }
        } else {
            paramValue[paramValueOffset] = line[i];
            paramValueOffset++;
        }
    }

    SASSERT(paramValueOffset > 0, "");

    // (4) set param value
    InitParam::setParam(paramName, paramValue);
}

const char* LAONSILL_HOME_ENVNAME = "LAONSILL_HOME";
const char* IPARAM_FILENAME  = {"laonsill.conf"};
void InitParam::load() {
    char* initParamPrefixPath = getenv(LAONSILL_HOME_ENVNAME);
    if (initParamPrefixPath == NULL) {
        cout << "ERROR: You must specify $LAONSILL_HOME" << endl;
        exit(-1);
    }

    char initParamPath[PATH_MAX];
    sprintf(initParamPath, "%s/%s", initParamPrefixPath, IPARAM_FILENAME);

    FILE* fp = fopen(initParamPath, "r");
    SASSERT(fp, "");

    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    while ((read = getline(&line, &len, fp)) != -1) {
        InitParam::loadParam(line, (int)len);
    }

    if (line)
        free(line);     // do not need to use SFREE

    fclose(fp);
}
