/**
 * @file PerfArgDef.h
 * @date 2016-11-07
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PERFARGDEF_H
#define PERFARGDEF_H 

#include <string.h>

#include "common.h"

#define PERFARGDEF_ARGNAME_MAXSIZE          (64)
#define PERFARGDEF_DESC_MAXSIZE             (256)
#define PERFARGDEF_TYPENAME_MAXSIZE         (32)

class PerfArgDef {
public: 
    enum PerfArgType : int {
        UINT8 = 0,
        INT8,
        UINT16,
        INT16,
        UINT32,
        INT32,
        UINT64,
        INT64,
        FLOAT,
        DOUBLE,
        LONGDOUBLE,
        MAX
    };

    PerfArgDef(const char *argName, const char* typeName, const char* desc,
        void* valuePtr, int valueLen) {
        strcpy(this->argName, argName);
        strcpy(this->typeName, typeName);
        strcpy(this->desc, desc);
        this->valuePtr = valuePtr;
        this->valueLen = valueLen;
    }
    virtual ~PerfArgDef() {}

    // use "string" instead of "fixed char array"?
    char    argName[PERFARGDEF_ARGNAME_MAXSIZE];
    char    typeName[PERFARGDEF_TYPENAME_MAXSIZE];
    char    desc[PERFARGDEF_DESC_MAXSIZE];
    void*   valuePtr;
    int     valueLen;
};
#endif /* PERFARGDEF_H */
