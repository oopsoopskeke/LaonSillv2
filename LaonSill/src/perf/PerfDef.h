/**
 * @file PerfDef.h
 * @date 2016-11-07
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PERFDEF_H
#define PERFDEF_H 

#include <string.h>

#include <vector>

#include "common.h"
#include "PerfArgDef.h"

#define PERFDEF_DESC_MAXSIZE                (256)

class PerfDef {
public: 
    PerfDef(const char* desc, bool jobScope, bool useTime, bool useAvgTime, bool useMaxTime,
        void* countPtr, void* timePtr, void* avgTimePtr, void* maxTimePtr, int argCount) {
        strcpy(this->desc, desc);
        this->jobScope = jobScope;
        this->useTime = useTime;
        this->useAvgTime = useAvgTime;
        this->useMaxTime = useMaxTime;
        this->countPtr = countPtr;
        this->timePtr = timePtr;
        this->avgTimePtr = avgTimePtr;
        this->maxTimePtr = maxTimePtr;
        this->argCount = argCount;
    }
    virtual ~PerfDef() {}

    bool    jobScope;
    bool    useTime;
    bool    useAvgTime;
    bool    useMaxTime;
    char    desc[PERFDEF_DESC_MAXSIZE];
    void*   countPtr;
    void*   timePtr;
    void*   avgTimePtr;
    void*   maxTimePtr;
    int     argCount;

    std::vector<PerfArgDef*> argArray;

    void addArgs(PerfArgDef* argDef) {
        this->argArray.push_back(argDef);
    }
};
#endif /* PERFDEF_H */
