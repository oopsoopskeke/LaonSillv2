/**
 * @file ColdLog.cpp
 * @date 2016-10-30
 * @author moonhoen lee
 * @brief 
 * @details
 * @todo
 *  (1) TODO: need file size limitation function
 *  (2) TODO: need archive function
 */

#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "ColdLog.h"
#include "Param.h"
#include "FileMgmt.h"
#include "SysLog.h"

using namespace std;

#define gettid()    syscall(SYS_gettid)   // there is no glibc wrapper for this system call;;

extern const char*  LAONSILL_HOME_ENVNAME;
const char*         ColdLog::coldLogFileName = {"cold.log"};
FILE*               ColdLog::fp = NULL;
int                 ColdLog::logLevel;
mutex               ColdLog::logMutex;

void ColdLog::init() {
    char coldLogFilePath[PATH_MAX];
    char coldLogDir[PATH_MAX];

    if (strcmp(SPARAM(COLDLOG_DIR), "") == 0) {
        SASSERT0((sprintf(coldLogDir, "%s/log", getenv(LAONSILL_HOME_ENVNAME)) != -1));
        SASSERT0((sprintf(coldLogFilePath, "%s/log/%s", getenv(LAONSILL_HOME_ENVNAME),
                        ColdLog::coldLogFileName) != -1));
    } else {
        SASSERT0((sprintf(coldLogDir, "%s", SPARAM(COLDLOG_DIR)) != -1));
        SASSERT0((sprintf(coldLogFilePath, "%s/%s", SPARAM(COLDLOG_DIR),
            ColdLog::coldLogFileName) != -1));
    }

    FileMgmt::checkDir(coldLogDir);

    SASSERT0(!ColdLog::fp);
    ColdLog::fp = fopen(coldLogFilePath, "a");

    ColdLog::logLevel = SPARAM(COLDLOG_LEVEL);
}

void ColdLog::destroy() {
    SASSERT(ColdLog::fp, "");
    fflush(ColdLog::fp);
    fclose(ColdLog::fp);
    ColdLog::fp = NULL;
}

const char* ColdLog::getLogLevelStr(ColdLog::LogLevel logLevel) {
    switch (logLevel) {
    case ColdLog::DEBUG:
        return "DBG";
    case ColdLog::WARNING:
        return "WRN";
    case ColdLog::INFO:
        return "INF";
    case ColdLog::ERROR:
        return "ERR";
    default:
        SASSERT(0, "");
    }

    return "";      // meaningless
}

const int SMART_FILENAME_OFFSET = 7;
void ColdLog::writeLogHeader(ColdLog::LogLevel level, const char* fileName, int lineNum) {
    struct timeval      val;
    struct tm*          tmPtr;
    char                filePath[PATH_MAX];

    SASSERT(ColdLog::fp, "");

    gettimeofday(&val, NULL);
    tmPtr = localtime(&val.tv_sec);
    SASSERT(strlen(fileName) > SMART_FILENAME_OFFSET, "");
    strcpy(filePath, fileName + SMART_FILENAME_OFFSET); // in order to get rid of "../src/"

    fprintf(ColdLog::fp, "[%s|%04d/%02d/%02d %02d:%02d:%02d.%06ld@%s:%d(%d/%d)] ",
        ColdLog::getLogLevelStr(level), tmPtr->tm_year + 1900, tmPtr->tm_mon + 1,
        tmPtr->tm_mday, tmPtr->tm_hour, tmPtr->tm_min, tmPtr->tm_sec, val.tv_usec,
        filePath, lineNum, (int)getpid(), ((int)gettid() - (int)getpid())); 
}
