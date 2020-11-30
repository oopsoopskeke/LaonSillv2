/**
 * @file StdOutLog.cpp
 * @date 2016-11-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>

#include "StdOutLog.h"

using namespace std;

#define gettid()    syscall(SYS_gettid) // there is no glibc wrapper for this system call;;

mutex StdOutLog::logMutex;

void StdOutLog::writeLogHeader() {
    struct timeval      val;
    struct tm*          tmPtr;
    
    gettimeofday(&val, NULL);
    tmPtr = localtime(&val.tv_sec);

    fprintf(stdout, "[%04d/%02d/%02d %02d:%02d:%02d:%06ld(%d/%d)] ",
        tmPtr->tm_year + 1900, tmPtr->tm_mon + 1, tmPtr->tm_mday,
        tmPtr->tm_hour, tmPtr->tm_min, tmPtr->tm_sec, val.tv_usec, 
        (int)getpid(), ((int)gettid() - (int)getpid())); 
}
