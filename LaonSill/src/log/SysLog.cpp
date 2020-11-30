/**
 * @file SysLog.cpp
 * @date 2016-10-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <assert.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>
#include <execinfo.h>
#include <cxxabi.h>

#include "Param.h"
#include "FileMgmt.h"
#include "SysLog.h"

using namespace std;

#define gettid()    syscall(SYS_gettid)   // there is no glibc wrapper for this system call;;

extern const char*  LAONSILL_HOME_ENVNAME;
const char*         SysLog::sysLogFileName = {"sys.log"};
FILE*               SysLog::fp = NULL;
mutex               SysLog::logMutex;

void SysLog::init() {
    char sysLogFilePath[PATH_MAX];
    char sysLogDir[PATH_MAX];

    if (strcmp(SPARAM(SYSLOG_DIR), "") == 0) {
        assert(sprintf(sysLogDir, "%s/log", getenv(LAONSILL_HOME_ENVNAME)) != -1);
        assert(sprintf(sysLogFilePath, "%s/log/%s", getenv(LAONSILL_HOME_ENVNAME),
                    SysLog::sysLogFileName) != -1);
    } else {
        assert(sprintf(sysLogDir, "%s", SPARAM(SYSLOG_DIR)) != -1);
        assert(sprintf(sysLogFilePath, "%s/%s", SPARAM(SYSLOG_DIR),
            SysLog::sysLogFileName) != -1);
    }

    FileMgmt::checkDir(sysLogDir);

    assert(SysLog::fp == NULL);
    SysLog::fp = fopen(sysLogFilePath, "a");
}

void SysLog::destroy() {
    assert(SysLog::fp);
    fflush(SysLog::fp);
    fclose(SysLog::fp);
    SysLog::fp = NULL;
}

const int SMART_FILENAME_OFFSET = 7;
void SysLog::writeLogHeader(const char* fileName, int lineNum) {
    struct timeval      val;
    struct tm*          tmPtr;
    char                filePath[PATH_MAX];

    assert(SysLog::fp);

    gettimeofday(&val, NULL);
    tmPtr = localtime(&val.tv_sec);
    assert(strlen(fileName) > SMART_FILENAME_OFFSET);
    strcpy(filePath, fileName + SMART_FILENAME_OFFSET); // in order to get rid of "../src/"

    fprintf(SysLog::fp, "[%04d/%02d/%02d %02d:%02d:%02d.%06ld@%s:%d(%d/%d)] ",
        tmPtr->tm_year + 1900, tmPtr->tm_mon + 1,
        tmPtr->tm_mday, tmPtr->tm_hour, tmPtr->tm_min, tmPtr->tm_sec, val.tv_usec,
        filePath, lineNum, (int)getpid(), ((int)gettid() - (int)getpid())); 
}

const int MAX_FRAMES = 256;
/**
 * thanks to Timo Bingmann
 * I referenced his/her source and the source is free to use regarding to his/her mention :)
 * https://panthema.net/2008/0901-stacktrace-demangled/
 */
void SysLog::printStackTrace(FILE* fp) {
    fprintf(fp, "stack trace:\n");

    // storage array for stack trace address data
    void* addrlist[MAX_FRAMES+1];

    // retrieve current stack addresses
    int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

    if (addrlen == 0) {
        fprintf(fp, "  <empty, possibly corrupt>\n");
        return;
    }

    // resolve addresses into strings containing "filename(function+address)",
    // this array must be free()-ed
    char** symbollist = backtrace_symbols(addrlist, addrlen);

    // allocate string which will be filled with the demangled function name
    size_t funcnamesize = 256;
    char* funcname = (char*)malloc(funcnamesize);
    char* syscom = (char*)malloc(funcnamesize);
    char* buf = (char*)malloc(1024);
    FILE* ptr;

    // iterate over the returned symbol lines. skip the first, it is the
    // address of this function.
    for (int i = 1; i < addrlen; i++) {
        char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

        // find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char *p = symbollist[i]; *p; ++p) {
            if (*p == '(')
                begin_name = p;
            else if (*p == '+')
                begin_offset = p;
            else if (*p == ')' && begin_offset) {
                end_offset = p;
                break;
            }
        }

        if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
            *begin_name++ = '\0';
            *begin_offset++ = '\0';
            *end_offset = '\0';

            // mangled name is now in [begin_name, begin_offset) and caller
            // offset in [begin_offset, end_offset). now apply
            // __cxa_demangle():

            int status;
            char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
            if (status == 0) {
                funcname = ret; // use possibly realloc()-ed string
                fprintf(fp, "  %s : %s+%s\n", symbollist[i], funcname, begin_offset);
            } else {
                // demangling failed. Oujtput function name as a C function with
                // no arguments.
                fprintf(fp, "  %s : %s()+%s\n", symbollist[i], begin_name, begin_offset);
            }

            sprintf(syscom, "addr2line %p -e %s", addrlist[i], symbollist[i]);
            if ((ptr = popen(syscom, "r")) != NULL) {
            	while (fgets(buf, funcnamesize, ptr) != NULL) {
            		fprintf(fp, "  \t%s", buf);
            	}
            	pclose(ptr);
            }
        }
        else
        {
            // couldn't parse the line? print the whole line.
            fprintf(fp, "  %s\n", symbollist[i]);
        }
    }

    free(buf);
    free(syscom);
    free(funcname);
    free(symbollist);
}
