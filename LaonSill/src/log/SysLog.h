/**
 * @file SysLog.h
 * @date 2016-10-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef SYSLOG_H
#define SYSLOG_H 

#include <mutex>

#include "common.h"
#include "Param.h"
#include "NetworkRecorder.h"

#define SYS_LOG(fmt, args...)                                                       \
    do {                                                                            \
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eSYSTEM, fmt, ##args);                       \
        if (SysLog::fp) {                                                           \
            std::unique_lock<std::mutex>  logLock(SysLog::logMutex);                \
            SysLog::writeLogHeader(__FILE__,__LINE__);                              \
            fprintf(SysLog::fp, fmt, ##args);                                       \
            fprintf(SysLog::fp, "\n");                                              \
            logLock.unlock();                                                       \
            fflush(SysLog::fp);                                                     \
        } else {                                                                    \
            std::unique_lock<std::mutex>  logLock(SysLog::logMutex);                \
            fprintf(stderr, fmt, ##args);                                           \
            fprintf(stderr, "\n");                                                  \
            logLock.unlock();                                                       \
            fflush(stderr);                                                         \
        }                                                                           \
    } while (0)

#define SASSERT(cond, fmt, args...)                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            SEVENT_PUSH(NETWORK_EVENT_TYPE_eASSERT, fmt, ##args);                   \
            if (SysLog::fp) {                                                       \
                std::unique_lock<std::mutex>  logLock(SysLog::logMutex);            \
                SysLog::writeLogHeader(__FILE__,__LINE__);                          \
                fprintf(SysLog::fp, fmt, ##args);                                   \
                fprintf(SysLog::fp, "\n");                                          \
                SysLog::printStackTrace(SysLog::fp);                                \
                logLock.unlock();                                                   \
                fflush(SysLog::fp);                                                 \
            } else {                                                                \
                std::unique_lock<std::mutex>  logLock(SysLog::logMutex);            \
                fprintf(stderr, fmt, ##args);                                       \
                fprintf(stderr, "\n");                                              \
                SysLog::printStackTrace(stderr);                                    \
                logLock.unlock();                                                   \
                fflush(stderr);                                                     \
            }                                                                       \
            if (SPARAM(SLEEP_WHEN_ASSERTED))                                        \
                sleep(INT_MAX);                                                     \
            else                                                                    \
                assert(0);                                                          \
        }                                                                           \
    } while (0)

#define SASSERT0(cond)                                                              \
    do {                                                                            \
        if (!(cond)) {                                                              \
            SEVENT_PUSH0(NETWORK_EVENT_TYPE_eASSERT);                               \
            if (SysLog::fp) {                                                       \
                std::unique_lock<std::mutex>  logLock(SysLog::logMutex);            \
                SysLog::writeLogHeader(__FILE__,__LINE__);                          \
                fprintf(SysLog::fp, "\n");                                          \
                SysLog::printStackTrace(SysLog::fp);                                \
                logLock.unlock();                                                   \
                fflush(SysLog::fp);                                                 \
            } else {                                                                \
                std::unique_lock<std::mutex>  logLock(SysLog::logMutex);            \
                fprintf(stderr, "\n");                                              \
                SysLog::printStackTrace(stderr);                                    \
                logLock.unlock();                                                   \
                fflush(stderr);                                                     \
            }                                                                       \
            if (SPARAM(SLEEP_WHEN_ASSERTED))                                        \
                sleep(INT_MAX);                                                     \
            else                                                                    \
                assert(0);                                                          \
        }                                                                           \
    } while (0)

#ifdef DEBUG_MODE
#define SASSUME(cond, fmt, args...)			SASSERT(cond, fmt, ##args)
#define SASSUME0(cond)                      SASSERT0(cond)
#else
#define SASSUME(cond, fmt, args...)			Nop()
#define SASSUME0(cond)			            Nop()
#endif

class SysLog {
public:
                        SysLog() {}
    virtual            ~SysLog() {}
    static void         init();
    static void         destroy();
    static FILE*        fp;
    static std::mutex   logMutex;
    static void         writeLogHeader(const char* fileName, int lineNum);
    static void         printStackTrace(FILE* fp);
private:
    static const char*  sysLogFileName;
};

#endif /* SYSLOG_H */
