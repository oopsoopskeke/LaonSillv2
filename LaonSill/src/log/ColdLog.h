/**
 * @file ColdLog.h
 * @date 2016-10-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef COLDLOG_H
#define COLDLOG_H 

#include <mutex>

#include "common.h"
#include "NetworkRecorder.h"

// XXX: 아무리 cold log지만 너무 lock이 긴게 아닌지... 
//      나중에 고민 해보자.
// XXX: logLock unlock 할필요 없어보이긴 한데.. 일단 해두자.
// XXX: fflush 타이밍을 고민해보자.
// fflush() is thread-safe
#define COLD_LOG(level, cond, fmt, args...)                                             \
    do {                                                                                \
        if ((level >= ColdLog::logLevel) && (cond)) {                                   \
            SEVENT_PUSH((NetworkEventType)level, fmt, ##args);                          \
            std::unique_lock<std::mutex>  logLock(ColdLog::logMutex);                   \
            ColdLog::writeLogHeader(level, __FILE__, __LINE__);                         \
            fprintf(ColdLog::fp, fmt, ##args);                                          \
            fprintf(ColdLog::fp, "\n");                                                 \
            logLock.unlock();                                                           \
            fflush(ColdLog::fp);                                                        \
        }                                                                               \
    } while (0)

#define COLD_LOG0(level, cond, fmt)                                                     \
    do {                                                                                \
        if ((level >= ColdLog::logLevel) && (cond)) {                                   \
            SEVENT_PUSH0((NetworkEventType)level);                                      \
            std::unique_lock<std::mutex>  logLock(ColdLog::logMutex);                   \
            ColdLog::writeLogHeader(level, __FILE__, __LINE__);                         \
            fprintf(ColdLog::fp, fmt);                                                  \
            fprintf(ColdLog::fp, "\n");                                                 \
            logLock.unlock();                                                           \
            fflush(ColdLog::fp);                                                        \
        }                                                                               \
    } while (0)

class ColdLog {
public: 
    // XXX: 일반적으로는 INFO가 WARNING보다 log level이 낮음.
    //      하지만 개인적인 취향으로 INFO를 더 높게 하였음.
    //      WARNING은 불필요하게 너무 많이 찍히는 경향이 있기 때문임.
    enum LogLevel : int {
        DEBUG = 0,
        WARNING,
        INFO,
        ERROR
    };

                        ColdLog() {}
    virtual            ~ColdLog() {}

    static void         init();
    static void         destroy();
    static FILE*        fp;
    static std::mutex   logMutex;
    static int          logLevel;
    static void         writeLogHeader(LogLevel logLevel, const char* fileName, int lineNum);

private:
    static const char*  coldLogFileName;
    static const char*  getLogLevelStr(LogLevel logLevel);
};

#endif /* COLDLOG_H */
