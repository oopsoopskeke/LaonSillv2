/**
 * @file HotLog.h
 * @date 2016-10-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef HOTLOG_H
#define HOTLOG_H 

#include <mutex>
#include <vector>
#include <thread>
#include <condition_variable>

#include "common.h"
#include "HotLogContext.h"
#include "HotCode.h"

#define HOT_LOG(eventId, ...)           HotCode::HOT_LOG##eventId(__VA_ARGS__)

class HotLog {
public: 
                                        HotLog() {}
    virtual                            ~HotLog() {}
    static void                         init();
    static int                          initForThread();
    static void                         markExit();
    static void                         launchThread(int threadCount);
    static void                         destroy();

    static HotLogContext*               getHotCodeContext();

private:
    static std::vector<HotLogContext*>  contextArray;
    static std::mutex                   contextMutex;

    static void                         doFlush(bool force);
    static void                         flusherThread(int contextCount);

    static std::thread*                 flusher;
    static std::mutex                   flusherMutex;
    static std::condition_variable      flusherCondVar;
    static bool                         flusherHalting;

    static thread_local int             contextId;
    static volatile int                 contextGenId;

    static struct aiocb**               flushCBs;
};

#endif /* HOTLOG_H */
