/**
 * @file SessContext.h
 * @date 2016-10-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef SESSCONTEXT_H
#define SESSCONTEXT_H 

#include <thread>
#include <mutex>
#include <condition_variable>

#include "common.h"

class SessContext {
public:
    SessContext(int sessId) {
        this->sessId = sessId;
        this->fd = -1;
        this->running = false;
        this->active = false;
    }
    virtual                    ~SessContext() {}
    int                         sessId;
    std::mutex                  sessMutex;
    std::condition_variable     sessCondVar;
    int                         fd;
    bool                        running;
    bool                        active;     // should be run
};

#endif /* SESSCONTEXT_H */
