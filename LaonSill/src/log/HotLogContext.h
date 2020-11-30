/**
 * @file HotLogContext.h
 * @date 2016-11-01
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef HOTLOGCONTEXT_H
#define HOTLOGCONTEXT_H 

#include <stdlib.h>
#include <assert.h>
#include <aio.h>

#include <atomic>

#include "Param.h"

class HotLogContext {
public: 
                                    HotLogContext(int fd);
    virtual                        ~HotLogContext();
   
    volatile std::atomic<uint64_t>  remainSize;
    volatile uint64_t               logOffset;
    uint64_t                        diskOffset;
    uint64_t                        diskGenNum;

    int                             fd;
    char*                           buffer;
    int                             failCount;

    struct aiocb                    aiocb;
    struct aiocb                    aiocbWA;
    uint64_t                        releaseSize;

    bool                            fillFlushInfo(bool force, bool& isWrapAround);
    bool                            checkMem(uint64_t len);
    void                            writeMem(char* buffer, uint64_t offset, uint64_t len);
    void                            updateMem(uint64_t len);
};

#endif /* HOTLOGCONTEXT_H */
