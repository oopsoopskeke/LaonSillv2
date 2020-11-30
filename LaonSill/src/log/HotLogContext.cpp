/**
 * @file HotLogContext.cpp
 * @date 2016-11-02
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string.h>
#include <unistd.h>

#include <iostream>

#include "HotLogContext.h"
#include "SysLog.h"

using namespace std;

HotLogContext::HotLogContext(int fd) {
    this->remainSize = SPARAM(HOTLOG_BUFFERSIZE);
    this->logOffset = 0UL;
    this->diskOffset = 0UL;
    this->diskGenNum = 0UL;    

    this->fd = fd;

    SASSERT0(SPARAM(HOTLOG_SLOTSIZE) > 0);
    SASSERT0(SPARAM(HOTLOG_SLOTSIZE) % 512 == 0);
    SASSERT0(SPARAM(HOTLOG_BUFFERSIZE) > 0);
    SASSERT0(SPARAM(HOTLOG_BUFFERSIZE) % SPARAM(HOTLOG_SLOTSIZE) == 0);

    this->buffer = (char*)valloc(SPARAM(HOTLOG_BUFFERSIZE));
    SASSERT0(this->buffer);   // XXX: needs error handling
    memset(this->buffer, 0x00, SPARAM(HOTLOG_BUFFERSIZE));

    this->failCount = 0;
    this->releaseSize = 0;
}

HotLogContext::~HotLogContext() {
    SASSERT0(this->fd != -1);
    close(this->fd);

    SASSERT0(this->buffer); 
    free(this->buffer);
    this->buffer = NULL;
}

bool HotLogContext::checkMem(uint64_t len) {
    // check whether it has enough space or not
    if (len > atomic_load(&this->remainSize)) {
        this->failCount += 1;
        return false;
    } else {
        return true;
    }
}

/**
 * NOTE: 
 *  문자열의 경우에 정해진 길이 A에 일부분인 B만 채우면 나머지 부분인 (A-B)가 낭비 된다.
 *  문자열을 남길때에 문자열길이(4Byte) + 실제 문자열(B)을 남기는 방식으로 진행하는 방법은 
 *  A - B > 4 + B인 경우에 유용하다. 하지만 정말 빠른 로그에서 4Byte도 아까울 수 있다.
 *  따라서 개발자가 낭비되는 문자열이 없도록 잘 사용할 수 있을 것이라 믿고, 정해진 문자열 
 * 크기만큼을 로깅하도록 설계 하였다.
 */
void HotLogContext::writeMem(char* buffer, uint64_t offset, uint64_t len) {
    // (1) write mem
    int64_t lenWA = this->logOffset + offset + len - SPARAM(HOTLOG_BUFFERSIZE);

    if (lenWA <= 0) {
        memcpy((void*)(this->buffer + this->logOffset + offset), (void*)buffer, len);
    } else if (len > lenWA) {
        memcpy((void*)(this->buffer + this->logOffset + offset), (void*)buffer, len - lenWA);
        memcpy((void*)(this->buffer), (void*)(buffer + (len - lenWA)), len - lenWA);
    } else {
        memcpy((void*)(this->buffer + this->logOffset + offset - SPARAM(HOTLOG_BUFFERSIZE)),
            (void*)buffer, len);
    }
}

void HotLogContext::updateMem(uint64_t len) {
    this->logOffset = (this->logOffset + len) % SPARAM(HOTLOG_BUFFERSIZE);
    atomic_fetch_sub(&this->remainSize, len);
}

/**
 * @brief               fill aio control block
 * @param force         true이면 HOTLOG_FLUSH_SLOTCOUNT와 상관 없이 무조건 flush 한다.
 * @param isWrapAround  specify whether wrap around or not
 *
 * @return              flush 할 로그가 있으면 true, 없으면 false
 */
bool HotLogContext::fillFlushInfo(bool force, bool& isWrapAround) {
    uint64_t flushOffset, flushSize, flushOffsetWA, flushSizeWA;
    int     logOffset = this->logOffset;

    // (1) 플러시할 영역이 있는지를 확인
    if ((this->diskOffset == logOffset) && (atomic_load(&this->remainSize) == 0UL))
        return false;

    // (2) wrap around 여부를 결정
    if ((this->diskOffset == 0UL) || (logOffset == 0UL))
        isWrapAround = false;
    else if (logOffset <= this->diskOffset)
        isWrapAround = true;
    else
        isWrapAround = false;

    // (3) 플러싱할 영역을 계산
    uint64_t releaseSize = 0UL;
    uint64_t flushTotalSize = 0UL;
    if (isWrapAround) {
        flushOffset = this->diskGenNum * SPARAM(HOTLOG_BUFFERSIZE) + this->diskOffset;
        flushSize = SPARAM(HOTLOG_BUFFERSIZE) - this->diskOffset;
        releaseSize += flushSize;
        flushTotalSize += flushSize;
        SASSERT0(flushSize > 0UL);
        flushOffsetWA = (this->diskGenNum + 1UL) * SPARAM(HOTLOG_BUFFERSIZE);
        if (force)
            flushSizeWA = ALIGNUP(logOffset, SPARAM(HOTLOG_SLOTSIZE));
        else
            flushSizeWA = ALIGNDOWN(logOffset, SPARAM(HOTLOG_SLOTSIZE));
        releaseSize += ALIGNDOWN(logOffset, SPARAM(HOTLOG_SLOTSIZE));
        flushTotalSize += flushSizeWA;
    } else {
        flushOffset = this->diskGenNum * SPARAM(HOTLOG_BUFFERSIZE) + this->diskOffset;
        flushSize = logOffset - this->diskOffset;
        if (force)
            flushSize = ALIGNUP(flushSize, SPARAM(HOTLOG_SLOTSIZE));
        else
            flushSize = ALIGNDOWN(flushSize, SPARAM(HOTLOG_SLOTSIZE));
        releaseSize += ALIGNDOWN(flushSize, SPARAM(HOTLOG_SLOTSIZE));
        flushTotalSize += flushSize;
    }

    // (4) flush 여부를 결정하고, 반환
    if (flushTotalSize == 0UL)
        return false;

    if ((!force) &&
        (releaseSize < SPARAM(HOTLOG_FLUSH_SLOTCOUNT) * SPARAM(HOTLOG_SLOTSIZE))) {
        return false;
    }

    // (5) fill aiocb
    this->aiocb.aio_fildes = this->fd;
    this->aiocb.aio_buf = (char*)(this->buffer + this->diskOffset);
    this->aiocb.aio_nbytes = flushSize;
    this->aiocb.aio_offset = flushOffset;
    this->aiocb.aio_lio_opcode = LIO_WRITE;
    this->aiocb.aio_sigevent.sigev_notify = SIGEV_NONE;

    if (isWrapAround) {
        this->aiocbWA.aio_fildes = this->fd;
        this->aiocbWA.aio_buf = this->buffer;
        this->aiocbWA.aio_nbytes = flushSizeWA;
        this->aiocbWA.aio_offset = flushOffsetWA;
        this->aiocbWA.aio_lio_opcode = LIO_WRITE;
        this->aiocbWA.aio_sigevent.sigev_notify = SIGEV_NONE;
    }

    // (6) 변경될 diskOffset, diskGenNum을 미리 갱신한다.
    //     단, remainSize는 실제 디스크 플러시가 끝나고 갱신해야 한다.
    //     미리 remainSize를 늘려 놓으면 아직 디스크 플러시가 끝나지 않고, 현재
    //     I/O의 용도로 활용중인 버퍼를 오염시킬 수 있기 때문이다.
    //     remainSize를 위해서 this->releaseSize에 반환할 releaseSize를 저장해 놓는다.
    this->releaseSize = releaseSize;
    this->diskOffset =
        ALIGNDOWN(logOffset, SPARAM(HOTLOG_SLOTSIZE)) % SPARAM(HOTLOG_BUFFERSIZE);

    if (isWrapAround || this->diskOffset == 0UL) {
        this->diskGenNum += 1;
    }

    return true;
}
