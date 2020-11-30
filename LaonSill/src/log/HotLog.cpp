/**
 * @file HotLog.cpp
 * @date 2016-10-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <assert.h>
#include <limits.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <fcntl.h>

#include <chrono>

#include "Param.h"
#include "FileMgmt.h"
#include "HotLog.h"
#include "SysLog.h"
#include "ColdLog.h"
#include "MemoryMgmt.h"

using namespace std;

#define gettid()    syscall(SYS_gettid)    // there is no glibc wrapper for this system call;;

vector<HotLogContext*>      HotLog::contextArray;
mutex                       HotLog::contextMutex;

thread*                     HotLog::flusher;
mutex                       HotLog::flusherMutex;
condition_variable          HotLog::flusherCondVar;
bool                        HotLog::flusherHalting = false;

thread_local int            HotLog::contextId;
volatile int                HotLog::contextGenId = 0;

struct aiocb**              HotLog::flushCBs;

extern const char*          LAONSILL_HOME_ENVNAME;

void HotLog::init() {
    // (1) 디렉토리를 체크하고 없으면 생성 한다.
    char hotLogDir[PATH_MAX];

    if (strcmp(SPARAM(HOTLOG_DIR), "") == 0) {
        SASSERT0((sprintf(hotLogDir, "%s/log", getenv(LAONSILL_HOME_ENVNAME)) != -1));
    } else {
        SASSERT0((sprintf(hotLogDir, "%s", SPARAM(HOTLOG_DIR)) != -1));
    }

    FileMgmt::checkDir(hotLogDir);
}

void HotLog::launchThread(int threadCount) {
    // (1) flusher 쓰레드를 생성한다.
    HotLog::flusher = NULL;
    SNEW_ONCE(HotLog::flusher, thread, HotLog::flusherThread, threadCount);
}

int HotLog::initForThread() {
    // (1) log file을 생성한다. 파일명은 pid.tid 형태이다.
    char    hotLogFilePath[PATH_MAX];
    int     pid = (int)getpid();
    int     tid = (int)gettid() - pid;

    if (strcmp(SPARAM(SYSLOG_DIR), "") == 0) {
        SASSERT(sprintf(hotLogFilePath, "%s/log/hot.%d.%d", getenv(LAONSILL_HOME_ENVNAME),
                        pid, tid) != -1, "");
    } else {
        SASSERT(sprintf(hotLogFilePath, "%s/hot.%d.%d",
                    SPARAM(HOTLOG_DIR), pid, tid) != -1, "");
    }

    int fd = FileMgmt::openFile(hotLogFilePath, O_CREAT|O_TRUNC|O_RDWR|O_DIRECT);
    SASSERT(fd != -1, "");

    // (2) contextArray에 자신의 fd를 추가한다.
    HotLogContext* newContext = NULL;
    SNEW_ONCE(newContext, HotLogContext, fd);
    unique_lock<mutex> fdLock(HotLog::contextMutex);
    HotLog::contextArray.push_back(newContext);
    HotLog::contextId = HotLog::contextGenId;
    HotLog::contextGenId += 1;
}

void HotLog::markExit() {
    HOT_LOG(0, HotLog::getHotCodeContext()->failCount);
}

void HotLog::destroy() {
    // XXX: 이 함수가 호출이 되는 시점에 다른 모든 쓰레드가 종료가 되었음이 보증 되어야 한다.
    // (1) flusher thread를 종료한다.
    HotLog::flusherHalting = true;
    HotLog::flusherCondVar.notify_one();

    HotLog::flusher->join();
    SDELETE(HotLog::flusher);
    HotLog::flusher = NULL;
}

HotLogContext* HotLog::getHotCodeContext() {
    return HotLog::contextArray[HotLog::contextId];
}

void HotLog::doFlush(bool force) {
    int aioCount = 0;

    // (1) I/O를 할 녀석들을 찾는다.
    typename vector<HotLogContext*>::iterator iter;
    for (iter = HotLog::contextArray.begin(); iter != HotLog::contextArray.end(); iter++) {
        bool            isWrapAround;
        HotLogContext*  context = (HotLogContext*)(*iter);

        bool doFlush;
        doFlush = context->fillFlushInfo(force, isWrapAround);

        if (!doFlush)
            continue;

        HotLog::flushCBs[aioCount] = &context->aiocb;
        aioCount++;

        if (isWrapAround) {
            HotLog::flushCBs[aioCount] = &context->aiocbWA;
            aioCount++;
        }
    }

    if (aioCount == 0)
        return;

    // (2) AIO를 수행한다.
    //  TODO: 물론 AIO가 끝난 HotLogContext의 memory buffer의 remainSize를 미리 확보하면
    //      좋긴 하다. 하지만, 본 쓰레드는 background thread인데 그렇게 열심히 일을 할 필요가
    //      있는지 의문이다. 나중에 여유있을 때에 고민해보자.
    //       또한 LIO_NOWAIT모드가 필요한지, 이 경우에 개별 AIO control block의 완료를 개별로 
    //      signal 통보를 받을 필요가 있을지 생각해보자.
    if (lio_listio( LIO_WAIT, HotLog::flushCBs, aioCount, NULL) == -1) {
        int err = errno;

        if (errno == EAGAIN) {
            SYS_LOG("lio_listio() EAGAIN error. please check AIO_MAX. "
                    "current max AIO count=%d, issued AIO count=%d",
                    (HotLog::contextGenId * 2), aioCount);
            SASSERT(0, "");
        } else if (errno == EINVAL) {
            SYS_LOG("lio_listio() EINVAL error. please check AIO_LISTIO_MAX. "
                    "current max AIO count=%d, issued AIO count=%d",
                    (HotLog::contextGenId * 2), aioCount);
            SASSERT(0, "");
        } else if (errno == EINTR) {
            // XXX: aio_suspend()
            COLD_LOG(ColdLog::WARNING, true, "lio_listio() EINTR error. call aio_suspend()");
            while (true) {
                int ret = aio_suspend(HotLog::flushCBs, aioCount, NULL);

                if (ret == 0)
                    break;

                err = errno;
                SASSERT(ret == -1, "");

                if (err == EINTR) {
                    // XXX: is it right?
                    continue;
                } else {
                    // ENOSYS는 생각하지 않겠다. glibc 2.1이상을 쓰도록 명시하자.
                    // EAGAIN is called when timeout is specified
                    COLD_LOG(ColdLog::ERROR, true, "lio_suspend() failed. errno=%d", err);
                    SASSERT(0, "");
                }
            }
        } else if (errno == EIO) {
            SYS_LOG("One of more of the operations specified by aiocb_list failed.");
            for (int i = 0; i < aioCount; i++) {
                int aioErr = aio_error(HotLog::flushCBs[i]);
                COLD_LOG(ColdLog::ERROR, (aioErr != 0),
                    "lio_listio() occurs an error. error code=%d", aioErr);
            }
            SASSERT(0, "");
        } else {
            COLD_LOG(ColdLog::ERROR, true, "lio_listio() failed. errno=%d", err);
            SASSERT(0, "");
        }
    }

    // (3) release remainSize
    for (iter = HotLog::contextArray.begin(); iter != HotLog::contextArray.end(); iter++) {
        HotLogContext* context = (HotLogContext*)(*iter);

        if (context->releaseSize != 0UL) {
            atomic_fetch_add(&context->remainSize, context->releaseSize);
            context->releaseSize = 0UL;
        }
    }
}

void HotLog::flusherThread(int contextCount) {
    // (1) contextArray가 다 추가되는 시점까지 대기한다.
    while (contextCount > HotLog::contextGenId) {
        sleep(1);
    }

    // (2) AIO control blocks 구조체를 생성한다.
    //     wrap around까지 고려해서 contextCount * 2개 만큼을 할당한다.
    HotLog::flushCBs = NULL;
    int allocSize = sizeof(struct aiocb*) * contextCount * 2;
    SMALLOC_ONCE(HotLog::flushCBs, struct aiocb*, allocSize);
    SASSERT0(HotLog::flushCBs != NULL);

    // (3) main loop
    //     특정 값(SPARAM(HOTLOG_FLUSH_THRESHOLD) * SPARAM(HOTLOG_BUFFERSIZE)) 이상 
    //     버퍼가 사용되면 디스크로 버퍼를 내리고, 내린 내용은 닦아 준다.
    while (!HotLog::flusherHalting) {
        unique_lock<mutex> flusherLock(HotLog::flusherMutex);
        HotLog::flusherCondVar.wait_for(flusherLock,
            chrono::milliseconds(SPARAM(HOTLOG_FLUSH_CYCLE_MSEC)));

        HotLog::doFlush(false);
    }

    // (4) 타이밍 이슈로 아직 디스크로 내려가지 않는 버퍼가 존재할 수 있다.
    //     다시한번 플러시 하자.
    HotLog::doFlush(true);

    // (5) 모든 리소스를 정리한다.
    // XXX: check whether erase() call destructor of container or not
    typename vector<HotLogContext*>::iterator iter = HotLog::contextArray.begin();
    typename vector<HotLogContext*>::iterator iterTmp;
    for (; iter != HotLog::contextArray.end(); iter++) {
        SDELETE(*iter);
    }

    SASSERT0(HotLog::flushCBs != NULL);
    SFREE(HotLog::flushCBs);
}
