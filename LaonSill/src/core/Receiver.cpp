/**
 * @file Receiver.cpp
 * @date 2017-06-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Receiver.h"
#include "ColdLog.h"
#include "HotLog.h"
#include "ThreadMgmt.h"
#include "Param.h"
#include "MemoryMgmt.h"
#include "SysLog.h"

using namespace std;

thread* Receiver::receiver;

void Receiver::receiverThread() {
    int threadID = ThreadMgmt::getThreadID(ThreadType::Receiver, 0);

    ThreadMgmt::setThreadReady(threadID);
    COLD_LOG(ColdLog::INFO, true, "receiver thread starts");
    
    HotLog::initForThread();

    // (1) 서버가 준비 될때까지 대기 한다.
    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }

    // (2) 메인 루프
    while (true) {
        ThreadEvent event =
            ThreadMgmt::wait(threadID, SPARAM(RECEIVER_PERIODIC_CHECK_TIME_MS)); 

        // Do Something
        if (event == ThreadEvent::Halt)
            break;
    }

    COLD_LOG(ColdLog::INFO, true, "receiver thread ends");
    HotLog::markExit();
}

void Receiver::launchThread() {
	// (1) receiver 쓰레드를 생성한다.
    Receiver::receiver = NULL;
    SNEW_ONCE(Receiver::receiver, thread, receiverThread);
    SASSERT0(Receiver::receiver != NULL);
}
