/**
 * @file Sender.cpp
 * @date 2017-06-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <thread>

#include "Sender.h"
#include "ColdLog.h"
#include "HotLog.h"
#include "ThreadMgmt.h"
#include "Param.h"
#include "MemoryMgmt.h"
#include "SysLog.h"

using namespace std;

thread* Sender::sender;

void Sender::senderThread() {
    int threadID = ThreadMgmt::getThreadID(ThreadType::Sender, 0);
    ThreadMgmt::setThreadReady(threadID);
    COLD_LOG(ColdLog::INFO, true, "sender thread starts");
    
    HotLog::initForThread();

    // (1) 서버가 준비 될때까지 대기 한다.
    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }


    // (2) 메인 루프
    while (true) {
        ThreadEvent event =
            ThreadMgmt::wait(threadID, SPARAM(PRODUCER_PERIODIC_CHECK_TIME_MS)); 

        // Do Something
        
        if (event == ThreadEvent::Halt)
            break;
    }

    COLD_LOG(ColdLog::INFO, true, "sender thread ends");
    HotLog::markExit();
}

void Sender::launchThread() {
	// (1) receiver 쓰레드를 생성한다.
    Sender::sender = NULL;
    SNEW_ONCE(Sender::sender, thread, senderThread);
    SASSUME0(Sender::sender != NULL);
}
