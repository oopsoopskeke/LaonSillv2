/**
 * @file Broker.cpp
 * @date 2016-12-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "common.h"
#include "Broker.h"
#include "SysLog.h"
#include "Param.h"
#include "MemoryMgmt.h"

using namespace std;

map<uint64_t, Job*>     Broker::contentMap;
mutex                   Broker::contentMutex;

map<uint64_t, int>      Broker::waiterMap;
mutex                   Broker::waiterMutex;

Broker::AdhocSess      *Broker::adhocSessArray;
list<int>               Broker::freeAdhocSessIDList;
mutex                   Broker::adhocSessMutex;

void Broker::init() {
    int allocSize = sizeof(Broker::AdhocSess) * SPARAM(BROKER_ADHOC_SESS_COUNT);
    SMALLOC_ONCE(Broker::adhocSessArray, Broker::AdhocSess, allocSize);
    SASSERT0(Broker::adhocSessArray != NULL);

    for (int i = 0; i < SPARAM(BROKER_ADHOC_SESS_COUNT); i++) {
        Broker::adhocSessArray[i].arrayIdx = i;
        Broker::adhocSessArray[i].sessID = i + SPARAM(BROKER_ADHOC_SESS_STARTS);
        Broker::adhocSessArray[i].found = false;
        SNEW_ONCE(Broker::adhocSessArray[i].sessMutex, mutex);
        SASSERT0(Broker::adhocSessArray[i].sessMutex != NULL);
        SNEW_ONCE(Broker::adhocSessArray[i].sessCondVar, condition_variable);
        SASSERT0(Broker::adhocSessArray[i].sessCondVar != NULL);

        Broker::freeAdhocSessIDList.push_back(i + SPARAM(BROKER_ADHOC_SESS_STARTS));
    }
}

void Broker::destroy() {
    for (int i = 0; i < SPARAM(BROKER_ADHOC_SESS_COUNT); i++) {
        SDELETE(Broker::adhocSessArray[i].sessMutex);
        SDELETE(Broker::adhocSessArray[i].sessCondVar);
    }
   
    SFREE(Broker::adhocSessArray);
}

/**
 * XXX: 현재는 publisher가 직접 waiter를 깨우고 있다.
 *      publisher의 부하가 많은 수 있기 때문에 이 부분에 대한 설계를 고민해야 한다.
 *      (예를들어서 background 관리 쓰레드가 깨워준다던지..)
 */
Broker::BrokerRetType Broker::publishEx(int jobID, int taskID, Job *content) {
    // (1) content를 집어 넣는다.
    uint64_t key = MAKE_JOBTASK_KEY(jobID, taskID);
    unique_lock<mutex> contentLock(Broker::contentMutex);
    Broker::contentMap[key] = content;
    contentLock.unlock();

    // (2) 기다리고 있는 subscriber가 있으면 깨워준다.
    map<uint64_t, int>::iterator    waiterMapIter;
    unique_lock<mutex> waiterLock(Broker::waiterMutex);
    waiterMapIter = Broker::waiterMap.find(key);
    if (waiterMapIter != Broker::waiterMap.end()) {
        int sessID = (int)(waiterMapIter->second);
        Broker::waiterMap.erase(waiterMapIter);
        waiterLock.unlock();

        // wakeup with sessID
        int adhocSessIdx = sessID - SPARAM(BROKER_ADHOC_SESS_STARTS);
        SASSERT0(adhocSessIdx < SPARAM(BROKER_ADHOC_SESS_COUNT));
        Broker::AdhocSess *adhocSess = &Broker::adhocSessArray[adhocSessIdx];
        SASSERT0(adhocSess->arrayIdx == adhocSessIdx);

        adhocSess->found = true;
        unique_lock<mutex> sessLock(*adhocSess->sessMutex);
        adhocSess->sessCondVar->notify_one();
        sessLock.unlock();

    } else {
        waiterLock.unlock();
    }

    return Broker::Success;
}

Broker::BrokerRetType Broker::publish(int jobID, Job *content) {
    return Broker::publishEx(jobID, 0, content);
}

/**
 * NOTE:
 * subscribe에서 가져간 content는 가져간 사람이 지워야 한다.
 * content Lock을 잡고, waiter lock 혹은 adhocSess lock을 잡도록 되어 있다.
 * 순서가 바뀌면 안된다.
 */
Broker::BrokerRetType Broker::subscribeEx(int jobID, int taskID, Job **content,
        Broker::BrokerAccessType access) {
    SASSERT0(access < BrokerAccessTypeMax);

    uint64_t key = MAKE_JOBTASK_KEY(jobID, taskID);
    map<uint64_t, Job*>::iterator contentMapIter;
    unique_lock<mutex> contentLock(Broker::contentMutex);

    // XXX: content Lock을 너무 오래 잡고 있다.
    //      나중에 성능봐서 Lock 튜닝하자
    // (1) 콘텐트가 들어 있는지 확인하고, 들어 있으면 content에 포인터를 연결시켜주고 
    //    종료한다.
    contentMapIter = Broker::contentMap.find(key);
    if (contentMapIter != Broker::contentMap.end()) {
        (*content) = (Job*)(contentMapIter->second);
        Broker::contentMap.erase(contentMapIter);
        contentLock.unlock();
        return Broker::Success;
    }

    // (2) non blocking이면 곧바로 리턴
    if (access == Broker::NoBlocking) {
        contentLock.unlock();
        return Broker::NoContent;
    }

    // (3) adhoc sess ID를 발급해준다.
    //    (원래는 sess ID를 인자로 받고, sess ID가 없는 경우에 대해서 adhoc sess ID를
    //     발급해 주려고 하였다.)
    int sessID = Broker::getFreeAdhocSessID(); 
    if (sessID == 0) {
        contentLock.unlock();
        return Broker::NoFreeAdhocSess;
    }

    // (4) waiter에 등록해준다. 
    unique_lock<mutex> waiterLock(Broker::waiterMutex);
    Broker::waiterMap[key] = sessID;
    waiterLock.unlock();

    contentLock.unlock();

    // (5) 깨워줄때까지 대기한다.
    int adhocSessIdx = sessID - SPARAM(BROKER_ADHOC_SESS_STARTS);
    SASSERT0(adhocSessIdx < SPARAM(BROKER_ADHOC_SESS_COUNT));
    AdhocSess *adhocSess = &Broker::adhocSessArray[adhocSessIdx];
    SASSERT0(adhocSess->arrayIdx == adhocSessIdx);

    unique_lock<mutex> sessLock(*adhocSess->sessMutex);
    adhocSess->sessCondVar->wait(sessLock,
        [&adhocSess] { return (adhocSess->found == true); });
    sessLock.unlock();

    SASSERT0(adhocSess->found == true);

    SASSERT0(Broker::contentMap[key] != NULL);
    (*content) = (Job*)Broker::contentMap[key];

    // (6) adhoc sess를 반환한다.
    Broker::releaseAdhocSessID(sessID);

    return Broker::Success;
}

Broker::BrokerRetType Broker::subscribe(int jobID, Job **content,
        Broker::BrokerAccessType access) {
    return Broker::subscribeEx(jobID, 0, content, access);
}

/*
 * @return  0 : there is no adhoc sess id
 *          otherwise : adhoc sess id
 */
int Broker::getFreeAdhocSessID() {
    unique_lock<mutex> adhocSessLock(Broker::adhocSessMutex);
    if (Broker::freeAdhocSessIDList.empty()) {
        adhocSessLock.unlock();
        return 0;
    }

    int sessID = Broker::freeAdhocSessIDList.front();
    Broker::freeAdhocSessIDList.pop_front();
    adhocSessLock.unlock();

    return sessID;
}

void Broker::releaseAdhocSessID(int sessID) {
    SASSERT0(sessID >= SPARAM(BROKER_ADHOC_SESS_STARTS));
    int adhocSessArrayIdx = sessID - SPARAM(BROKER_ADHOC_SESS_STARTS);
    SASSERT0(adhocSessArrayIdx < SPARAM(BROKER_ADHOC_SESS_COUNT));

    Broker::adhocSessArray[adhocSessArrayIdx].found = false;

    unique_lock<mutex> adhocSessLock(Broker::adhocSessMutex);
    Broker::freeAdhocSessIDList.push_back(sessID);
    adhocSessLock.unlock();
}
