/**
 * @file InputDataProvider.cpp
 * @date 2017-07-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "InputDataProvider.h"
#include "SysLog.h"
#include "Param.h"
#include "WorkContext.h"
#include "ColdLog.h"
#include "ThreadMgmt.h"
#include "PhysicalPlan.h"

#include "DataReader.h"
#include "Param.h"
#include "MemoryMgmt.h"

using namespace std;

map<InputPoolKey, InputPool*>   InputDataProvider::poolMap;
mutex                           InputDataProvider::poolMapMutex;

map<std::string, PoolInfo>      InputDataProvider::poolInfoMap;
mutex                           InputDataProvider::poolInfoMutex;

map<DRType, DRCBFuncs>          InputDataProvider::drFuncMap;

// FIXME: turn in into an auto-generated function
void InputDataProvider::init() {
    // (1) add Datum CB funcs
    DRCBFuncs funcDatum;
    funcDatum.allocFunc             = DataReader<class Datum>::allocElem;
    funcDatum.deallocFunc           = DataReader<class Datum>::deallocElem;
    funcDatum.fillFunc              = DataReader<class Datum>::fillElem;
    drFuncMap[DRType::DatumType]    = funcDatum;
}

// input layer에서 이 함수를 호출해서 pool을 등록해야 한다.
void InputDataProvider::addPool(string networkID, int dopID, string layerName, DRType drType,
    void* reader) {
    // (1) register inputPool
    InputPoolKey inputPoolKey;
    inputPoolKey.networkID = networkID;
    inputPoolKey.dopID = dopID;
    inputPoolKey.layerName = layerName;

    SASSERT0(drFuncMap.find(drType) != drFuncMap.end());
    DRCBFuncs funcs = drFuncMap[drType];

    // build network 시의 reshape()를 대비하여 1개의 원소는 미리 읽어 둔다.
    InputPool* newInputPool = NULL;
    SNEW(newInputPool, InputPool);
    SASSUME0(newInputPool != NULL);

    newInputPool->head = 0;
    newInputPool->tail = 1;
    newInputPool->remainElemCnt = SPARAM(INPUT_DATA_PROVIDER_ELEM_COUNT) - 1;
    newInputPool->elemCnt = SPARAM(INPUT_DATA_PROVIDER_ELEM_COUNT);
    newInputPool->activeElemCnt = 1;
    for (int j = 0; j < newInputPool->elemCnt; j++) {
        void* elemPtr;
        funcs.allocFunc(&elemPtr);
        newInputPool->elemArray.push_back(elemPtr);
    }
    newInputPool->reader = reader;
    newInputPool->drType = drType;
    newInputPool->waiterTID = -1;
    funcs.fillFunc(newInputPool->reader, newInputPool->elemArray[0]);

    unique_lock<mutex> poolMapLock(poolMapMutex);
    SASSERT0(poolMap.find(inputPoolKey) == poolMap.end());
    poolMap[inputPoolKey] = newInputPool;
    poolMapLock.unlock();

    // (2) register poolInfo
    PoolInfo poolInfo;
    poolInfo.networkID = networkID;
    poolInfo.threadID = -1;
    poolInfo.cleanupThreadID = -1;

    unique_lock<mutex> poolInfoLock(poolInfoMutex);
    if (poolInfoMap.find(networkID) == poolInfoMap.end()) {
        poolInfoMap[networkID] = poolInfo;
    }

    poolInfoMap[networkID].inputPoolKeyList.push_back(inputPoolKey);
}

// 이 함수는 네트워크 destory시에 호출을 해야 한다.
void InputDataProvider::removePool(string networkID) {
    PoolInfo* poolInfo;
    unique_lock<mutex> poolInfoLock(poolInfoMutex);
    SASSERT0(poolInfoMap.find(networkID) != poolInfoMap.end());
    poolInfoMap[networkID].cleanupThreadID = WorkContext::curThreadID;
    poolInfo = &poolInfoMap[networkID];
    poolInfoLock.unlock();

    if (poolInfo->threadID != -1)  {
        ThreadMgmt::signal(poolInfo->threadID, ThreadEvent::FinishJob);
        ThreadMgmt::wait(WorkContext::curThreadID, 0UL);
    }

    for (int i = 0; i < poolInfo->inputPoolKeyList.size(); i++) {
        InputPoolKey inputPoolKey = poolInfo->inputPoolKeyList[i];

        unique_lock<mutex> inputPoolLock(poolMapMutex);
        SASSERT0(poolMap.find(inputPoolKey) != poolMap.end());
        InputPool* inputPool = poolMap[inputPoolKey];
        poolMap.erase(inputPoolKey);
        inputPoolLock.unlock();

        SASSERT0(drFuncMap.find(inputPool->drType) != drFuncMap.end());
        DRCBFuncs funcs = drFuncMap[inputPool->drType];

        SASSERT0(inputPool->waiterTID == -1);
        
        vector<void*>::iterator iter = inputPool->elemArray.begin();
        while (iter != inputPool->elemArray.end()) {
            void* removingElem = (*iter);
            funcs.deallocFunc(removingElem);
            iter = inputPool->elemArray.erase(iter);
        }

        SDELETE(inputPool);
    }

    poolInfoLock.lock();
    SASSERT0(poolInfoMap.find(networkID) != poolInfoMap.end());
    poolInfoMap.erase(networkID);
    poolInfoLock.unlock();
}

InputPool* InputDataProvider::getInputPool(string networkID, int dopID, string layerName) {
    InputPoolKey inputPoolKey;
    inputPoolKey.networkID = networkID;
    inputPoolKey.dopID = dopID;
    inputPoolKey.layerName = layerName;

    InputPool* result;

    unique_lock<mutex> poolMapLock(poolMapMutex);
    SASSERT0(poolMap.find(inputPoolKey) != poolMap.end());
    result = poolMap[inputPoolKey];
    poolMapLock.unlock();

    return result;
}

void* InputDataProvider::getData(InputPool* pool, bool peek) {
    SASSUME0(SPARAM(USE_INPUT_DATA_PROVIDER));

    void *data = NULL;
    while (true) {
        unique_lock<mutex> datumLock(pool->mutex);
        if (pool->activeElemCnt == 0) {
            if (SPARAM(INPUT_DATA_PROVIDER_BLOCKING)) {
                pool->waiterTID = WorkContext::curThreadID;
            }
            datumLock.unlock();

            COLD_LOG(ColdLog::WARNING, true, "The available data pool is empty.");

            if (!SPARAM(INPUT_DATA_PROVIDER_BLOCKING))
                return NULL;

            ThreadMgmt::wait(WorkContext::curThreadID,
                SPARAM(INPUT_DATA_PROVIDER_CALLER_WAIT_TIME_MS));
        } else {
            data = pool->elemArray[pool->head];

            if (!peek) {
                pool->head = (pool->head + 1) % pool->elemCnt;
                pool->activeElemCnt -= 1;
                pool->remainElemCnt += 1;
            }
            pool->waiterTID = -1;
            datumLock.unlock();

            break; 
        }
    }

    SASSUME0(data != NULL);
    return data;
}

void InputDataProvider::handleIDP(string networkID) {
    int timeout = SPARAM(INPUT_DATA_PROVIDER_WAIT_TIME_MS);

    unique_lock<mutex> poolInfoLock(poolInfoMutex);
    if (poolInfoMap.find(networkID) == poolInfoMap.end()) {
        // input data provider job이 시작이 되기 전에 사용자가 네트워크를 destroy 하였을
        // 경우에 발생할 수 있다. 이걸 예외처리해야 할지 아니면 허용해야 할지 고민..
        COLD_LOG(ColdLog::WARNING, true,
            "The specified network has been destroyed or has not yet been registered."
            " networkID=%s", networkID.c_str());
    }
    PoolInfo *poolInfo = &poolInfoMap[networkID];

    if (poolInfo->threadID != -1) {
        COLD_LOG(ColdLog::WARNING, true, "IDP is already started. networkID=%s",
            networkID.c_str());
        return;
    }

    poolInfo->threadID = WorkContext::curThreadID;
    poolInfoLock.unlock();

    vector<InputPool*> inputPools;

    unique_lock<mutex> poolMapLock(poolMapMutex);
    for (int i = 0 ; i < poolInfo->inputPoolKeyList.size(); i++) {
        InputPoolKey inputPoolKey = poolInfo->inputPoolKeyList[i];
        SASSERT0(poolMap.find(inputPoolKey) != poolMap.end());
        inputPools.push_back(poolMap[inputPoolKey]); 
    }
    poolMapLock.unlock();

    handler(inputPools);

    while (true) {
        //ThreadEvent event = ThreadMgmt::wait(WorkContext::curThreadID, timeout); 
        ThreadEvent event = ThreadMgmt::wait(WorkContext::curThreadID, 0UL);
        if (event & Halt) {
            // 이러한 상황에서.. 메모리는 누가 해제해야 하나? 
            // 어차피 프로세스 종료니까 heap에 올라간 메모리 해제에 대해서 
            // 신경쓰지 않아도 되긴 하는데..
            // 그래도 깔끔하게 처리할지 말지 고민중.
            break;
        } else if (event & FinishJob) {
            int threadID;
            unique_lock<mutex> poolInfoLock(poolInfoMutex);
            SASSERT0(poolInfoMap.find(networkID) != poolInfoMap.end());
            threadID = poolInfoMap[networkID].cleanupThreadID;
            poolInfoLock.unlock();

            SASSERT0(threadID != -1);
            ThreadMgmt::signal(threadID, ThreadEvent::Wakeup);
            break;
        }

        handler(inputPools);
    }
}

// handler()의 종료조건이 명확하지 않다. 현재는 계속 일을하다가 network destroy 시에 종료가
// 되는 형태이다.
void InputDataProvider::handler(vector<InputPool*> inputPools) {
    while (true) {
        bool hasProgress = false;

        for (int i = 0; i < inputPools.size(); i++) {
            InputPool* pool = inputPools[i];
           
            unique_lock<mutex> elemPoolLock(pool->mutex);
            int remainElemCnt = pool->remainElemCnt;
            elemPoolLock.unlock();

            if (remainElemCnt == 0)
                continue;

            hasProgress = true;

            SASSUME0(drFuncMap.find(pool->drType) != drFuncMap.end());
            DRCBFuncs funcs = drFuncMap[pool->drType];

            for (int j = 0; j < remainElemCnt; j++) {
                int elemIndex = (pool->tail + j) % pool->elemCnt;
                void* elemPtr = pool->elemArray[elemIndex];
                funcs.fillFunc(pool->reader, elemPtr);
            }

            elemPoolLock.lock();
            pool->remainElemCnt -= remainElemCnt;
            pool->activeElemCnt += remainElemCnt;
            pool->tail = (pool->tail + remainElemCnt) % pool->elemCnt;
            int waiterTID = pool->waiterTID;
            pool->waiterTID = -1;
            elemPoolLock.unlock();

            if (waiterTID != -1)
                ThreadMgmt::signal(waiterTID, ThreadEvent::Wakeup);
        }

        if (!hasProgress)
            break;
    }
}
