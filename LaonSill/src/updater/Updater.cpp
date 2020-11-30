/**
 * @file Updater.cpp
 * @date 2017-06-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Updater.h"
#include "SysLog.h"
#include "Worker.h"
#include "PhysicalPlan.h"
#include "WorkContext.h"
#include "MemoryMgmt.h"
#include "Param.h"

using namespace std;

map<UpdaterKey, UpdaterValue*>      Updater::updaterMap; 
mutex                               Updater::updaterMutex;

void Updater::addUpdater(string networkID, int layerID, int paramCount, int nodeID,
        int devID) {
   
    UpdaterKey key;
    key.networkID = networkID;
    key.layerID = layerID;

    UpdaterValue *value = NULL;
    SNEW(value, UpdaterValue);
    SASSERT0(value != NULL);
    value->nodeID = nodeID;
    value->devID = devID;

    for (int i = 0; i < paramCount; i++) {
        int paramType = i;
        Data<float>* updateData = NULL;
        Data<float>* updateDataHis1 = NULL;
        Data<float>* updateDataHis2 = NULL;

        SNEW(updateData, Data<float>, string("update_Data_") + 
                                           networkID + 
                                           string("_") + to_string(layerID) +
                                           string("_") + to_string(paramType) +
                                           string("_") + to_string(nodeID) +
                                           string("_") + to_string(devID));
        SASSUME0(updateData != NULL);

        SNEW(updateDataHis1, Data<float>, string("update_DataHis1_") + 
                                           networkID + 
                                           string("_") + to_string(layerID) +
                                           string("_") + to_string(paramType) +
                                           string("_") + to_string(nodeID) +
                                           string("_") + to_string(devID));
        SASSUME0(updateDataHis1 != NULL);

        SNEW(updateDataHis2, Data<float>, string("update_DataHis2_") + 
                                           networkID + 
                                           string("_") + to_string(layerID) +
                                           string("_") + to_string(paramType) +
                                           string("_") + to_string(nodeID) +
                                           string("_") + to_string(devID));
        SASSUME0(updateDataHis2 != NULL);


        value->tensorDataPtrs.push_back(updateData);
        value->tensorDataHis1Ptrs.push_back(updateDataHis1);
        value->tensorDataHis2Ptrs.push_back(updateDataHis2);
    }
    value->paramCount = paramCount;
    value->reshape = false;
    value->access = true;

    unique_lock<mutex> lock(updaterMutex);
    SASSERT0(updaterMap.find(key) == updaterMap.end()); 
    updaterMap[key] = value;
}

void Updater::unsetReshape(string networkID, int layerID) {
    UpdaterKey key;
    key.networkID = networkID;
    key.layerID = layerID;

    unique_lock<mutex> lock(updaterMutex);
    SASSUME0(updaterMap.find(key) != updaterMap.end());
    UpdaterValue* value = updaterMap[key];
    unique_lock<mutex> accessLock(value->mutex);
    value->reshape = false;
    accessLock.unlock();
}

// @return  false : cannot access tensor (locked)
//          true : can access tensor but it could not be done
//                 (for cluster or multi-device scenario)
bool Updater::updateParams(string networkID, int layerID, int planID, int dopID,
    vector<UpdateParam> updateParams, bool needSyncGrad) {

    Data<float>* tensorSourceParam;
    Data<float>* tensorTargetParam;
    Data<float>* tensorDataHis1;
    Data<float>* tensorDataHis2;

    int dopCount;
    if (!needSyncGrad) {
        // needSyncGrad가 false인 경우는 UpdaterTaskDefs를 Task Consumer가 실행하는
        // 경우이다. updateTaskDefs는 curPlanInfo가 아닐 수 있기 때문에 
        // 직접 planGlobalInfoMap으로 부터 dopCount를 가져와야 한다.
        dopCount = PhysicalPlan::getDOPCount(networkID);
    } else {
        dopCount = WorkContext::curPlanInfo->dopCount;
    }

    if (dopCount == 1) {
        for (int i = 0; i < updateParams.size(); i++) {
            tensorSourceParam = (Data<float>*)updateParams[i].paramDataPtr;
            tensorDataHis1 = (Data<float>*)updateParams[i].paramHis1Ptr;
            tensorDataHis2 = (Data<float>*)updateParams[i].paramHis2Ptr;

            Update<float>::updateParam(updateParams[i].context, tensorDataHis1,
                tensorDataHis2, tensorSourceParam);

            if (SPARAM(UPDATER_RESET_GRADIENT))
                tensorSourceParam->reset_device_grad();
        }

        PhysicalPlan::markFinish(networkID, dopID, planID);
        return true;
    }

    // param을 reset 해줘야 하는데 dopCount > 1 인 상황 대처가 안돼서 일단 막아둠
    // 보고 해결하실 수 있으신 분 처리 바람
    SASSERT(false, "");


    UpdaterKey key;
    key.networkID = networkID;
    key.layerID = layerID;

    unique_lock<mutex> updaterMapLock(updaterMutex);
    SASSUME0(updaterMap.find(key) != updaterMap.end());
    UpdaterValue *value = updaterMap[key];
    updaterMapLock.unlock();

    int bufSize = tensorTargetParam->getCount() * sizeof(float);

    // (1) 상황에 맞게 param의 gradient를 동기화한다.
    if (needSyncGrad) {
        if (value->nodeID != SPARAM(NODE_ID)) {
            // cluster mode
            SASSERT(false, "Not implemented yet");
        }

        bool needReshape;
        unique_lock<mutex> accessLock(value->mutex);
        if (!value->access) {
            return false;
        }
        value->access = false;
        needReshape = value->reshape;
        value->reshape = false;
        accessLock.unlock();
     
        if (value->reshape) {
            for (int i = 0; i < updateParams.size(); i++) {
                tensorSourceParam = (Data<float>*)updateParams[i].paramDataPtr;
                tensorTargetParam = (Data<float>*)value->tensorDataPtrs[i];
                tensorDataHis1 = (Data<float>*)value->tensorDataHis1Ptrs[i];
                tensorDataHis2 = (Data<float>*)value->tensorDataHis2Ptrs[i];

                tensorTargetParam->reshape(tensorSourceParam->getShape());
                tensorDataHis1->reshape(tensorSourceParam->getShape());
                tensorDataHis2->reshape(tensorSourceParam->getShape());
            }
        }

        // XXX: should I use async copy to increase performance..?
        //      if I use async copy there can be timing issue.
        //        example>
        //          TaskConsumer A : memcpyAsync tensor
        //          TaskConsumer B : update tensor
        //          A should be done before B!! but I'm not sure
        for (int i = 0; i < updateParams.size(); i++) {
            tensorSourceParam = (Data<float>*)updateParams[i].paramDataPtr;
            tensorTargetParam = (Data<float>*)value->tensorDataPtrs[i];
            checkCudaErrors(cudaMemcpy((void*)tensorSourceParam->device_grad(),
                                       tensorTargetParam->mutable_device_grad(), bufSize, 
                                       cudaMemcpyDeviceToDevice));
        }

        if (value->devID != Worker::gpuIdx) {
            // makes an updater task and inserts it into updater task queue
            int consumerIdx = Worker::getConsumerIdx(Worker::gpuIdx);
            SASSUME0(consumerIdx >= 0);
            Worker::addUpdateTensorTask(consumerIdx, networkID, dopID, layerID, planID,
                updateParams);
            return false;
        }
    }

    // (2) update param을 수행
    for (int i = 0; i < updateParams.size(); i++) {
        tensorTargetParam = (Data<float>*)value->tensorDataPtrs[i];
        tensorDataHis1 = (Data<float>*)value->tensorDataHis1Ptrs[i];
        tensorDataHis2 = (Data<float>*)value->tensorDataHis2Ptrs[i];

        Update<float>::updateParam(updateParams[i].context, tensorDataHis1, tensorDataHis2,
            tensorTargetParam);
    }

    // (3) 상황에 맞게 param의 data를 동기화한다.
    if (value->nodeID != SPARAM(NODE_ID)) {
        // cluster mode
        SASSERT(false, "Not implemented yet");
    }

    for (int i = 0; i < updateParams.size(); i++) {
        tensorSourceParam = (Data<float>*)updateParams[i].paramDataPtr;
        tensorTargetParam = (Data<float>*)value->tensorDataPtrs[i];
        checkCudaErrors(cudaMemcpy(tensorSourceParam->mutable_device_data(),
                                   (void*)tensorTargetParam->device_data(), bufSize, 
                                   cudaMemcpyDeviceToDevice));
    }

    unique_lock<mutex> accessLock(value->mutex);
    SASSUME0(value->access == false);
    value->access = true;
    accessLock.unlock();

    PhysicalPlan::markFinish(networkID, dopID, planID);

    return true;
}

// 각 레이어의 업데이트 함수는 이 함수를 호출해야 한다.
bool Updater::updateParams(vector<UpdateParam> updateParams) {
    string networkID = WorkContext::curNetworkID;
    int layerID = WorkContext::curLayerProp->layerID;
    int planID = LP_UPDATE_PLANID(layerID);
    int dopID = WorkContext::curDOPID;

    return Updater::updateParams(networkID, layerID, planID, dopID, updateParams, true);
}
