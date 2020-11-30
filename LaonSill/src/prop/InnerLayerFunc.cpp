/**
 * @file InnerLayerFunc.cpp
 * @date 2017-05-26
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string>

#include "InnerLayerFunc.h"
#include "PhysicalPlan.h"
#include "LayerFunc.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "WorkContext.h"
#include "MemoryMgmt.h"

using namespace std;

// FIXME: 소스 정리 필요... 나중에 하자.. 지금 너무 귀찮다.

void InnerLayerFunc::initLayer(int innerLayerIdx) {
    // set inner layer context
    SASSUME0(innerLayerIdx < SLPROP_BASE(innerLayerIDs).size());
    int oldLayerID = WorkContext::curLayerProp->layerID;
    string networkID = WorkContext::curLayerProp->networkID;
    int newLayerID = SLPROP_BASE(innerLayerIDs)[innerLayerIdx];
    WorkContext::updateLayer(networkID, newLayerID);
    int layerType = WorkContext::curLayerProp->layerType;

    void* instancePtr = LayerFunc::initLayer(layerType);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    SASSUME0(pp->instanceMap.find(newLayerID) == pp->instanceMap.end());
    pp->instanceMap[newLayerID] = instancePtr;

    // restore context
    WorkContext::updateLayer(networkID, oldLayerID);
}

void InnerLayerFunc::destroyLayer(int innerLayerIdx) {
    // set inner layer context
    SASSUME0(innerLayerIdx < SLPROP_BASE(innerLayerIDs).size());
    int oldLayerID = WorkContext::curLayerProp->layerID;
    string networkID = WorkContext::curLayerProp->networkID;
    int newLayerID = SLPROP_BASE(innerLayerIDs)[innerLayerIdx];
    WorkContext::updateLayer(networkID, newLayerID);
    int layerType = WorkContext::curLayerProp->layerType;

    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    SASSUME0(pp->instanceMap.find(newLayerID) != pp->instanceMap.end());
    LayerFunc::destroyLayer(layerType, pp->instanceMap[newLayerID]);

    // restore context
    WorkContext::updateLayer(networkID, oldLayerID);
}

void InnerLayerFunc::setInOutTensor(int innerLayerIdx, void *tensorPtr, bool isInput,
    int index) {
    // set inner layer context
    SASSUME0(innerLayerIdx < SLPROP_BASE(innerLayerIDs).size());
    int oldLayerID = WorkContext::curLayerProp->layerID;
    string networkID = WorkContext::curLayerProp->networkID;
    int newLayerID = SLPROP_BASE(innerLayerIDs)[innerLayerIdx];
    WorkContext::updateLayer(networkID, newLayerID);
    int layerType = WorkContext::curLayerProp->layerType;

    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    SASSUME0(pp->instanceMap.find(newLayerID) != pp->instanceMap.end());
    void* instancePtr = pp->instanceMap[newLayerID];

    if (tensorPtr == NULL) {
        string tensorName;
        if (isInput) {
            SASSUME0(index < SLPROP_BASE(input).size());
            tensorName = SLPROP_BASE(input)[index];
        } else {
            SASSUME0(index < SLPROP_BASE(output).size());
            tensorName = SLPROP_BASE(output)[index];
        }
        SNEW(tensorPtr, Data<float>, tensorName);
        SASSUME0(tensorPtr != NULL);
    }

    LayerFunc::setInOutTensor(layerType, instancePtr, tensorPtr, isInput, index);

    // restore context
    WorkContext::updateLayer(networkID, oldLayerID);
}

bool InnerLayerFunc::allocLayerTensors(int innerLayerIdx) {
    // set inner layer context
    SASSUME0(innerLayerIdx < SLPROP_BASE(innerLayerIDs).size());
    int oldLayerID = WorkContext::curLayerProp->layerID;
    string networkID = WorkContext::curLayerProp->networkID;
    int newLayerID = SLPROP_BASE(innerLayerIDs)[innerLayerIdx];
    WorkContext::updateLayer(networkID, newLayerID);
    int layerType = WorkContext::curLayerProp->layerType;

    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    SASSUME0(pp->instanceMap.find(newLayerID) != pp->instanceMap.end());
    void* instancePtr = pp->instanceMap[newLayerID];

    bool ret = LayerFunc::allocLayerTensors(layerType, instancePtr);

    // restore context
    WorkContext::updateLayer(networkID, oldLayerID);

    return ret;
}


void InnerLayerFunc::runForward(int innerLayerIdx, int miniBatchIdx) {
    // set inner layer context
    SASSUME0(innerLayerIdx < SLPROP_BASE(innerLayerIDs).size());
    int oldLayerID = WorkContext::curLayerProp->layerID;
    string networkID = WorkContext::curLayerProp->networkID;
    int newLayerID = SLPROP_BASE(innerLayerIDs)[innerLayerIdx];
    WorkContext::updateLayer(networkID, newLayerID);
    int layerType = WorkContext::curLayerProp->layerType;

    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    SASSUME0(pp->instanceMap.find(newLayerID) != pp->instanceMap.end());
    void* instancePtr = pp->instanceMap[newLayerID];

    LayerFunc::runForward(layerType, instancePtr, miniBatchIdx);

    // restore context
    WorkContext::updateLayer(networkID, oldLayerID);
}

void InnerLayerFunc::runBackward(int innerLayerIdx) {
    // set inner layer context
    SASSUME0(innerLayerIdx < SLPROP_BASE(innerLayerIDs).size());
    int oldLayerID = WorkContext::curLayerProp->layerID;
    string networkID = WorkContext::curLayerProp->networkID;
    int newLayerID = SLPROP_BASE(innerLayerIDs)[innerLayerIdx];
    WorkContext::updateLayer(networkID, newLayerID);
    int layerType = WorkContext::curLayerProp->layerType;

    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    SASSUME0(pp->instanceMap.find(newLayerID) != pp->instanceMap.end());
    void* instancePtr = pp->instanceMap[newLayerID];

    LayerFunc::runBackward(layerType, instancePtr);

    // restore context
    WorkContext::updateLayer(networkID, oldLayerID);
}

void InnerLayerFunc::learn(int innerLayerIdx) {
    // set inner layer context
    SASSUME0(innerLayerIdx < SLPROP_BASE(innerLayerIDs).size());
    int oldLayerID = WorkContext::curLayerProp->layerID;
    string networkID = WorkContext::curLayerProp->networkID;
    int newLayerID = SLPROP_BASE(innerLayerIDs)[innerLayerIdx];
    WorkContext::updateLayer(networkID, newLayerID);
    int layerType = WorkContext::curLayerProp->layerType;

    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    SASSUME0(pp->instanceMap.find(newLayerID) != pp->instanceMap.end());
    void* instancePtr = pp->instanceMap[newLayerID];

    LayerFunc::learn(layerType, instancePtr);

    // restore context
    WorkContext::updateLayer(networkID, oldLayerID);
}
