/**
 * @file Donator.cpp
 * @date 2017-02-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Donator.h"
#include "BaseLayer.h"
#include "ConvLayer.h"
#include "BatchNormLayer.h"
#include "FullyConnectedLayer.h"
#include "SysLog.h"
#include "ColdLog.h"
#include "MemoryMgmt.h"
#include "PropMgmt.h"

using namespace std;

template<typename Dtype>
map<uint32_t, DonatorData>  Donator<Dtype>::donatorMap;
template<typename Dtype>
mutex                       Donator<Dtype>::donatorMutex;

template<typename Dtype>
void Donator<Dtype>::donate(uint32_t donatorID, void* layerPtr) {
    DonatorData newData;
    newData.donatorID = donatorID;
    newData.donatorRefCount = 1;
    newData.refCount = 0;
    newData.layerPtr = layerPtr;
    newData.cleanUp = false;

    unique_lock<mutex> donatorLock(Donator::donatorMutex);

    if (!SNPROP(useCompositeModel)) {
        SASSERT((Donator::donatorMap.count(donatorID) == 0), "already layer(ID=%u) donated.",
            donatorID);
    }

    if (Donator::donatorMap.count(donatorID) != 0) {
        Donator::donatorMap[donatorID].donatorRefCount =
            Donator::donatorMap[donatorID].donatorRefCount + 1;
    } else {
        Donator::donatorMap[donatorID] = newData;
    }
}

template<typename Dtype>
void Donator<Dtype>::receive(uint32_t donatorID, void* layerPtr) {
    unique_lock<mutex> donatorLock(Donator::donatorMutex);
    SASSERT((Donator::donatorMap.count(donatorID) == 1), "there is no donator(ID=%u).",
        donatorID);
    DonatorData data = Donator::donatorMap[donatorID];
    data.refCount += 1;
    donatorLock.unlock();

    // FIXME: dangerous casting.. should be fixed in the futre
    Layer<Dtype>* donatorLayer = (Layer<Dtype>*)data.layerPtr;
    Layer<Dtype>* receiverLayer = (Layer<Dtype>*)layerPtr;

    LearnableLayer<Dtype>* donatorLearnableLayer = (LearnableLayer<Dtype>*)data.layerPtr;
    LearnableLayer<Dtype>* receiverLearnableLayer = (LearnableLayer<Dtype>*)layerPtr;
   
    SASSERT(donatorLayer->type == receiverLayer->type,
        "both donator and receiver should have same layer type."
        "donator layer type=%d, receiver layer type=%d",
        (int)donatorLayer->type, (int)receiverLayer->type);

    donatorLearnableLayer->donateParam(receiverLearnableLayer);
}

template<typename Dtype>
void Donator<Dtype>::releaseDonator(uint32_t donatorID) {
    unique_lock<mutex> donatorLock(Donator::donatorMutex);

    SASSERT((Donator::donatorMap.count(donatorID) == 1), "there is no donator(ID=%u).",
        donatorID);

    Donator::donatorMap[donatorID].donatorRefCount = 
        Donator::donatorMap[donatorID].donatorRefCount - 1;

    DonatorData data = Donator::donatorMap[donatorID];

    if (data.donatorRefCount > 0)
        return;

    if (data.refCount > 0) {
        data.cleanUp = true;
    } else {
        Layer<Dtype>* layer = (Layer<Dtype>*)data.layerPtr;
        SDELETE(layer);
        Donator::donatorMap.erase(donatorID);
    }
}

template<typename Dtype>
void Donator<Dtype>::releaseReceiver(uint32_t donatorID) {
    unique_lock<mutex> donatorLock(Donator::donatorMutex);
    SASSERT((Donator::donatorMap.count(donatorID) == 1), "there is no donator(ID=%u).",
        donatorID);
    DonatorData data = Donator::donatorMap[donatorID];
   
    data.refCount -= 1;
    if ((data.refCount == 0) && data.cleanUp) {
        Layer<Dtype>* layer = (Layer<Dtype>*)data.layerPtr;
        SDELETE(layer);
        Donator::donatorMap.erase(donatorID);
    }
}

template class Donator<float>;
