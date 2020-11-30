/**
 * @file LayerFunc.cpp
 * @date 2017-05-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <iostream>

#include "LayerFunc.h"
#include "BaseLayer.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "LayerPropList.h"
#include "MemoryMgmt.h"

using namespace std;

CBLayerFunc* LayerFunc::layerFuncs;

int& tensorRefByIndex(TensorShape& tensorShape, const int index) {
    SASSERT0(index >= 0 && index < 4);

    switch(index) {
        case 0: return tensorShape.N;
        case 1: return tensorShape.C;
        case 2: return tensorShape.H;
        case 3: return tensorShape.W;
    }
    return tensorShape.N;
}

int tensorValByIndex(TensorShape& tensorShape, const int index) {
    SASSERT0(index >= 0 && index < 4);

    switch(index) {
        case 0: return tensorShape.N;
        case 1: return tensorShape.C;
        case 2: return tensorShape.H;
        case 3: return tensorShape.W;
    }
    return tensorShape.N;
}

size_t tensorCount(TensorShape& tensorShape, const int startAxis, const int endAxis) {
    size_t count = 1;
    int _endAxis = endAxis == -1 ? Data<float>::SHAPE_SIZE : endAxis; 

    for (int i = startAxis; i < _endAxis; i++) {
        count *= tensorValByIndex(tensorShape, i);
    }
    return count;
}

void LayerFunc::init() {
    int layerTypeSize = Layer<float>::LayerTypeMax;

    LayerFunc::layerFuncs = NULL;
    int allocSize = sizeof(CBLayerFunc) * layerTypeSize;
    SMALLOC_ONCE(LayerFunc::layerFuncs, CBLayerFunc, allocSize);
    SASSERT0(LayerFunc::layerFuncs != NULL); 
}

void LayerFunc::destroy() {
    SASSERT0(LayerFunc::layerFuncs != NULL);
    SFREE(LayerFunc::layerFuncs);
}

void LayerFunc::registerLayerFunc(int layerType, CBInitLayer initLayer,
    CBDestroyLayer destroyLayer, CBSetInOutTensor setInOutTensor,
    CBAllocLayerTensors allocLayerTensors, CBForward forward, CBBackward backward,
    CBLearn learn) {    
    SASSERT0(layerType < Layer<float>::LayerTypeMax);
    LayerFunc::layerFuncs[layerType].initLayer = initLayer;
    LayerFunc::layerFuncs[layerType].destroyLayer = destroyLayer;
    LayerFunc::layerFuncs[layerType].setInOutTensor = setInOutTensor;
    LayerFunc::layerFuncs[layerType].allocLayerTensors = allocLayerTensors;
    LayerFunc::layerFuncs[layerType].forward = forward;
    LayerFunc::layerFuncs[layerType].backward = backward;
    LayerFunc::layerFuncs[layerType].learn = learn;
}

void LayerFunc::registerLayerFunc2(int layerType, CBInitLayer initLayer,
    CBDestroyLayer destroyLayer, CBSetInOutTensor setInOutTensor,
    CBAllocLayerTensors allocLayerTensors, CBForward forward, CBBackward backward,
    CBLearn learn, CBCheckShape checkShape, CBCalcGPUSize calcGPUSize) {    
    SASSERT0(layerType < Layer<float>::LayerTypeMax);
    LayerFunc::layerFuncs[layerType].initLayer = initLayer;
    LayerFunc::layerFuncs[layerType].destroyLayer = destroyLayer;
    LayerFunc::layerFuncs[layerType].setInOutTensor = setInOutTensor;
    LayerFunc::layerFuncs[layerType].allocLayerTensors = allocLayerTensors;
    LayerFunc::layerFuncs[layerType].forward = forward;
    LayerFunc::layerFuncs[layerType].backward = backward;
    LayerFunc::layerFuncs[layerType].learn = learn;
    LayerFunc::layerFuncs[layerType].checkShape = checkShape;
    LayerFunc::layerFuncs[layerType].calcGPUSize = calcGPUSize;
}

void* LayerFunc::initLayer(int layerType) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "init layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    return LayerFunc::layerFuncs[layerType].initLayer();
}

void LayerFunc::destroyLayer(int layerType, void* instancePtr) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "destroy layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    LayerFunc::layerFuncs[layerType].destroyLayer(instancePtr);
}

void LayerFunc::setInOutTensor(int layerType, void* instancePtr, void *tensorPtr,
    bool isInput, int index) {

    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "set in/out layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    LayerFunc::layerFuncs[layerType].setInOutTensor(instancePtr, tensorPtr, isInput, index);
}

bool LayerFunc::allocLayerTensors(int layerType, void* instancePtr) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "alloc layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    return LayerFunc::layerFuncs[layerType].allocLayerTensors(instancePtr);
}

void LayerFunc::runForward(int layerType, void* instancePtr, int miniBatchIdx) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "forward layer(layer=%s). miniBatchIdx=%d",
            LayerPropList::getLayerName(layerType).c_str(), miniBatchIdx);
    LayerFunc::layerFuncs[layerType].forward(instancePtr, miniBatchIdx);
}

void LayerFunc::runBackward(int layerType, void* instancePtr) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "backward layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    LayerFunc::layerFuncs[layerType].backward(instancePtr);
}

void LayerFunc::learn(int layerType, void* instancePtr) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "learn layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    LayerFunc::layerFuncs[layerType].learn(instancePtr);
}

bool LayerFunc::checkShape(int layerType, vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_BLOCK(SPARAM(PRINT_LAYERFUNC_LOG),
            cout << "check shape(layer=" << LayerPropList::getLayerName(layerType) << ")";
            for (int i = 0; i < inputShape.size(); i++)
                cout << " {" << inputShape[i].N << "," << inputShape[i].C << "," <<
                    inputShape[i].H << "," << inputShape[i].W << "}";
            cout << endl;);
    return LayerFunc::layerFuncs[layerType].checkShape(inputShape, outputShape);
}

uint64_t LayerFunc::calcGPUSize(int layerType, std::vector<TensorShape> inputShape) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_BLOCK(SPARAM(PRINT_LAYERFUNC_LOG),
            cout << "calc GPU size(layer=" << LayerPropList::getLayerName(layerType) << ")";
            for (int i = 0; i < inputShape.size(); i++)
                cout << " {" << inputShape[i].N << "," << inputShape[i].C << "," <<
                    inputShape[i].H << "," << inputShape[i].W << "}";
            cout << endl;);
    return LayerFunc::layerFuncs[layerType].calcGPUSize(inputShape);
}
