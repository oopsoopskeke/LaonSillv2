/**
 * @file LayerFunc.h
 * @date 2017-05-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef LAYERFUNC_H
#define LAYERFUNC_H 

#include "common.h"

#include <vector>

typedef struct TensorShape_s {
    int N;
    int C;
    int H;
    int W;
} TensorShape;

int& tensorRefByIndex(TensorShape& tensorShape, const int index);
int tensorValByIndex(TensorShape& tensorShape, const int index);
size_t tensorCount(TensorShape& tensorShape, const int startAxis = 0, const int endAxis = -1);

typedef void* (*CBInitLayer) ();
typedef void (*CBDestroyLayer) (void* instancePtr);
typedef void (*CBSetInOutTensor) (void* instancePtr, void* tensorPtr, bool isInput, int index);
typedef bool (*CBAllocLayerTensors) (void* instancePtr);
typedef void (*CBForward) (void* instancePtr, int miniBatchIndex);
typedef void (*CBBackward) (void* instancePtr);
typedef void (*CBLearn) (void* instancePtr);
typedef bool (*CBCheckShape) (std::vector<TensorShape> inputShape,
        std::vector<TensorShape> &outputShape);
typedef uint64_t (*CBCalcGPUSize) (std::vector<TensorShape> inputShape);

typedef struct CBLayerFunc_s {
    CBInitLayer         initLayer;
    CBDestroyLayer      destroyLayer;
    CBSetInOutTensor    setInOutTensor;
    CBAllocLayerTensors allocLayerTensors;
    CBForward           forward;
    CBBackward          backward;
    CBLearn             learn;
    CBCheckShape        checkShape;
    CBCalcGPUSize       calcGPUSize;
} CBLayerFunc;

class LayerFunc {
public: 
    LayerFunc() {}
    virtual ~LayerFunc() {}

    static void init();
    static void destroy();
    static void registerLayerFunc(int layerType, CBInitLayer initLayer,
                                  CBDestroyLayer destroyLayer,
                                  CBSetInOutTensor setInOutTensor, 
                                  CBAllocLayerTensors allocLayerTensors, CBForward forward,
                                  CBBackward backward, CBLearn learn);

    static void registerLayerFunc2(int layerType, CBInitLayer initLayer,
                                  CBDestroyLayer destroyLayer,
                                  CBSetInOutTensor setInOutTensor, 
                                  CBAllocLayerTensors allocLayerTensors, CBForward forward,
                                  CBBackward backward, CBLearn learn,
                                  CBCheckShape checkShape, CBCalcGPUSize calcGPUSize);

    static void* initLayer(int layerType);
    static void destroyLayer(int layerType, void* instancePtr);
    static void setInOutTensor(int layerType, void* instancePtr, void *tensorPtr,
        bool isInput, int index);
    static bool allocLayerTensors(int layerType, void* instancePtr);
    static void runForward(int layerType, void* instancePtr, int miniBatchIdx);
    static void runBackward(int layerType, void* instancePtr);
    static void learn(int layerType, void* instancePtr);
    static bool checkShape(int layerType, std::vector<TensorShape> inputShape,
            std::vector<TensorShape> &outputShape);
    static uint64_t calcGPUSize(int layerType, std::vector<TensorShape> inputShape);

private:
    static CBLayerFunc      *layerFuncs;
};

#endif /* LAYERFUNC_H */
