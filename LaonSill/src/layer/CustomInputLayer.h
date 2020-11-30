/**
 * @file CustomInputLayer.h
 * @date 2017-06-29
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef CUSTOMINPUTLAYER_H
#define CUSTOMINPUTLAYER_H 

#include <vector>

#include "common.h"
#include "InputLayer.h"
#include "BaseLayer.h"
#include "Data.h"
#include "LayerFunc.h"

typedef void (*CBCustomInputForward) (int miniBatchIdx, int batchSize, void* args, std::vector<float*> &data);

template<typename Dtype>
class CustomInputLayer : public InputLayer<Dtype> {
public: 
    CustomInputLayer() {}
    virtual ~CustomInputLayer();

    void feedforward();
    using Layer<Dtype>::feedforward;
    void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();

    void reshape();
   
    void registerCBFunc(CBCustomInputForward forwardFunc, void* args);

private:
    CBCustomInputForward    forwardFunc;
    void*                   forwardFuncArgs;
    std::vector<float*>     dataArray;

public:
    /****************************************************************************
     * layer callback functions 
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
    static bool checkShape(std::vector<TensorShape> inputShape,
            std::vector<TensorShape> &outputShape);
    static uint64_t calcGPUSize(std::vector<TensorShape> inputShape);
};
#endif /* CUSTOMINPUTLAYER_H */
