/**
 * @file CrossEntropyWithLossLayer.h
 * @date 2017-02-06
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef SIGMOIDWITHLOSSLAYER_H
#define SIGMOIDWITHLOSSLAYER_H 

#include "common.h"
#include "LossLayer.h"
#include "LayerConfig.h"
#include "LayerFunc.h"

template <typename Dtype>
class CrossEntropyWithLossLayer : public LossLayer<Dtype> {
public: 

    CrossEntropyWithLossLayer();
    virtual ~CrossEntropyWithLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();
    void setTargetValue(Dtype value);

private:
    int     depth;

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

#endif /* SIGMOIDWITHLOSSLAYER_H */
