/*
 * DummyLossLayer.h
 *
 *  Created on: Sep 7, 2017
 *      Author: jkim
 */

#ifndef DUMMYLOSSLAYER_H_
#define DUMMYLOSSLAYER_H_

#include "common.h"
#include "LossLayer.h"
#include "LayerPropList.h"
#include "LayerFunc.h"

template <typename Dtype>
class DummyLossLayer : public LossLayer<Dtype> {
public:
	DummyLossLayer();
	virtual ~DummyLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

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

#endif /* DUMMYLOSSLAYER_H_ */
