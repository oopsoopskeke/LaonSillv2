/*
 * SilenceLayer.h
 *
 *  Created on: Aug 7, 2017
 *      Author: jkim
 */

#ifndef SILENCELAYER_H_
#define SILENCELAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

/**
 * Ignores input data while producing no output data.
 * (This is useful to suppress outputs during testing.)
 */
template <typename Dtype>
class SilenceLayer : public Layer<Dtype> {
public:
	SilenceLayer();
	virtual ~SilenceLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();


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

#endif /* SILENCELAYER_H_ */
