/*
 * RoIPoolingLayer.h
 *
 *  Created on: Dec 1, 2016
 *      Author: jkim
 */

#ifndef ROIPOOLINGLAYER_H_
#define ROIPOOLINGLAYER_H_

#include <vector>

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class RoIPoolingLayer : public Layer<Dtype> {
public:
	RoIPoolingLayer();
	virtual ~RoIPoolingLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();


private:
	uint32_t channels;
	uint32_t height;
	uint32_t width;
	Data<int> maxIdx;


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

#endif /* ROIPOOLINGLAYER_H_ */
