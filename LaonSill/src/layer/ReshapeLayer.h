/*
 * ReshapeLayer.h
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#ifndef RESHAPELAYER_H_
#define RESHAPELAYER_H_


#include <vector>

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"


template <typename Dtype>
class ReshapeLayer : public Layer<Dtype> {
public:
	ReshapeLayer();
	virtual ~ReshapeLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	std::vector<uint32_t> copyAxes;
	int inferredAxis;
	uint32_t constantCount;


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

#endif /* RESHAPELAYER_H_ */
