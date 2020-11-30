/*
 * ElementWiseLayer.h
 *
 *  Created on: Aug 4, 2017
 *      Author: jkim
 */

#ifndef ELEMENTWISELAYER_H_
#define ELEMENTWISELAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "EnumDef.h"
#include "LayerFunc.h"

/**
 * Compute elementwise operations, such as product and sum,
 * along multiple input data.
 */
template <typename Dtype>
class ElementWiseLayer : public Layer<Dtype> {
public:
	ElementWiseLayer();
	virtual ~ElementWiseLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	ElementWiseOp op;
	std::vector<Dtype> coeffs;
	bool stableProdGrad;
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

#endif /* ELEMENTWISELAYER_H_ */
