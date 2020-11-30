/*
 * PriorBoxLayer.h
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#ifndef PRIORBOXLAYER_H_
#define PRIORBOXLAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

/*
 * @brief Generate the prior boxes of designated sizes and aspect ratios across
 *        all dimensions
 */
template <typename Dtype>
class PriorBoxLayer : public Layer<Dtype> {
public:
	PriorBoxLayer();
	virtual ~PriorBoxLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:



private:
	int numPriors;





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

#endif /* PRIORBOXLAYER_H_ */
