/*
 * FlattenLayer.h
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#ifndef FLATTENLAYER_H_
#define FLATTENLAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

/*
 * @breif Reshapes the input data into flat vectors
 */
template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
public:
	FlattenLayer();
	virtual ~FlattenLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	//void initialize();

private:
	//uint32_t axis;
	//uint32_t endAxis;





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

#endif /* FLATTENLAYER_H_ */
