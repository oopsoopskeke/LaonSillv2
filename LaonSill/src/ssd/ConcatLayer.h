/*
 * ConcatLayer.h
 *
 *  Created on: Apr 26, 2017
 *      Author: jkim
 */

#ifndef CONCATLAYER_H_
#define CONCATLAYER_H_


#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

/*
 * @brief Takes at least two Datas and concatenates them along either the num
 *        or channel dimension, outputting the result.
 */
template <typename Dtype>
class ConcatLayer : public Layer<Dtype> {
public:
	ConcatLayer();
	virtual ~ConcatLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:

private:
	int count;
	int numConcat;
	int concatInputSize;
	int concatAxis;

	int tempCount;


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

#endif /* CONCATLAYER_H_ */
