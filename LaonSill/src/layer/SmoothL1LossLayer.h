/*
 * SmoothL1LossLayer.h
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#ifndef SMOOTHL1LOSSLAYER_H_
#define SMOOTHL1LOSSLAYER_H_


#include "common.h"
#include "LossLayer.h"
#include "LayerPropList.h"
#include "LayerFunc.h"

template <typename Dtype>
class SmoothL1LossLayer : public LossLayer<Dtype> {
public:
	SmoothL1LossLayer();
	SmoothL1LossLayer(_SmoothL1LossPropLayer* prop);
	virtual ~SmoothL1LossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

	virtual inline int exactNumBottomBlobs() const { return -1; }
	virtual inline int minBottomBlobs() const { return 2; }
	virtual inline int maxBottomBlobs() const { return 4; }

private:
	Data<Dtype> diff;
	Data<Dtype> errors;
	Data<Dtype> ones;
	bool hasWeights;

	_SmoothL1LossPropLayer* prop;
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

public:
    static int INNER_ID;

};

#endif /* SMOOTHL1LOSSLAYER_H_ */
