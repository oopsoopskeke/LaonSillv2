/*
 * InterpolationLayer.h
 *
 *  Created on: Aug 7, 2017
 *      Author: jkim
 */

#ifndef INTERPOLATIONLAYER_H_
#define INTERPOLATIONLAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

/**
 * Change the spatial resolution by bi-linear interpolation
 * The target size is specified in terms of pixels.
 * The start and end pixels of the input are mapped to the start
 * and end pixels of the output.
 */
template <typename Dtype>
class InterpolationLayer : public Layer<Dtype> {
public:
	InterpolationLayer();
	virtual ~InterpolationLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	int batches;
	int channels;
	int heightIn;
	int widthIn;
	int heightOut;
	int widthOut;
	int padBeg;
	int padEnd;
	int heightInEff;
	int widthInEff;


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

#endif /* INTERPOLATIONLAYER_H_ */
