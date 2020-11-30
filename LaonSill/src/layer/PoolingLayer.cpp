/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#include "PoolingLayer.h"
#include "PropMgmt.h"
#include "StdOutLog.h"

using namespace std;

#define POOLINGLAYER_LOG 0

template <typename Dtype>
PoolingLayer<Dtype>::PoolingLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Pooling;

#if POOLINGLAYER_LOG
	STDOUT_LOG("Layer: %s", SLPROP_BASE(name).c_str());
	SLPROP(Pooling, poolDim).print();
#endif
	this->globalPooling = SLPROP(Pooling, globalPooling);
	this->poolDim = SLPROP(Pooling, poolDim);
	this->poolingType = SLPROP(Pooling, poolingType);
	this->pooling_fn = NULL;

	if (this->globalPooling) {
		SASSERT(this->poolDim.pad == 0 && this->poolDim.stride == 1,
				"With globalPooling true, only pad = 0 and stride = 1 is supported.");
	}


	//this->pooling_fn = PoolingFactory<Dtype>::create(poolingType, poolDim);

	checkCUDNN(cudnnCreateTensorDescriptor(&this->inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputTensorDesc));
}


template <typename Dtype>
PoolingLayer<Dtype>::~PoolingLayer() {
	PoolingFactory<Dtype>::destroy(this->pooling_fn);
	checkCUDNN(cudnnDestroyTensorDescriptor(this->inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputTensorDesc));
}

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* PoolingLayer<Dtype>::initLayer() {
	PoolingLayer* layer = NULL;
	SNEW(layer, PoolingLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void PoolingLayer<Dtype>::destroyLayer(void* instancePtr) {
    PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void PoolingLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool PoolingLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void PoolingLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void PoolingLayer<Dtype>::backwardTensor(void* instancePtr) {
	PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void PoolingLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool PoolingLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];

    outputShape1.N = inputShape[0].N;
    outputShape1.C = inputShape[0].C;

    pool_dim poolDim = SLPROP(Pooling, poolDim);

	if (SLPROP(Pooling, globalPooling)) {
        outputShape1.H = 1;
        outputShape1.W = 1;
    } else {
        outputShape1.H = (inputShape[0].H + 2 * poolDim.pad - poolDim.rows) / 
            poolDim.stride + 1;
        outputShape1.W = (inputShape[0].W + 2 * poolDim.pad - poolDim.cols) / 
            poolDim.stride + 1;
    }

    if (outputShape1.H < 1 || outputShape1.W < 1)
        return false;

    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t PoolingLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class PoolingLayer<float>;
