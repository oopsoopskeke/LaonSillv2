/*
 * ReluLayer.cpp
 *
 *  Created on: Jan 25, 2017
 *      Author: jkim
 */

#include "ReluLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "MemoryMgmt.h"

using namespace std;

#define RELULAYER_LOG 0


template<typename Dtype>
ReluLayer<Dtype>::ReluLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Relu;

	checkCUDNN(cudnnCreateTensorDescriptor(&this->tensorDesc));
	checkCUDNN(cudnnCreateActivationDescriptor(&this->activationDesc));
	checkCUDNN(cudnnSetActivationDescriptor(this->activationDesc, CUDNN_ACTIVATION_RELU,
			CUDNN_PROPAGATE_NAN, 0.0));
}

template <typename Dtype>
ReluLayer<Dtype>::~ReluLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(this->tensorDesc));
	checkCUDNN(cudnnDestroyActivationDescriptor(this->activationDesc));
}

template <typename Dtype>
void ReluLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputShape;

	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->tensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	this->_outputData[0]->reshape(inputShape);

	STDOUT_COND_LOG(RELULAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);
	STDOUT_COND_LOG(RELULAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);
}

template <typename Dtype>
void ReluLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

	const bool useLeaky = SLPROP(Relu, useLeaky);
    if (useLeaky) {
        applyLeakyForward();
    } else {
	    checkCUDNN(cudnnActivationForward(Cuda::cudnnHandle, this->activationDesc,
					&Cuda::alpha, this->tensorDesc, d_inputData,
					&Cuda::beta, this->tensorDesc, d_outputData));
    }
}

template <typename Dtype>
void ReluLayer<Dtype>::backpropagation() {
	const Dtype* d_outputData = this->_outputData[0]->device_data();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();

	const bool useLeaky = SLPROP(Relu, useLeaky);
    if (useLeaky) {
        applyLeakyBackward();
    } else {
	    checkCUDNN(cudnnActivationBackward(Cuda::cudnnHandle, this->activationDesc,
					&Cuda::alpha, this->tensorDesc, d_outputData, this->tensorDesc,
					d_outputGrad, this->tensorDesc, d_inputData,
					&Cuda::beta, this->tensorDesc, d_inputGrad));
    }
}


/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* ReluLayer<Dtype>::initLayer() {
	ReluLayer* layer = NULL;
	SNEW(layer, ReluLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ReluLayer<Dtype>::destroyLayer(void* instancePtr) {
    ReluLayer<Dtype>* layer = (ReluLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ReluLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    ReluLayer<Dtype>* layer = (ReluLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ReluLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ReluLayer<Dtype>* layer = (ReluLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ReluLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    ReluLayer<Dtype>* layer = (ReluLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void ReluLayer<Dtype>::backwardTensor(void* instancePtr) {
    ReluLayer<Dtype>* layer = (ReluLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void ReluLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool ReluLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t ReluLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class ReluLayer<float>;
