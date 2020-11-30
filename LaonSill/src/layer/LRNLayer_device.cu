/*
 * LRNLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "LRNLayer.h"
#include "PropMgmt.h"
#include "Util.h"
#include "MemoryMgmt.h"

using namespace std;


template <typename Dtype>
LRNLayer<Dtype>::LRNLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::LRN;
	const lrn_dim& lrnDim = SLPROP(LRN, lrnDim);

	checkCUDNN(cudnnCreateTensorDescriptor(&this->inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputTensorDesc));
	checkCUDNN(cudnnCreateLRNDescriptor(&this->lrnDesc));
	checkCUDNN(cudnnSetLRNDescriptor(this->lrnDesc,
			lrnDim.local_size, lrnDim.alpha, lrnDim.beta, lrnDim.k));
}


template <typename Dtype>
LRNLayer<Dtype>::~LRNLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(this->inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputTensorDesc));
	checkCUDNN(cudnnDestroyLRNDescriptor(this->lrnDesc));
}


// (1 + alpha/n * sigma(i)(xi^2))^beta
template <typename Dtype>
void LRNLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();
	checkCUDNN(cudnnLRNCrossChannelForward(Cuda::cudnnHandle,
			this->lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&Cuda::alpha, this->inputTensorDesc, d_inputData,
			&Cuda::beta, this->outputTensorDesc, d_outputData));

	const string& name = SLPROP_BASE(name);
	this->_outputData[0]->print_data(name + string("/d_output:"));
}

template <typename Dtype>
void LRNLayer<Dtype>::backpropagation() {
	const vector<bool>& propDown = SLPROP_BASE(propDown);
	if (propDown[0]) {
		const Dtype* d_outputData = this->_outputData[0]->device_data();
		const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
		const Dtype* d_inputData = this->_inputData[0]->device_data();
		Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();
		checkCUDNN(cudnnLRNCrossChannelBackward(Cuda::cudnnHandle,
				this->lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
				&Cuda::alpha, this->outputTensorDesc,
                d_outputData, this->outputTensorDesc, d_outputGrad,
				this->inputTensorDesc, d_inputData,
				&Cuda::beta, this->inputTensorDesc, d_inputGrad));
	}
}

template LRNLayer<float>::LRNLayer();
template LRNLayer<float>::~LRNLayer();
template void LRNLayer<float>::feedforward();
template void LRNLayer<float>::backpropagation();



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* LRNLayer<Dtype>::initLayer() {
	LRNLayer* layer = NULL;
	SNEW(layer, LRNLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void LRNLayer<Dtype>::destroyLayer(void* instancePtr) {
    LRNLayer<Dtype>* layer = (LRNLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void LRNLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    LRNLayer<Dtype>* layer = (LRNLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool LRNLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    LRNLayer<Dtype>* layer = (LRNLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void LRNLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	LRNLayer<Dtype>* layer = (LRNLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void LRNLayer<Dtype>::backwardTensor(void* instancePtr) {
	LRNLayer<Dtype>* layer = (LRNLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void LRNLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool LRNLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t LRNLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template void* LRNLayer<float>::initLayer();
template void LRNLayer<float>::destroyLayer(void* instancePtr);
template void LRNLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool LRNLayer<float>::allocLayerTensors(void* instancePtr);
template void LRNLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void LRNLayer<float>::backwardTensor(void* instancePtr);
template void LRNLayer<float>::learnTensor(void* instancePtr);
template bool LRNLayer<float>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape);
template uint64_t LRNLayer<float>::calcGPUSize(vector<TensorShape> inputShape);
