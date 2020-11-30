/*
 * SilenceLayer.cpp
 *
 *  Created on: Aug 7, 2017
 *      Author: jkim
 */

#include "SilenceLayer.h"
#include "MathFunctions.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

template <typename Dtype>
SilenceLayer<Dtype>::SilenceLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Silence;
}

template <typename Dtype>
SilenceLayer<Dtype>::~SilenceLayer() {}


template <typename Dtype>
void SilenceLayer<Dtype>::reshape() {
}

template <typename Dtype>
void SilenceLayer<Dtype>::feedforward() {
	reshape();
	// Do nothing
}

template <typename Dtype>
void SilenceLayer<Dtype>::backpropagation() {
	for (int i = 0; i < this->_inputData.size(); i++) {
		if (SLPROP_BASE(propDown)[i]) {
			soooa_gpu_set(this->_inputData[i]->getCount(), Dtype(0),
					this->_inputData[i]->mutable_device_grad());
		}
	}
}


/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* SilenceLayer<Dtype>::initLayer() {
	SilenceLayer* layer = NULL;
	SNEW(layer, SilenceLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void SilenceLayer<Dtype>::destroyLayer(void* instancePtr) {
    SilenceLayer<Dtype>* layer = (SilenceLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void SilenceLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	if (isInput) {
		SASSERT0(index < 1);
	} else {
		SASSERT0(index < 1);
	}

    SilenceLayer<Dtype>* layer = (SilenceLayer<Dtype>*)instancePtr;
    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool SilenceLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    SilenceLayer<Dtype>* layer = (SilenceLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void SilenceLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	SilenceLayer<Dtype>* layer = (SilenceLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void SilenceLayer<Dtype>::backwardTensor(void* instancePtr) {
	SilenceLayer<Dtype>* layer = (SilenceLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void SilenceLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool SilenceLayer<Dtype>::checkShape(std::vector<TensorShape> inputShape,
        std::vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t SilenceLayer<Dtype>::calcGPUSize(std::vector<TensorShape> inputShape) {
    return 0UL;
}

template class SilenceLayer<float>;
