/*
 * DummyLossLayer.cpp
 *
 *  Created on: Sep 7, 2017
 *      Author: jkim
 */

#include "DummyLossLayer.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
DummyLossLayer<Dtype>::DummyLossLayer()
: LossLayer<Dtype>() {
	this->type = Layer<Dtype>::DummyLoss;
}

template <typename Dtype>
DummyLossLayer<Dtype>::~DummyLossLayer() {}

template <typename Dtype>
void DummyLossLayer<Dtype>::reshape() {}

template <typename Dtype>
void DummyLossLayer<Dtype>::feedforward() {}

template <typename Dtype>
void DummyLossLayer<Dtype>::backpropagation() {}

template <typename Dtype>
Dtype DummyLossLayer<Dtype>::cost() {
	return Dtype(1.0);
}



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* DummyLossLayer<Dtype>::initLayer() {
	DummyLossLayer* layer = NULL;
	SNEW(layer, DummyLossLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void DummyLossLayer<Dtype>::destroyLayer(void* instancePtr) {
    DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void DummyLossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DummyLossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void DummyLossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void DummyLossLayer<Dtype>::backwardTensor(void* instancePtr) {
	DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void DummyLossLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool DummyLossLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    TensorShape outputShape1;
    outputShape1.N = inputShape[0].N;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 1;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t DummyLossLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class DummyLossLayer<float>;


