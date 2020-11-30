/*
 * DepthConcatLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "DepthConcatLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

//#define DEPTHCONCAT_LOG

template <typename Dtype>
DepthConcatLayer<Dtype>::DepthConcatLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::DepthConcat;
}


template <typename Dtype>
DepthConcatLayer<Dtype>::~DepthConcatLayer() {

}

template <typename Dtype>
void DepthConcatLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	// 입력 데이터의 shape가 변경된 것이 있는 지 확인
	bool inputShapeReshaped = false;
	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (Layer<Dtype>::_isInputShapeChanged(i)) {
			inputShapeReshaped = true;
			this->_inputShape[i] = this->_inputData[i]->getShape();
		}
	}

	if (!inputShapeReshaped) {
		return;
	}

	uint32_t batches 	= this->_inputShape[0][0];
	uint32_t channels 	= 0;
	uint32_t rows 		= this->_inputShape[0][2];
	uint32_t cols 		= this->_inputShape[0][3];

	for (uint32_t i = 0; i < this->_inputData.size(); i++) {
		channels += this->_inputData[i]->getShape()[1];
	}

	this->_outputData[0]->reshape({batches, channels, rows, cols});
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::feedforward() {
	reshape();

	uint32_t batchOffset = 0;
	for (uint32_t i = 0; i < this->_inputData.size(); i++) {
		batchOffset += this->_inputData[i]->getCountByAxis(1);
	}

	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();
	const uint32_t batchSize = this->_inputData[0]->getShape()[0];
	uint32_t inBatchOffset = 0;
	for (uint32_t i = 0; i < this->_inputData.size(); i++) {
		const Dtype* d_inputData = this->_inputData[i]->device_data();
		const uint32_t inputCountByChannel = this->_inputData[i]->getCountByAxis(1);
		if (i > 0) {
			inBatchOffset += this->_inputData[i-1]->getCountByAxis(1);
		}
		for (uint32_t j = 0; j < batchSize; j++) {
			checkCudaErrors(cudaMemcpyAsync(
					d_outputData+batchOffset*j+inBatchOffset,
					d_inputData+inputCountByChannel*j,
					inputCountByChannel,
					cudaMemcpyDeviceToDevice));
		}
	}
}


template <typename Dtype>
void DepthConcatLayer<Dtype>::backpropagation() {
	uint32_t batchOffset = 0;
	for (uint32_t i = 0; i < this->_inputData.size(); i++) {
		batchOffset += this->_inputData[i]->getCountByAxis(1);
	}

	const Dtype* d_outputData = this->_outputData[0]->device_data();
	const uint32_t batchSize = this->_inputData[0]->getShape()[0];
	uint32_t inBatchOffset = 0;
	for (uint32_t i = 0; i < this->_inputData.size(); i++) {
		Dtype* d_inputData = this->_inputData[i]->mutable_device_data();
		const uint32_t inputCountByChannel = this->_inputData[i]->getCountByAxis(1);
		if (i > 0) {
			inBatchOffset += this->_inputData[i-1]->getCountByAxis(1);
		}
		for (uint32_t j = 0; j < batchSize; j++) {
			checkCudaErrors(cudaMemcpyAsync(
					d_inputData+inputCountByChannel*j,
					d_outputData+batchOffset*j+inBatchOffset,
					inputCountByChannel,
					cudaMemcpyDeviceToDevice));
		}
	}
}


template class DepthConcatLayer<float>;






/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* DepthConcatLayer<Dtype>::initLayer() {
	DepthConcatLayer* layer = NULL;
	SNEW(layer, DepthConcatLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void DepthConcatLayer<Dtype>::destroyLayer(void* instancePtr) {
    DepthConcatLayer<Dtype>* layer = (DepthConcatLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void DepthConcatLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    DepthConcatLayer<Dtype>* layer = (DepthConcatLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DepthConcatLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DepthConcatLayer<Dtype>* layer = (DepthConcatLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void DepthConcatLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	DepthConcatLayer<Dtype>* layer = (DepthConcatLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void DepthConcatLayer<Dtype>::backwardTensor(void* instancePtr) {
	DepthConcatLayer<Dtype>* layer = (DepthConcatLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void DepthConcatLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool DepthConcatLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() < 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    for (int i = 1; i < inputShape.size(); i++) {
        outputShape1.C += inputShape[i].C;
    }
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t DepthConcatLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template void* DepthConcatLayer<float>::initLayer();
template void DepthConcatLayer<float>::destroyLayer(void* instancePtr);
template void DepthConcatLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool DepthConcatLayer<float>::allocLayerTensors(void* instancePtr);
template void DepthConcatLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void DepthConcatLayer<float>::backwardTensor(void* instancePtr);
template void DepthConcatLayer<float>::learnTensor(void* instancePtr);
template bool DepthConcatLayer<float>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape);
template uint64_t DepthConcatLayer<float>::calcGPUSize(vector<TensorShape> inputShape);
