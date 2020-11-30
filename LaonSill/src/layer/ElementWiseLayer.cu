/*
 * ElementWiseLayer.cpp
 *
 *  Created on: Aug 4, 2017
 *      Author: jkim
 */

#include "ElementWiseLayer.h"
#include "PropMgmt.h"
#include "MathFunctions.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "Cuda.h"
#include "MemoryMgmt.h"



using namespace std;

template <typename Dtype>
ElementWiseLayer<Dtype>::ElementWiseLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::ElementWise;
	SASSERT(SLPROP(ElementWise, coeff).size() == 0 ||
			SLPROP(ElementWise, coeff).size() == SLPROP_BASE(input).size(),
			"ElementWise Layer takes one coefficient per input data.");
	SASSERT(!(SLPROP(ElementWise, operation) == ElementWiseOp::PROD &&
			SLPROP(ElementWise, coeff).size()),
			"ElementWise Layer only takes coefficients for summation.");
	this->op = SLPROP(ElementWise, operation);
	// Data-wise coefficients for the elementwise operation.
	const int inputSize = SLPROP_BASE(input).size();
	this->coeffs = vector<Dtype>(inputSize, 1);
	if (SLPROP(ElementWise, coeff).size()) {
		for (int i = 0; i < inputSize; i++) {
			this->coeffs[i] = SLPROP(ElementWise, coeff)[i];
		}
	}
	this->stableProdGrad = SLPROP(ElementWise, stableProdGrad);
}

template <typename Dtype>
ElementWiseLayer<Dtype>::~ElementWiseLayer() {

}

template <typename Dtype>
void ElementWiseLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	const int inputSize = this->_inputData.size();

    // 모든 input의 shape들이 동일해야 하므로 
    // 첫번째 입력에 대해서만 shape 변화 확인.
    // (모든 input shape들이 동일함은 아래의 테스트에서 확인.
    if (!Layer<Dtype>::_isInputShapeChanged(0)) {
        return;
    }

    this->_inputShape[0] = this->_inputData[0]->getShape();
	for (int i = 1; i < inputSize; i++) {
		SASSERT0(this->_inputData[i]->getShape() == this->_inputData[0]->getShape());
        this->_inputShape[i] = this->_inputData[i]->getShape();
	}
	this->_outputData[0]->reshapeLike(this->_inputData[0]);

	// If max operation, we will initialize the vector index part.
	if (this->op == ElementWiseOp::MAX && this->_outputData.size() == 1) {
		this->maxIdx.reshape(this->_inputData[0]->getShape());
	}
}



template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
		const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data, int* mask) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		Dtype maxval = -FLT_MAX;
		int maxidx = -1;
		if (bottom_data_a[index] > bottom_data_b[index]) {
			// only update for very first bottom_data blob (blob_idx == 0)
			if (blob_idx == 0) {
				maxval = bottom_data_a[index];
				top_data[index] = maxval;
				maxidx = blob_idx;
				mask[index] = maxidx;
			}
		} else {
			maxval = bottom_data_b[index];
			top_data[index] = maxval;
			maxidx = blob_idx + 1;
			mask[index] = maxidx;
		}
	}
}


template <typename Dtype>
void ElementWiseLayer<Dtype>::feedforward() {
	reshape();

	int* mask = NULL;
	const int count = this->_outputData[0]->getCount();
	Dtype* outputData = this->_outputData[0]->mutable_device_data();

	switch(this->op) {
	case ElementWiseOp::PROD:
		soooa_gpu_mul(count, this->_inputData[0]->device_data(),
				this->_inputData[1]->device_data(), outputData);
		for (int i = 2; i < this->_inputData.size(); i++) {
			soooa_gpu_mul(count, outputData, this->_inputData[i]->device_data(), outputData);
		}
		break;
	case ElementWiseOp::SUM:
		soooa_gpu_set(count, Dtype(0.), outputData);
		for (int i = 0; i < this->_inputData.size(); i++) {
			soooa_gpu_axpy(count, this->coeffs[i], this->_inputData[i]->device_data(),
					outputData);
		}
		break;
	case ElementWiseOp::MAX:
		mask = this->maxIdx.mutable_device_data();
		MaxForward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
				count, this->_inputData[0]->device_data(), this->_inputData[1]->device_data(),
				0, outputData, mask);
		for (int i = 2; i < this->_inputData.size(); i++) {
			MaxForward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
					count, outputData, this->_inputData[i]->device_data(), i - 1, outputData,
					mask);
		}
		CUDA_POST_KERNEL_CHECK;
		break;
	default:
		SASSERT(false, "Unknown elementwise operation.");
	}
}


template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff, const int blob_idx,
		const int* mask, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		Dtype gradient = 0;
		if (mask[index] == blob_idx) {
			gradient += top_diff[index];
		}
		bottom_diff[index] = gradient;
	}
}




template <typename Dtype>
void ElementWiseLayer<Dtype>::backpropagation() {
	const int* mask = NULL;
	const int count = this->_outputData[0]->getCount();
	const Dtype* outputData = this->_outputData[0]->device_data();
	const Dtype* outputGrad = this->_outputData[0]->device_grad();
	for (int i = 0; i < this->_inputData.size(); i++) {
		if (SLPROP_BASE(propDown)[i]) {
			const Dtype* inputData = this->_inputData[i]->device_data();
			Dtype* inputGrad = this->_inputData[i]->mutable_device_grad();
			switch(this->op) {
			case ElementWiseOp::PROD:
				if (this->stableProdGrad) {
					bool initialized = false;
					for (int j = 0; j < this->_inputData.size(); j++) {
						if (i == j) {
							continue;
						}
						if (!initialized) {
							soooa_copy(count, this->_inputData[j]->device_data(), inputGrad);
							initialized = true;
						} else {
							soooa_gpu_mul(count, this->_inputData[j]->device_data(),
									inputGrad, inputGrad);
						}
					}
				} else {
					soooa_gpu_div(count, outputData, inputData, inputGrad);
				}
				soooa_gpu_mul(count, inputGrad, outputGrad, inputGrad);
				break;
			case ElementWiseOp::SUM:
				if (this->coeffs[i] == Dtype(1.)) {
					soooa_copy(count, outputGrad, inputGrad);
				} else {
					soooa_gpu_scale(count, this->coeffs[i], outputGrad, inputGrad);
				}
				break;
			case ElementWiseOp::MAX:
				mask = this->maxIdx.device_data();
				MaxBackward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
						count, outputGrad, i, mask, inputGrad);
				break;
			default:
				SASSERT(false, "Unknown elementwise operation.");
			}
		}
	}
}







/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* ElementWiseLayer<Dtype>::initLayer() {
	ElementWiseLayer* layer = NULL;
	SNEW(layer, ElementWiseLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ElementWiseLayer<Dtype>::destroyLayer(void* instancePtr) {
    ElementWiseLayer<Dtype>* layer = (ElementWiseLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ElementWiseLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	if (!isInput) {
		SASSERT0(index < 1);
	}

    ElementWiseLayer<Dtype>* layer = (ElementWiseLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ElementWiseLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ElementWiseLayer<Dtype>* layer = (ElementWiseLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ElementWiseLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	ElementWiseLayer<Dtype>* layer = (ElementWiseLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void ElementWiseLayer<Dtype>::backwardTensor(void* instancePtr) {
	ElementWiseLayer<Dtype>* layer = (ElementWiseLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void ElementWiseLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool ElementWiseLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() < 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t ElementWiseLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    uint64_t size = 0;

	if (SLPROP(ElementWise, operation) == ElementWiseOp::MAX) {
        const size_t inputCount = tensorCount(inputShape[0]);
        size += ALIGNUP(sizeof(Dtype) * inputCount, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    } 
    return size;
}


template class ElementWiseLayer<float>;
