/*
 * PermuteLayer.cpp
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#include "PermuteLayer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
__global__ void PermuteKernel(const int nthreads, Dtype* const bottom_data,
		const bool forward, const int* permute_order, const int* old_steps,
		const int* new_steps, const int num_axes, Dtype* const top_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int temp_idx = index;
		int old_idx = 0;
		for (int i = 0; i < num_axes; ++i) {
			int order = permute_order[i];
			old_idx += (temp_idx / new_steps[i]) * old_steps[order];
			temp_idx %= new_steps[i];
		}
		if (forward) {
			top_data[index] = bottom_data[old_idx];
		} else {
			bottom_data[old_idx] = top_data[index];
		}
	}
}



template <typename Dtype>
PermuteLayer<Dtype>::PermuteLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Permute;

	this->numAxes = 4;

	vector<uint32_t> orders;
	// push the specified new orders.
	for (int i = 0; i < SLPROP(Permute, order).size(); i++) {
		int order = SLPROP(Permute, order)[i];
		SASSERT(order < this->numAxes, "order should be less than the input dimension.");
		if (std::find(orders.begin(), orders.end(), order) != orders.end()) {
			SASSERT(false, "there are duplicate orders");
		}
		orders.push_back(order);
	}

	// push the rest orders. And save original step sizes for each axis
	for (int i = 0; i < this->numAxes; i++) {
		if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
			orders.push_back(i);
		}
	}
	SASSERT0(this->numAxes == orders.size());

	// check if we need to reorder the data or keep it
	this->needPermute = false;
	for (int i = 0; i < this->numAxes; i++) {
		if (orders[i] != i) {
			// as long as there is one order which is different from the natural order
			// of the data, we need to permute. Otherwise, we share the data and grad
			this->needPermute = true;
			break;
		}
	}

	this->permuteOrder_.reshape({this->numAxes, 1, 1, 1});
	this->oldSteps_.reshape({this->numAxes, 1, 1, 1});
	this->newSteps_.reshape({this->numAxes, 1, 1, 1});

	for (int i = 0; i < this->numAxes; i++) {
		this->permuteOrder_.mutable_host_data()[i] = orders[i];
	}

	SLPROP(Permute, order) = orders;
}

template <typename Dtype>
PermuteLayer<Dtype>::~PermuteLayer() {

}

template <typename Dtype>
void PermuteLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	vector<uint32_t> outputShape;
	for (int i = 0; i < this->numAxes; i++) {
		if (i == this->numAxes - 1) {
			this->oldSteps_.mutable_host_data()[i] = 1;
		} else {
			this->oldSteps_.mutable_host_data()[i] =
					this->_inputData[0]->getCountByAxis(i + 1);
		}
		outputShape.push_back(this->_inputData[0]->getShape(
				this->permuteOrder_.host_data()[i]));
	}
	this->_outputData[0]->reshape(outputShape);

	for (int i = 0; i < this->numAxes; i++) {
		if (i == this->numAxes - 1) {
			this->newSteps_.mutable_host_data()[i] = 1;
		} else {
			this->newSteps_.mutable_host_data()[i] =
					this->_outputData[0]->getCountByAxis(i + 1);
		}
	}
}

template <typename Dtype>
void PermuteLayer<Dtype>::feedforward() {
	reshape();

	if (this->needPermute) {
		Dtype* inputData = this->_inputData[0]->mutable_device_data();
		Dtype* outputData = this->_outputData[0]->mutable_device_data();
		int count = this->_outputData[0]->getCount();
		const int* permuteOrder = this->permuteOrder_.device_data();
		const int* newSteps = this->newSteps_.device_data();
		const int* oldSteps = this->oldSteps_.device_data();

		bool forward = true;
		PermuteKernel<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
				count, inputData, forward, permuteOrder, oldSteps, newSteps, this->numAxes,
				outputData);
		CUDA_POST_KERNEL_CHECK;
	} else {
		// if there is no need to permute, we share data to save memory
		this->_outputData[0]->share_data(this->_inputData[0]);
	}
}

template <typename Dtype>
void PermuteLayer<Dtype>::backpropagation() {
	if (this->needPermute) {
		Dtype* outputGrad = this->_outputData[0]->mutable_device_grad();
		Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
		const int count = this->_inputData[0]->getCount();
		const int* permuteOrder = this->permuteOrder_.device_data();
		const int* newSteps = this->newSteps_.device_data();
		const int* oldSteps = this->oldSteps_.device_data();

		bool forward = false;
		PermuteKernel<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
				count, inputGrad, forward, permuteOrder, oldSteps, newSteps, this->numAxes,
				outputGrad);
		CUDA_POST_KERNEL_CHECK;
	} else {
		// if there is no need to permute, we share grad to save memory
		//this->_inputData[0]->_grad = this->_outputData[0]->_grad;
		this->_inputData[0]->share_grad(this->_outputData[0]);
	}
}


























/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* PermuteLayer<Dtype>::initLayer() {
	PermuteLayer* layer = NULL;
	SNEW(layer, PermuteLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void PermuteLayer<Dtype>::destroyLayer(void* instancePtr) {
    PermuteLayer<Dtype>* layer = (PermuteLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void PermuteLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 1);
	} else {
		SASSERT0(index < 1);
	}

    PermuteLayer<Dtype>* layer = (PermuteLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool PermuteLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    PermuteLayer<Dtype>* layer = (PermuteLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void PermuteLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	PermuteLayer<Dtype>* layer = (PermuteLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void PermuteLayer<Dtype>::backwardTensor(void* instancePtr) {
	PermuteLayer<Dtype>* layer = (PermuteLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void PermuteLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool PermuteLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1;
    const vector<uint32_t>& order = SLPROP(Permute, order);
    outputShape1.N = order[0];
    outputShape1.C = order[1];
    outputShape1.H = order[2];
    outputShape1.W = order[3];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t PermuteLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {

    const int numAxes = 4;
    size_t size = 0;

    // permuteOrder
    size += numAxes;

    // oldSteps
    size += numAxes;

    // newSteps
    size += numAxes;

    return ALIGNUP(sizeof(Dtype) * size, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

}

template class PermuteLayer<float>;
