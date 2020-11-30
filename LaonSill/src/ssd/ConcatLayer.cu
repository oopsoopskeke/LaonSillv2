/*
 * ConcatLayer.cpp
 *
 *  Created on: Apr 26, 2017
 *      Author: jkim
 */

#include "ConcatLayer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
		const bool forward, const int num_concats, const int concat_size,
		const int top_concat_axis, const int bottom_concat_axis,
		const int offset_concat_axis, Dtype* out_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int total_concat_size = concat_size * bottom_concat_axis;
		const int concat_num = index / total_concat_size;
		const int concat_index = index % total_concat_size;
		const int top_index = concat_index +
				(concat_num * top_concat_axis + offset_concat_axis) * concat_size;
		if (forward) {
			out_data[top_index] = in_data[index];
		} else {
			out_data[index] = in_data[top_index];
		}
	}
}


template <typename Dtype>
ConcatLayer<Dtype>::ConcatLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Concat;

	SASSERT(SLPROP(Concat, axis) >= 0, "axis should be specified ... ");
	this->tempCount = 0;
}

template <typename Dtype>
ConcatLayer<Dtype>::~ConcatLayer() {
}

template <typename Dtype>
void ConcatLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	bool inputShapeChanged = false;
	for (int i = 0; i < this->_inputData.size(); i++) {
		if (Layer<Dtype>::_isInputShapeChanged(0)) {
			inputShapeChanged = true;
			break;
		}
	}

	if (!inputShapeChanged)
		return;

	const int numAxes = this->_inputData[0]->numAxes();
	SASSERT0(SLPROP(Concat, axis) < numAxes);

	// Initialize with the first Data.
	vector<uint32_t> outputShape = this->_inputData[0]->getShape();
	this->numConcat = this->_inputData[0]->getCountByAxis(0, SLPROP(Concat, axis));
	this->concatInputSize = this->_inputData[0]->getCountByAxis(SLPROP(Concat, axis) + 1);

	int inputCountSum = this->_inputData[0]->getCount();
	for (int i = 1; i < this->_inputData.size(); i++) {
		SASSERT(numAxes == this->_inputData[i]->numAxes(),
				"[%s] All inputs must have the same #axes.", this->getName().c_str());
		for (int j = 0; j < numAxes; j++) {
			if (j == SLPROP(Concat, axis))
				continue;
			SASSERT(outputShape[j] == this->_inputData[i]->getShape(j),
					"[%s] All inputs must have the same shape, except at concatAxis.",
					this->getName().c_str());
		}
		inputCountSum += this->_inputData[i]->getCount();
		outputShape[SLPROP(Concat, axis)] += this->_inputData[i]->getShape(SLPROP(Concat, axis));
	}
	this->_outputData[0]->reshape(outputShape);
	SASSERT0(inputCountSum == this->_outputData[0]->getCount());

	if (this->_inputData.size() == 1) {
		this->_outputData[0]->share_data(this->_inputData[0]);
		this->_outputData[0]->share_grad(this->_inputData[0]);
	}
}

template <typename Dtype>
void ConcatLayer<Dtype>::feedforward() {
	reshape();

	if (this->_inputData.size() == 1)
		return;

	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	int offsetConcatAxis = 0;
	const int outputConcatAxis = this->_outputData[0]->getShape(SLPROP(Concat, axis));
	const bool kForward = true;
	for (int i = 0; i < this->_inputData.size(); i++) {
		const Dtype* inputData = this->_inputData[i]->device_data();
		const int inputConcatAxis = this->_inputData[i]->getShape(SLPROP(Concat, axis));
		const int inputConcatSize = inputConcatAxis * this->concatInputSize;
		const int nthreads = inputConcatSize * this->numConcat;
		Concat<Dtype><<<SOOOA_GET_BLOCKS(nthreads), SOOOA_CUDA_NUM_THREADS>>>(
				nthreads, inputData, kForward, this->numConcat, this->concatInputSize,
				outputConcatAxis, inputConcatAxis, offsetConcatAxis, outputData);
		offsetConcatAxis += inputConcatAxis;
	}
}

template <typename Dtype>
void ConcatLayer<Dtype>::backpropagation() {
	if (this->_inputData.size() == 1)
		return;

	const Dtype* outputGrad = this->_outputData[0]->device_grad();
	int offsetConcatAxis = 0;
	const int outputConcatAxis = this->_outputData[0]->getShape(SLPROP(Concat, axis));
	const bool kForward = false;
	for (int i = 0; i < this->_inputData.size(); i++) {
		const int inputConcatAxis = this->_inputData[i]->getShape(SLPROP(Concat, axis));
		if (SLPROP_BASE(propDown)[i]) {
			Dtype* inputGrad = this->_inputData[i]->mutable_device_grad();
			const int inputConcatSize = inputConcatAxis * this->concatInputSize;
			const int nthreads = inputConcatSize * this->numConcat;
			Concat<Dtype><<<SOOOA_GET_BLOCKS(nthreads), SOOOA_CUDA_NUM_THREADS>>>(
					nthreads, outputGrad, kForward, this->numConcat, this->concatInputSize,
					outputConcatAxis, inputConcatAxis, offsetConcatAxis, inputGrad);
		}
		offsetConcatAxis += inputConcatAxis;
	}

	/*
	if (this->name == "mbox_priorbox") {
		this->tempCount++;
		if (this->tempCount == 2) {
			this->_printOn();
			//this->_outputData[0]->print_grad({}, false, -1);
			for (int i = 0; i < this->_inputData.size(); i++) {
				//this->_inputData[i]->print_grad({}, false, -1);
				cout << this->_inputData[i]->asum_device_grad() << endl;
			}
			this->_printOff();
			exit(1);
		}
	}
	*/
}















/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* ConcatLayer<Dtype>::initLayer() {
	ConcatLayer* layer = NULL;
	SNEW(layer, ConcatLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ConcatLayer<Dtype>::destroyLayer(void* instancePtr) {
    ConcatLayer<Dtype>* layer = (ConcatLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ConcatLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	if (!isInput) {
		SASSERT0(index < 1);
	}

    ConcatLayer<Dtype>* layer = (ConcatLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ConcatLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ConcatLayer<Dtype>* layer = (ConcatLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ConcatLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	ConcatLayer<Dtype>* layer = (ConcatLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void ConcatLayer<Dtype>::backwardTensor(void* instancePtr) {
	ConcatLayer<Dtype>* layer = (ConcatLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void ConcatLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}



int getCountByAxis(const vector<uint32_t>& shape, const int start, const int end) {
    SASSERT0(shape.size() == 4);
    SASSERT0(start >= 0 && end >= start && end < 4);

    int count = 1;
    for (int i = start; i <= end; i++) {
        count *= shape[i];
    }
    return count;
}


template<typename Dtype>
bool ConcatLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() < 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    const int axis = SLPROP(Concat, axis);
    for (int i = 1; i < inputShape.size(); i++) {
        tensorRefByIndex(outputShape1, axis) += tensorValByIndex(inputShape[i], axis);
    }
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t ConcatLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class ConcatLayer<float>;
