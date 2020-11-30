/*
 * ReshapeLayer.cpp
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */


#if 1

#include <vector>

#include "ReshapeLayer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "Util.h"
#include "MemoryMgmt.h"

#define RESHAPELAYER_LOG 0

using namespace std;

template <typename Dtype>
ReshapeLayer<Dtype>::ReshapeLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Reshape;

	const vector<int>& shape = SLPROP(Reshape, shape);
	SASSERT0(shape.size() == 4);

	this->inferredAxis = -1;
	this->copyAxes.clear();
	const uint32_t topNumAxis = shape.size();
	this->constantCount = 1;

	for (uint32_t i = 0; i < topNumAxis; i++) {
		const int topDim = shape[i];
		if (topDim == 0) {
			copyAxes.push_back(i);
		} else if (topDim == -1) {
			SASSERT(inferredAxis == -1, "new shape contains multiple -1 dims ... ");
			this->inferredAxis = i;
		} else {
			this->constantCount *= topDim;
		}
	}
}


template <typename Dtype>
ReshapeLayer<Dtype>::~ReshapeLayer() {

}

template <typename Dtype>
void ReshapeLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	// Layer does not allow in-place computation.
	assert(this->_inputData[0] != this->_outputData[0]);


	const vector<uint32_t>& inputDataShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputDataShape;

	const uint32_t dim = inputDataShape.size();
	vector<uint32_t> outputDataShape(dim);

	for (uint32_t i = 0; i < dim; i++) {
		if (SLPROP(Reshape, shape)[i] > 0)
			outputDataShape[i] = SLPROP(Reshape, shape)[i];
	}
	for (uint32_t i = 0; i < copyAxes.size(); i++) {
		outputDataShape[this->copyAxes[i]] = inputDataShape[this->copyAxes[i]];
	}

	if (this->inferredAxis >= 0) {
		const uint32_t inputDataSize = this->_inputData[0]->getCount();
		uint32_t fixedSize = 1;
		for (uint32_t i = 0; i < dim; i++) {
			if (outputDataShape[i] > 0)
				fixedSize *= outputDataShape[i];
		}
		assert(inputDataSize % fixedSize == 0 &&
				"input count must be divisible by the product");
		outputDataShape[this->inferredAxis] = inputDataSize / fixedSize;
	}

	this->_outputData[0]->reshape(outputDataShape);

#if RESHAPELAYER_LOG
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			SLPROP_BASE(name).c_str(), outputDataShape[0], outputDataShape[1],
			outputDataShape[2], outputDataShape[3]);
#endif


	assert(this->_inputData[0]->getCount() == this->_outputData[0]->getCount() &&
			"output count must match input count");

	this->_outputData[0]->share_data(this->_inputData[0]);
	this->_outputData[0]->share_grad(this->_inputData[0]);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::feedforward() {
	reshape();

#if RESHAPELAYER_LOG
	Data<Dtype>::printConfig = true;
	const vector<uint32_t> shape;
	this->_inputData[0]->print_data(shape, false);
	this->_inputData[0]->print_grad(shape, false);
	this->_outputData[0]->print_data(shape, false);
	this->_outputData[0]->print_grad(shape, false);
	Data<Dtype>::printConfig = false;
#endif

}

template <typename Dtype>
void ReshapeLayer<Dtype>::backpropagation() {
	// do nothing ...


	/*
	if (SLPROP_BASE(name) == "rpn_cls_score_reshape") {
		Data<Dtype>::printConfig = true;

		this->_outputData[0]->print_grad({}, false);
		this->_inputData[0]->print_grad({}, false);

		Data<Dtype>::printConfig = false;
	}
	*/
}


template class ReshapeLayer<float>;




/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* ReshapeLayer<Dtype>::initLayer() {
	ReshapeLayer* layer = NULL;
	SNEW(layer, ReshapeLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ReshapeLayer<Dtype>::destroyLayer(void* instancePtr) {
    ReshapeLayer<Dtype>* layer = (ReshapeLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ReshapeLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    ReshapeLayer<Dtype>* layer = (ReshapeLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ReshapeLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ReshapeLayer<Dtype>* layer = (ReshapeLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ReshapeLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	ReshapeLayer<Dtype>* layer = (ReshapeLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void ReshapeLayer<Dtype>::backwardTensor(void* instancePtr) {
	ReshapeLayer<Dtype>* layer = (ReshapeLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void ReshapeLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool ReshapeLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;


	const vector<int>& shape = SLPROP(Reshape, shape);
	vector<uint32_t> copyAxes;
	const uint32_t topNumAxis = shape.size();
    const int dim = Data<Dtype>::SHAPE_SIZE;
	int inferredAxis = -1;
	int constantCount = 1;

	for (uint32_t i = 0; i < topNumAxis; i++) {
		const int topDim = shape[i];
		if (topDim == 0) {
			copyAxes.push_back(i);
		} else if (topDim == -1) {
			inferredAxis = i;
		} else {
			constantCount *= topDim;
		}
	}

    TensorShape outputShape1;
	for (uint32_t i = 0; i < dim; i++) {
		if (shape[i] > 0) {
            tensorRefByIndex(outputShape1, i) = shape[i];
        }
	}
	for (uint32_t i = 0; i < copyAxes.size(); i++) {
		tensorRefByIndex(outputShape1, copyAxes[i]) = 
            tensorValByIndex(inputShape[0], copyAxes[i]);
	}

	if (inferredAxis >= 0) {
		const size_t inputDataSize = tensorCount(inputShape[0]);
		uint32_t fixedSize = 1;
		for (uint32_t i = 0; i < dim; i++) {
			if (tensorValByIndex(outputShape1, i) > 0)
				fixedSize *= tensorValByIndex(outputShape1, i);
		}
        tensorRefByIndex(outputShape1, inferredAxis) = inputDataSize / fixedSize;
	}

    outputShape.push_back(outputShape1);
    return true;
}

template<typename Dtype>
uint64_t ReshapeLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template void* ReshapeLayer<float>::initLayer();
template void ReshapeLayer<float>::destroyLayer(void* instancePtr);
template void ReshapeLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool ReshapeLayer<float>::allocLayerTensors(void* instancePtr);
template void ReshapeLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void ReshapeLayer<float>::backwardTensor(void* instancePtr);
template void ReshapeLayer<float>::learnTensor(void* instancePtr);
template bool ReshapeLayer<float>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape);
template uint64_t ReshapeLayer<float>::calcGPUSize(vector<TensorShape> inputShape);




#endif
