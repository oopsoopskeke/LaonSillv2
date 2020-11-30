/*
 * FlattenLayer.cpp
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#include "FlattenLayer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
FlattenLayer<Dtype>::FlattenLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Flatten;
}

template <typename Dtype>
FlattenLayer<Dtype>::~FlattenLayer() {

}

template <typename Dtype>
void FlattenLayer<Dtype>::reshape() {
	SASSERT(SLPROP_BASE(input)[0] != SLPROP_BASE(output)[0],
			"Flatten layer does not allow in-place computation.");

	//if (SLPROP_BASE(name) == "conv4_3_norm_mbox_loc_flat") {
	//	cout << endl;
	//}

	Layer<Dtype>::_adjustInputShape();
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	// TODO: Data에 CanonicalAxis 구현 필요
	const int startAxis = SLPROP(Flatten, axis);
	const int endAxis = this->_inputData[0]->getShape().size()-1;

	vector<uint32_t> outputShape;
	for (int i = 0; i < startAxis; i++) {
		outputShape.push_back(this->_inputData[0]->getShape(i));
	}
	const int flattenedDim = this->_inputData[0]->getCountByAxis(startAxis, endAxis + 1);
	outputShape.push_back(flattenedDim);
	for (int i = endAxis + 1; i < this->_inputData[0]->numAxes(); i++) {
		outputShape.push_back(this->_inputData[0]->getShape(i));
	}

	// TODO: flatten후 shape size가 4가 아닌 상황,
	// 4가 되도록 보정해야 함.
	for (int i = outputShape.size(); i < this->_inputData[0]->numAxes(); i++) {
		outputShape.push_back(1);
	}

	//cout << "FlattenLayer: " << SLPROP_BASE(name) << endl;
	//for (int i = 0; i < outputShape.size(); i++) {
	//	cout << outputShape[i] << ",";
	//}
	//cout << endl;

	this->_outputData[0]->reshape(outputShape);
	SASSERT0(this->_outputData[0]->getCount() == this->_inputData[0]->getCount());
}

template <typename Dtype>
void FlattenLayer<Dtype>::feedforward() {
	reshape();
	this->_outputData[0]->share_data(this->_inputData[0]);
}

template <typename Dtype>
void FlattenLayer<Dtype>::backpropagation() {
	this->_inputData[0]->share_grad(this->_outputData[0]);
}













/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* FlattenLayer<Dtype>::initLayer() {
	FlattenLayer* layer = NULL;
	SNEW(layer, FlattenLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void FlattenLayer<Dtype>::destroyLayer(void* instancePtr) {
    FlattenLayer<Dtype>* layer = (FlattenLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void FlattenLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	if (isInput) {
		SASSERT0(index < 1);
	} else {
		SASSERT0(index < 1);
	}

    FlattenLayer<Dtype>* layer = (FlattenLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool FlattenLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    FlattenLayer<Dtype>* layer = (FlattenLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void FlattenLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	FlattenLayer<Dtype>* layer = (FlattenLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void FlattenLayer<Dtype>::backwardTensor(void* instancePtr) {
	FlattenLayer<Dtype>* layer = (FlattenLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void FlattenLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool FlattenLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

	const int startAxis = SLPROP(Flatten, axis);
	const int endAxis = 3;

    TensorShape outputShape1 = inputShape[0];

    for (int i = 0; i < startAxis; i++) {
        tensorRefByIndex(outputShape1, i) = tensorValByIndex(inputShape[0], i);
    }

    int flattenedDim = 1;
    for (int i = startAxis; i <= endAxis; i++) {
        flattenedDim *= tensorValByIndex(inputShape[0], i);
    }
    tensorRefByIndex(outputShape1, startAxis) = flattenedDim;
    for (int i = startAxis + 1; i <= endAxis; i++) {
        tensorRefByIndex(outputShape1, i) = 1;
    }
    return true;
}

template<typename Dtype>
uint64_t FlattenLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class FlattenLayer<float>;
