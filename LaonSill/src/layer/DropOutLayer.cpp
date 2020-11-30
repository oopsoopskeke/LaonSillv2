/**
 * @file DropOutLayer.cpp
 * @date 2017-04-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "DropOutLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
DropOutLayer<Dtype>::DropOutLayer() : Layer<Dtype>() {
	this->type = Layer<Dtype>::DropOut;

	SyncMem<Dtype>* ptr = NULL;
	//SNEW(ptr, SyncMem<Dtype>);
	ptr = new SyncMem<Dtype>();
	SASSUME0(ptr != NULL);
    shared_ptr<SyncMem<Dtype>> tempMask(ptr);
    this->mask = tempMask;
}

template<typename Dtype>
DropOutLayer<Dtype>::~DropOutLayer() {

}

template <typename Dtype>
void DropOutLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputShape;

	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	this->_outputData[0]->reshape(inputShape);
    this->mask->reshape(batches * channels * rows * cols);
}

template <typename Dtype>
void DropOutLayer<Dtype>::feedforward() {
	reshape();
    doDropOutForward();
}

template <typename Dtype>
void DropOutLayer<Dtype>::backpropagation() {
    doDropOutBackward();
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* DropOutLayer<Dtype>::initLayer() {
	DropOutLayer* layer = NULL;
	SNEW(layer, DropOutLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void DropOutLayer<Dtype>::destroyLayer(void* instancePtr) {
    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void DropOutLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DropOutLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void DropOutLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void DropOutLayer<Dtype>::backwardTensor(void* instancePtr) {
    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void DropOutLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool DropOutLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t DropOutLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class DropOutLayer<float>;
