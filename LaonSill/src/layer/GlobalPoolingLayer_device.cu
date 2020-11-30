/**
 * @file GlobalPoolingLayer_device.cu
 * @date 2017-12-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "GlobalPoolingLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
void GlobalPoolingLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

}

template <typename Dtype>
void GlobalPoolingLayer<Dtype>::feedforward() {
	reshape();
}

template <typename Dtype>
void GlobalPoolingLayer<Dtype>::backpropagation() {
}

template void GlobalPoolingLayer<float>::reshape();
template void GlobalPoolingLayer<float>::feedforward();
template void GlobalPoolingLayer<float>::backpropagation();

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* GlobalPoolingLayer<Dtype>::initLayer() {
	GlobalPoolingLayer* layer = NULL;
	SNEW(layer, GlobalPoolingLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void GlobalPoolingLayer<Dtype>::destroyLayer(void* instancePtr) {
    GlobalPoolingLayer<Dtype>* layer = (GlobalPoolingLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void GlobalPoolingLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    GlobalPoolingLayer<Dtype>* layer = (GlobalPoolingLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool GlobalPoolingLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    GlobalPoolingLayer<Dtype>* layer = (GlobalPoolingLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void GlobalPoolingLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	GlobalPoolingLayer<Dtype>* layer = (GlobalPoolingLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void GlobalPoolingLayer<Dtype>::backwardTensor(void* instancePtr) {
	GlobalPoolingLayer<Dtype>* layer = (GlobalPoolingLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void GlobalPoolingLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool GlobalPoolingLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t GlobalPoolingLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template void* GlobalPoolingLayer<float>::initLayer();
template void GlobalPoolingLayer<float>::destroyLayer(void* instancePtr);
template void GlobalPoolingLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool GlobalPoolingLayer<float>::allocLayerTensors(void* instancePtr);
template void GlobalPoolingLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void GlobalPoolingLayer<float>::backwardTensor(void* instancePtr);
template void GlobalPoolingLayer<float>::learnTensor(void* instancePtr);
template bool GlobalPoolingLayer<float>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape);
template uint64_t GlobalPoolingLayer<float>::calcGPUSize(vector<TensorShape> inputShape);
