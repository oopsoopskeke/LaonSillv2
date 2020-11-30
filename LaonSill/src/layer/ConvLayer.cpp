/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "Util.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* ConvLayer<Dtype>::initLayer() {
	ConvLayer* layer = NULL;
	SNEW(layer, ConvLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ConvLayer<Dtype>::destroyLayer(void* instancePtr) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ConvLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ConvLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ConvLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void ConvLayer<Dtype>::backwardTensor(void* instancePtr) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void ConvLayer<Dtype>::learnTensor(void* instancePtr) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    layer->update();
}

template<typename Dtype>
bool ConvLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    const bool deconv = SLPROP(Conv, deconv);
	filter_dim& filterDim = SLPROP(Conv, filterDim);

    if (inputShape.size() != 1)
        return false;

    if (filterDim.channels != inputShape[0].C)
        return false;

    TensorShape outputShape1;
    if (!deconv) {
        outputShape1.N = inputShape[0].N;
        outputShape1.C = filterDim.filters;
        outputShape1.H = int((inputShape[0].H - filterDim.rows + 2 * filterDim.pad) / 
                filterDim.stride) + 1;
        outputShape1.W = int((inputShape[0].W - filterDim.cols + 2 * filterDim.pad) / 
                filterDim.stride) + 1;
    } else {
	    const int deconvExtraCell = SLPROP(Conv, deconvExtraCell);
        outputShape1.N = inputShape[0].N;
        outputShape1.C = filterDim.filters;
        outputShape1.H = filterDim.stride * (inputShape[0].H - 1) + filterDim.rows -
            2 * filterDim.pad + deconvExtraCell;
        outputShape1.W = filterDim.stride * (inputShape[0].W - 1) + filterDim.cols -
            2 * filterDim.pad + deconvExtraCell;
    }

    if ((outputShape1.H < 1) || (outputShape1.W < 1))
        return false;

    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t ConvLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
	Optimizer opt = (Optimizer)SNPROP(optimizer);
	int paramHistoryDataCount = Update<Dtype>::getParamHistoryDataCount(opt);
	filter_dim& filterDim = SLPROP(Conv, filterDim);
    bool biasTerm = SLPROP(Conv, biasTerm);
   
    uint64_t filterSize = ALIGNUP(sizeof(Dtype) * filterDim.filters * filterDim.channels * 
        filterDim.cols * filterDim.rows, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    uint64_t biasSize = 0UL;
    if (biasTerm) {
        biasSize = 
            ALIGNUP(sizeof(Dtype) * filterDim.filters, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    }

    return paramHistoryDataCount * (filterSize + biasSize);
}

template class ConvLayer<float>;
