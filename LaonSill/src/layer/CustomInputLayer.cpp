/**
 * @file CustomInputLayer.cpp
 * @date 2017-06-29
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "CustomInputLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

template<typename Dtype>
void CustomInputLayer<Dtype>::registerCBFunc(CBCustomInputForward forwardFunc, void* args) {
    this->forwardFunc = forwardFunc;
    this->forwardFuncArgs = args;
}

template<typename Dtype>
CustomInputLayer<Dtype>::~CustomInputLayer() {
    float* data;

    for (int i = 0; i < this->dataArray.size(); i++) {
        data = this->dataArray[i];
        SFREE(data);
    }
}

template<typename Dtype>
void CustomInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void CustomInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    int batchSize = SNPROP(batchSize);
    reshape();

    this->forwardFunc(baseIndex, batchSize, this->forwardFuncArgs, dataArray);

    for (int i = 0; i < SLPROP_BASE(output).size(); i++) {
        int size = SLPROP(CustomInput, inputElemCounts)[i] * batchSize;
        this->_inputData[i]->set_device_with_host_data(dataArray[i], 0, size);
    }
}

template<typename Dtype>
int CustomInputLayer<Dtype>::getNumTrainData() {
    return SLPROP(CustomInput, trainDataCount);
}

template<typename Dtype>
int CustomInputLayer<Dtype>::getNumTestData() {
    return 0;
}

template<typename Dtype>
void CustomInputLayer<Dtype>::reshape() {
    int batchSize = SNPROP(batchSize);

	if (this->_inputData.size() < 1) {
        SASSERT0(SLPROP_BASE(output).size() == SLPROP(CustomInput, inputElemCounts).size());
        SASSERT0(this->dataArray.size() == 0);

		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
            int elemCnt = SLPROP(CustomInput, inputElemCounts)[i] * batchSize;
            float* data = NULL;
            int allocSize = sizeof(float) * elemCnt;
            SMALLOC(data, float, allocSize);
            this->dataArray.push_back(data);
		}
	}

	Layer<Dtype>::_adjustInputShape();

    for (int i = 0; i < SLPROP(CustomInput, inputElemCounts).size(); i++) {
        this->_inputShape[i][0] = batchSize;
        this->_inputShape[i][1] = SLPROP(CustomInput, inputElemCounts)[i];
        this->_inputShape[i][2] = 1;
        this->_inputShape[i][3] = 1;
    
        this->_inputData[i]->reshape(this->_inputShape[i]);
    }
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* CustomInputLayer<Dtype>::initLayer() {
	CustomInputLayer* layer = NULL;
	SNEW(layer, CustomInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void CustomInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    CustomInputLayer<Dtype>* layer = (CustomInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void CustomInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    CustomInputLayer<Dtype>* layer = (CustomInputLayer<Dtype>*)instancePtr;

    SASSERT0(!isInput);
    SASSERT0(index < 2);
    SASSERT0(layer->_outputData.size() == index);
    layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool CustomInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    CustomInputLayer<Dtype>* layer = (CustomInputLayer<Dtype>*)instancePtr;
    layer->reshape();

    if (SNPROP(miniBatch) == 0) {
        int trainDataNum = layer->getNumTrainData();
        if (trainDataNum % SNPROP(batchSize) == 0) {
            SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize);
        } else {
            SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize) + 1;
        }
        WorkContext::curPlanInfo->miniBatchCount = SNPROP(miniBatch);
    }
    return true;
}

template<typename Dtype>
void CustomInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    CustomInputLayer<Dtype>* layer = (CustomInputLayer<Dtype>*)instancePtr;
    layer->feedforward(miniBatchIdx);
}

template<typename Dtype>
void CustomInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void CustomInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool CustomInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    for (int i = 0; i < SLPROP(CustomInput, inputElemCounts).size(); i++) {
        TensorShape outputShape1;

        outputShape1.N = SNPROP(batchSize);
        outputShape1.C = SLPROP(CustomInput, inputElemCounts)[i];
        outputShape1.H = 1;
        outputShape1.W = 1;

        outputShape.push_back(outputShape1);
    }

    return true;
}

template<typename Dtype>
uint64_t CustomInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class CustomInputLayer<float>;
