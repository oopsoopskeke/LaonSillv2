/**
 * @file SigmoidLayer2_device.cu
 * @date 2017-02-07
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "Sigmoid2Layer.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "ColdLog.h"
#include "Perf.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

#define SIGMOID2LAYER_LOG   1

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

template <typename Dtype>
__global__ void Forward(const Dtype *input, int size, Dtype *output)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	output[idx] = 1.0 / (1.0 + exp((-1.0) * input[idx]));
}

template <typename Dtype>
__global__ void Backward(const Dtype *outputGrad, const Dtype *output, int size,
    Dtype *inputGrad)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	inputGrad[idx] = outputGrad[idx] * output[idx] * (1.0 - output[idx]);
}

template <typename Dtype>
Sigmoid2Layer<Dtype>::Sigmoid2Layer() : Layer<Dtype>() {
	this->type = Layer<Dtype>::Sigmoid2;
}

template <typename Dtype>
void Sigmoid2Layer<Dtype>::feedforward() {
    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* outputData = this->_outputData[0]->mutable_device_data();
    int size = this->_inputData[0]->getCountByAxis(0);

    Forward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, size, outputData);
}

template <typename Dtype>
void Sigmoid2Layer<Dtype>::backpropagation() {
	const Dtype* outputGrads = this->_outputData[0]->device_grad();
    const Dtype* output = this->_outputData[0]->device_data();
	Dtype* inputGrads = this->_inputData[0]->mutable_device_grad();
    int size = this->_inputData[0]->getCountByAxis(0);

    Backward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        outputGrads, output, size, inputGrads);
}

template <typename Dtype>
void Sigmoid2Layer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
		const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
		const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
		assert(count == inputDataCount);
	}

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();

    // XXX: 현재 FC에 대해서만 생각하였음
    // TODO: Conv Layer에 대한 구현 필요
	uint32_t batches = inputShape[0];
	uint32_t channels = inputShape[1];
	uint32_t rows = inputShape[2];
	uint32_t cols = inputShape[3];

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({batches, channels, rows, cols});

	STDOUT_COND_LOG(SIGMOID2LAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);
	STDOUT_COND_LOG(SIGMOID2LAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* Sigmoid2Layer<Dtype>::initLayer() {
	Sigmoid2Layer* layer = NULL;
	SNEW(layer, Sigmoid2Layer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void Sigmoid2Layer<Dtype>::destroyLayer(void* instancePtr) {
    Sigmoid2Layer<Dtype>* layer = (Sigmoid2Layer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void Sigmoid2Layer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    Sigmoid2Layer<Dtype>* layer = (Sigmoid2Layer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool Sigmoid2Layer<Dtype>::allocLayerTensors(void* instancePtr) {
    Sigmoid2Layer<Dtype>* layer = (Sigmoid2Layer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void Sigmoid2Layer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    Sigmoid2Layer<Dtype>* layer = (Sigmoid2Layer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void Sigmoid2Layer<Dtype>::backwardTensor(void* instancePtr) {
    Sigmoid2Layer<Dtype>* layer = (Sigmoid2Layer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void Sigmoid2Layer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool Sigmoid2Layer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t Sigmoid2Layer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class Sigmoid2Layer<float>;
