/**
 * @file HyperTangentLayer_device.cu
 * @date 2017-03-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "HyperTangentLayer.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "ColdLog.h"
#include "Perf.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

#define HYPERTANGENT_LOG   0

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

template <typename Dtype>
__global__ void HyperTangentForward(const Dtype *input, int size, Dtype *output)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	output[idx] = tanhf(input[idx]);
}

template <typename Dtype>
__global__ void HyperTangentBackward(const Dtype *outputGrad, const Dtype *output, int size,
    Dtype *inputGrad)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	inputGrad[idx] = outputGrad[idx] * (1.0 - output[idx] * output[idx]);
}

template <typename Dtype>
HyperTangentLayer<Dtype>::HyperTangentLayer() {
	this->type = Layer<Dtype>::HyperTangent;
}

template <typename Dtype>
void HyperTangentLayer<Dtype>::feedforward() {
    reshape();
    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* outputData = this->_outputData[0]->mutable_device_data();
    int size = this->_inputData[0]->getCountByAxis(0);

    HyperTangentForward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, size, outputData);
}

template <typename Dtype>
void HyperTangentLayer<Dtype>::backpropagation() {
	const Dtype* outputGrads = this->_outputData[0]->device_grad();
    const Dtype* outputData = this->_outputData[0]->device_data();
	Dtype* inputGrads = this->_inputData[0]->mutable_device_grad();
    int size = this->_inputData[0]->getCountByAxis(0);

    HyperTangentBackward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        outputGrads, outputData, size, inputGrads);
}

template <typename Dtype>
void HyperTangentLayer<Dtype>::reshape() {
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

	STDOUT_COND_LOG(HYPERTANGENT_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);
	STDOUT_COND_LOG(HYPERTANGENT_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* HyperTangentLayer<Dtype>::initLayer() {
	HyperTangentLayer* layer = NULL;
	SNEW(layer, HyperTangentLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void HyperTangentLayer<Dtype>::destroyLayer(void* instancePtr) {
    HyperTangentLayer<Dtype>* layer = (HyperTangentLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void HyperTangentLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    HyperTangentLayer<Dtype>* layer = (HyperTangentLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool HyperTangentLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    HyperTangentLayer<Dtype>* layer = (HyperTangentLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void HyperTangentLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    HyperTangentLayer<Dtype>* layer = (HyperTangentLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void HyperTangentLayer<Dtype>::backwardTensor(void* instancePtr) {
    HyperTangentLayer<Dtype>* layer = (HyperTangentLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void HyperTangentLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool HyperTangentLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t HyperTangentLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class HyperTangentLayer<float>;
