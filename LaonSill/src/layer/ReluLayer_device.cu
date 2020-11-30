/**
 * @file ReluLayer_device.cu
 * @date 2017-02-15
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "ReluLayer.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "ColdLog.h"
#include "Perf.h"
#include "PropMgmt.h"

using namespace std;


///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels
//

template <typename Dtype>
__global__ void ApplyLeakyForward(const Dtype* input, Dtype* output, int size, Dtype leaky)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    if (input[idx] < 0)
        output[idx] = leaky * input[idx];
    else 
        output[idx] = input[idx];
}

template <typename Dtype>
__global__ void ApplyLeakyBackward(const Dtype* input, const Dtype* outputGrad,
    Dtype* inputGrad, int size, const double leaky)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    if (input[idx] < 0)
        inputGrad[idx] = leaky * outputGrad[idx];
    else
        inputGrad[idx] = outputGrad[idx];
}

template <typename Dtype>
void ReluLayer<Dtype>::applyLeakyForward() {
	int size = this->_outputData[0]->getCountByAxis(0);
    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* outputData = this->_outputData[0]->mutable_device_data();

    ApplyLeakyForward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, outputData, size, (Dtype)SLPROP(Relu, leaky));
}

template <typename Dtype>
void ReluLayer<Dtype>::applyLeakyBackward() {
	int size = this->_outputData[0]->getCountByAxis(0);
    const Dtype* inputData = this->_inputData[0]->device_data();
    const Dtype* outputGrad = this->_outputData[0]->device_grad();
    Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();

    const double leaky = SLPROP(Relu, leaky);
    ApplyLeakyBackward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, outputGrad, inputGrad, size, leaky);
}

template void ReluLayer<float>::applyLeakyForward();
template void ReluLayer<float>::applyLeakyBackward();

