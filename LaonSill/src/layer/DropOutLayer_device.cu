/**
 * @file DropOutLayer_device.cu
 * @date 2017-04-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "DropOutLayer.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "ColdLog.h"
#include "Perf.h"
#include "PropMgmt.h"

using namespace std;

template<typename Dtype>
__global__ void DropOut(int size, const Dtype* in, Dtype* out, const Dtype* mask, Dtype scale) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    if (mask[idx] > 0.9999) {
        out[idx] = in[idx] * scale;
    } else {
        out[idx] = in[idx];
    }
}

template<typename Dtype>
void DropOutLayer<Dtype>::doDropOutBackward() {
    const Dtype* outputGrad = this->_outputData[0]->device_grad();
    Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
    const Dtype* maskDev = this->mask->device_mem();
    int size = this->_outputData[0]->getCount();

    DropOut<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(size, outputGrad, 
        inputGrad,  maskDev, (Dtype)SLPROP(DropOut, scale));
}

template<typename Dtype>
void DropOutLayer<Dtype>::doDropOutForward() {
    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* outputData = this->_outputData[0]->mutable_device_data();
    Dtype* mask = this->mask->mutable_host_mem();
    int size = this->_inputData[0]->getCount();

    for (int i = 0; i < size; i++) {
        float prob = rand() / (RAND_MAX + 1.0);

        if (prob < SLPROP(DropOut, probability))
            mask[i] = 0.0;
        else
            mask[i] = 1.0;
    }

    const Dtype* maskDev = this->mask->device_mem();
    DropOut<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(size, inputData, outputData,
        maskDev, (Dtype)SLPROP(DropOut, scale));
}

template void DropOutLayer<float>::doDropOutForward();
template void DropOutLayer<float>::doDropOutBackward();
