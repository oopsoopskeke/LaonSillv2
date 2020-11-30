/*
 * SyncMem.cpp
 *
 *  Created on: 2016. 8. 24.
 *      Author: jhkim
 */

#include <math_functions.hpp>
#include <cfloat>

#include "SyncMem.h"

//#define SYNCMEM_LOG

#define MEM_MAX (FLT_MAX / 10)

template <typename Dtype>
__global__ void BoundMem(Dtype* mem, const Dtype bound, uint32_t* updateCount,
    const unsigned int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	if(mem[idx] > bound) {
		mem[idx] = bound;
		*updateCount++;
	} else if(mem[idx] < -bound) {
		mem[idx] = -bound;
		*updateCount++;
	}
}

template <>
uint32_t SyncMem<float>::bound_mem() {
	float* d_mem = mutable_device_mem();
	const float bound = MEM_MAX;

	_h_int = 0;
	checkCudaErrors(cudaMemcpy(_d_int, &_h_int, sizeof(uint32_t), cudaMemcpyHostToDevice));
	BoundMem<<<SOOOA_GET_BLOCKS((unsigned int)_size), SOOOA_CUDA_NUM_THREADS>>>(
			d_mem, bound, _d_int, (unsigned int)_size);
	CUDA_POST_KERNEL_CHECK;
	checkCudaErrors(cudaMemcpyAsync(&_h_int, _d_int, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	return _h_int;
}

template <>
uint32_t SyncMem<int>::bound_mem() {
	assert(false &&
			"SyncMem<int>::bound_mem() is not supported ... ");
}
