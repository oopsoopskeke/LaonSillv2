/**
 * @file	Cuda.h
 * @date	2016/6/16
 * @author	jhkim
 * @brief	Cuda관련 매크로 및 상수, 클래스 선언.
 * @details
 */

#ifndef CUDA_H_
#define CUDA_H_

#include <sstream>
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>

#include "common.h"
#include "SysLog.h"
#include "StdOutLog.h"

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN


#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      	cudaDeviceReset();											   \
      	STDOUT_LOG("[ERROR] %s", _error.str().c_str());                                      \
      	SASSERT(false, _error.str().c_str());	  	  	  			   \
    }                                                                  \
} while(0)

#define checkCUBLAS(status) do {                                   							 \
    std::stringstream _error;                                                                \
    if (status != 0) {                                                                       \
    	_error << "CUBLAS failure: ";                                                        \
    	switch(status) {                                                                     \
    		case CUBLAS_STATUS_SUCCESS: _error << "CUBLAS_STATUS_SUCCESS";                   \
    		case CUBLAS_STATUS_NOT_INITIALIZED: _error << "CUBLAS_STATUS_NOT_INITIALIZED";   \
    		case CUBLAS_STATUS_ALLOC_FAILED: _error << "CUBLAS_STATUS_ALLOC_FAILED";         \
    		case CUBLAS_STATUS_INVALID_VALUE: _error << "CUBLAS_STATUS_INVALID_VALUE";       \
    		case CUBLAS_STATUS_ARCH_MISMATCH: _error << "CUBLAS_STATUS_ARCH_MISMATCH";       \
    		case CUBLAS_STATUS_MAPPING_ERROR: _error << "CUBLAS_STATUS_MAPPING_ERROR";       \
    		case CUBLAS_STATUS_EXECUTION_FAILED: _error << "CUBLAS_STATUS_EXECUTION_FAILED"; \
    		case CUBLAS_STATUS_INTERNAL_ERROR: _error << "CUBLAS_STATUS_INTERNAL_ERROR";     \
    		case CUBLAS_STATUS_NOT_SUPPORTED: _error << "CUBLAS_STATUS_NOT_SUPPORTED";       \
    		case CUBLAS_STATUS_LICENSE_ERROR: _error << "CUBLAS_STATUS_LICENSE_ERROR";       \
    		default: _error << "UNKNOWN_CUBLAS_STATUS";                                      \
    	}                                                                                    \
		cudaDeviceReset();                                                 					 \
		STDOUT_LOG("[ERROR] %s", _error.str().c_str());                                      \
        SASSERT(false, _error.str().c_str());                                                \
    }                                                                                        \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << cudaGetErrorString(status);		   \
      cudaDeviceReset();                                                 					 \
      STDOUT_LOG("[ERROR] %s", _error.str().c_str());                                        \
      SASSERT(false, _error.str().c_str());                                                  \
    }                                                                                        \
} while(0)




// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) 										\
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 			\
		i < (n); 													\
		i += blockDim.x * gridDim.x)


// CUDA: use 512 threads per block
const int SOOOA_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int SOOOA_GET_BLOCKS(const int N) {
  return (N + SOOOA_CUDA_NUM_THREADS - 1) / SOOOA_CUDA_NUM_THREADS;
}

#define CUDA_KERNEL_LOOP(i, n) 										\
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 			\
		i < (n); 													\
		i += blockDim.x * gridDim.x)


// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK checkCudaErrors(cudaPeekAtLastError())

// Block width for CUDA kernels
#define BW 128

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
	return (nominator + denominator - 1) / denominator;
}

/**
 * @brief Cuda 라이브러리 사용에 필요한 핸들을 생성하여 전역으로 관리하는 클래스.
 * @detail Cuda 장치를 설정하고 cudnn, cublas 관련 핸들을 전역으로 생성, 삭제, 리프레시
 *        하는 역할을 함.
 */
class Cuda {
public:
	Cuda();
	virtual ~Cuda();

	/**
	 * @details 지정된 개수 만큼의 Cuda 장치를 설정하고 cudnn, cublas 핸들을 생성.
	 * @param usingGPUCount Cuda를 사용할 장치 개수. 0을 입력하면 머신에 존재하는 모든
     *                      Cuda장치를 사용함.
	 */
	static void create(int usingGPUCount);
	/**
	 * @details create()를 통해 생성한 Cuda관련 리소스를 정리.
	 */
	static void destroy();
	/**
	 * @details Cuda를 실행할 장치 재설정.
	 */
	static void refresh();

	static int gpuid;				    ///< Cuda를 사용할 장치의 id
	static int gpuCount;			    ///< Cuda를 사용할 장치 개수.
    static std::vector<int> availableGPU;    ///< 사용가능한 GPU. (peer access가 가능한 GPU)
	static thread_local cudnnHandle_t cudnnHandle;		///< cudnn 라이브러리 컨텍스트 핸들.
	static thread_local cublasHandle_t cublasHandle;	///< cublas 라이브러리 컨텍스트 핸들.

	static const float alpha;
	static const float beta;
	static const float negativeOne;

};

#endif /* CUDA_H_ */
