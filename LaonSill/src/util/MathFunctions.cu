#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include "MathFunctions.h"
#include "SysLog.h"
#include "Cuda.h"
#include "RNG.h"

template <typename Dtype>
void soooa_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void soooa_set<int>(const int N, const int alpha, int* Y);
template void soooa_set<float>(const int N, const float alpha, float* Y);
template void soooa_set<double>(const int N, const double alpha, double* Y);


template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void soooa_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    checkCudaErrors(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void soooa_gpu_set<int>(const int N, const int alpha, int* Y);
template void soooa_gpu_set<unsigned int>(const int N, const unsigned int alpha, unsigned int* Y);
template void soooa_gpu_set<float>(const int N, const float alpha, float* Y);
template void soooa_gpu_set<double>(const int N, const double alpha, double* Y);





template <typename Dtype>
void soooa_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    //if (Caffe::mode() == Caffe::GPU) {
//#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      checkCudaErrors(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
//#else
//      NO_GPU;
//#endif
    //} else {
     // memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    //}
  }
}

template void soooa_copy<int>(const int N, const int* X, int* Y);
template void soooa_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void soooa_copy<float>(const int N, const float* X, float* Y);
template void soooa_copy<double>(const int N, const double* X, double* Y);




template <typename Dtype>
__global__ void sub_kernel(const uint32_t n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void soooa_gpu_sub<float>(const uint32_t N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void soooa_gpu_sub<double>(const uint32_t N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}



template <typename Dtype>
__global__ void mul_kernel(const uint32_t n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void soooa_gpu_mul<float>(const uint32_t N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void soooa_gpu_mul<double>(const uint32_t N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}



template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = a[index] / b[index];
	}
}

template <>
void soooa_gpu_div<float>(const int N, const float* a, const float* b, float* y) {
	div_kernel<float><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(N, a, b, y);
}


template <>
void soooa_gpu_dot<float>(const uint32_t n, const float* x, const float* y,
    float* out) {
	checkCUBLAS(cublasSdot(Cuda::cublasHandle, n, x, 1, y, 1, out));
}

template <>
void soooa_gpu_dot<double>(const uint32_t n, const double* x, const double* y,
    double * out) {
	checkCUBLAS(cublasDdot(Cuda::cublasHandle, n, x, 1, y, 1, out));
}

template <>
void soooa_gpu_scal<float>(const uint32_t N, const float alpha, float *X) {
	checkCUBLAS(cublasSscal(Cuda::cublasHandle, N, &alpha, X, 1));
}

template <>
void soooa_gpu_scal<double>(const uint32_t N, const double alpha, double *X) {
	checkCUBLAS(cublasDscal(Cuda::cublasHandle, N, &alpha, X, 1));
}

template <>
void soooa_gpu_axpy<float>(const uint32_t N, const float alpha, const float* X,
    float* Y) {
	checkCUBLAS(cublasSaxpy(Cuda::cublasHandle, N, &alpha, X, 1, Y, 1));
}

template <>
void soooa_gpu_axpy<double>(const uint32_t N, const double alpha, const double* X,
    double* Y) {
	checkCUBLAS(cublasDaxpy(Cuda::cublasHandle, N, &alpha, X, 1, Y, 1));
}


template <>
void soooa_gpu_axpby<float>(const uint32_t N, const float alpha, const float* X,
    const float beta, float* Y) {
	soooa_gpu_scal<float>(N, beta, Y);
	soooa_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void soooa_gpu_axpby<double>(const uint32_t N, const double alpha, const double* X,
    const double beta, double* Y) {
	soooa_gpu_scal<double>(N, beta, Y);
	soooa_gpu_axpy<double>(N, alpha, X, Y);
}


template <>
void soooa_gpu_asum<float>(const int n, const float* x, float* y) {
	checkCUBLAS(cublasSasum(Cuda::cublasHandle, n, x, 1, y));
}

template <>
void soooa_gpu_asum<double>(const int n, const double* x, double* y) {
	checkCUBLAS(cublasDasum(Cuda::cublasHandle, n, x, 1, y));
}

void soooa_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
	  checkCudaErrors(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template<>
void soooa_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, const float* A, const float* B, const float beta, float* C) {
	// Note that cublas follows fortran order.
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;

	cublasOperation_t cuTransA =
	(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB =
	(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

	checkCUBLAS(cublasSgemm(Cuda::cublasHandle, cuTransB, cuTransA, N, M, K,
			&alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void soooa_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
		const float alpha, const float* A, const float* x, const float beta, float* y) {
	cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	checkCUBLAS(cublasSgemv(Cuda::cublasHandle, cuTransA, N, M, &alpha, A, N, x,
			1, &beta, y, 1));
}


template <>
void soooa_gpu_scale<float>(const int n, const float alpha, const float* x, float* y) {
	checkCUBLAS(cublasScopy(Cuda::cublasHandle, n, x, 1, y, 1));
	checkCUBLAS(cublasSscal(Cuda::cublasHandle, n, &alpha, y, 1));
}





template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] += alpha;
	}
}

template <>
void soooa_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void soooa_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}





template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a, const Dtype alpha, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = pow(a[index], alpha);
	}
}

template <>
void soooa_gpu_powx<float>(const int n, const float* a, const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n, a, alpha, y);
}






template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = a[index] + b[index];
	}
}

template <>
void soooa_gpu_add<float>(const int n, const float* a, const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n, a, b, y);
}


template<typename Dtype>
__global__ void square_kernel(const int n, const Dtype* a, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = a[index] * a[index];
	}
}

template<>
void soooa_gpu_square<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  square_kernel<float><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(N, a, y);

}

template<>
void soooa_gpu_square<double>(const int N, const double* a, double* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	square_kernel<double><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(N, a, y);
}









// Bi-linear interpolation
// IN : [channels height1 width1] cropped from a bigger [Height1 Width1] image
// OUT: [channels height2 width2] cropped from a bigger [Height2 Width2] image
template <typename Dtype, bool packed>
__global__ void soooa_gpu_interp2_kernel(const int n, const float rheight, const float rwidth,
const int channels, const Dtype *data1, const int x1, const int y1, const int height1,
const int width1, const int Height1, const int Width1, Dtype *data2, const int x2,
const int y2, const int height2, const int width2, const int Height2, const int Width2) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		const int w2 = index % width2; // 0:width2-1
		const int h2 = index / width2; // 0:height2-1
		// special case: just copy
		if (height1 == height2 && width1 == width2) {
			const int h1 = h2;
			const int w1 = w2;
			if (packed) {
				const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
				Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
				for (int c = 0; c < channels; ++c) {
					pos2[0] = pos1[0];
					pos1++;
					pos2++;
				}
			}
			else {
				const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
				Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
				for (int c = 0; c < channels; ++c) {
					pos2[0] = pos1[0];
					pos1 += Width1 * Height1;
					pos2 += Width2 * Height2;
				}
			}
			return;
		}
		//
		const float h1r = rheight * h2;
		const int h1 = h1r;
		const int h1p = (h1 < height1 - 1) ? 1 : 0;
		const Dtype h1lambda = h1r - h1;
		const Dtype h0lambda = Dtype(1.) - h1lambda;
		//
		const float w1r = rwidth * w2;
		const int w1 = w1r;
		const int w1p = (w1 < width1 - 1) ? 1 : 0;
		const Dtype w1lambda = w1r - w1;
		const Dtype w0lambda = Dtype(1.) - w1lambda;
		//
		if (packed) {
			const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
			Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
			for (int c = 0; c < channels; ++c) {
				pos2[0] =
						h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[channels * w1p]) +
						h1lambda * (w0lambda * pos1[channels * h1p * Width1] + w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
				pos1++;
				pos2++;
			}
		}
		else {
			const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
			Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
			for (int c = 0; c < channels; ++c) {
				pos2[0] =
						h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) +
						h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
				pos1 += Width1 * Height1;
				pos2 += Width2 * Height2;
			}
		}
	}
}






template <typename Dtype, bool packed>
void soooa_gpu_interp2(const int channels, const Dtype* data1, const int x1, const int y1,
		const int height1, const int width1, const int Height1, const int Width1,
		Dtype* data2, const int x2, const int y2, const int height2, const int width2,
		const int Height2, const int Width2) {
	SASSERT0(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 &&
			x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
	SASSERT0(Width1 >= width1 + x1 && Height1 >= height1 + y1 &&
			Width2 >= width2 + x2 && Height2 >= height2 + y2);
	const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
	const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
	const int numKernels = height2 * width2;

	soooa_gpu_interp2_kernel<Dtype, packed><<<SOOOA_GET_BLOCKS(numKernels), SOOOA_CUDA_NUM_THREADS>>>(
			numKernels, rheight, rwidth, channels,
			data1, x1, y1, height1, width1, Height1, Width1,
			data2, x2, y2, height2, width2, Height2, Width2);
	CUDA_POST_KERNEL_CHECK;
}

template void soooa_gpu_interp2<float, false>(const int channels, const float* data1,
		const int x1, const int y1,
		const int height1, const int width1, const int Height1, const int Width1,
		float* data2, const int x2, const int y2, const int height2, const int width2,
		const int Height2, const int Width2);

template void soooa_gpu_interp2<float, true>(const int channels, const float* data1,
		const int x1, const int y1,
		const int height1, const int width1, const int Height1, const int Width1,
		float* data2, const int x2, const int y2, const int height2, const int width2,
		const int Height2, const int Width2);





// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, bool packed>
__global__ void soooa_gpu_interp2_kernel_backward(const int n, const float rheight,
		const float rwidth, const int channels, Dtype *data1, const int x1, const int y1,
		const int height1, const int width1, const int Height1, const int Width1,
		const Dtype *data2, const int x2, const int y2, const int height2, const int width2,
		const int Height2, const int Width2) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		const int w2 = index % width2; // 0:width2-1
		const int h2 = index / width2; // 0:height2-1
		// special case: just copy
		if (height1 == height2 && width1 == width2) {
			const int h1 = h2;
			const int w1 = w2;
			if (packed) {
				Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
				const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
				for (int c = 0; c < channels; ++c) {
					pos1[0] += pos2[0];
					pos1++;
					pos2++;
				}
			}
			else {
				Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
				const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
				for (int c = 0; c < channels; ++c) {
					pos1[0] += pos2[0];
					pos1 += Width1 * Height1;
					pos2 += Width2 * Height2;
				}
			}
			return;
		}
		//
		const float h1r = rheight * h2;
		const int h1 = h1r;
		const int h1p = (h1 < height1 - 1) ? 1 : 0;
		const Dtype h1lambda = h1r - h1;
		const Dtype h0lambda = Dtype(1.) - h1lambda;
		//
		const float w1r = rwidth * w2;
		const int w1 = w1r;
		const int w1p = (w1 < width1 - 1) ? 1 : 0;
		const Dtype w1lambda = w1r - w1;
		const Dtype w0lambda = Dtype(1.) - w1lambda;
		//
		if (packed) {
			Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
			const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
			for (int c = 0; c < channels; ++c) {
				atomicAdd(&pos1[0], h0lambda * w0lambda * pos2[0]);
				atomicAdd(&pos1[channels * w1p], h0lambda * w1lambda * pos2[0]);
				atomicAdd(&pos1[channels * h1p * Width1], h1lambda * w0lambda * pos2[0]);
				atomicAdd(&pos1[channels * (h1p * Width1 + w1p)], h1lambda * w1lambda * pos2[0]);
				pos1++;
				pos2++;
			}
		}
		else {
			Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
			const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
			for (int c = 0; c < channels; ++c) {
				atomicAdd(&pos1[0], h0lambda * w0lambda * pos2[0]);
				atomicAdd(&pos1[w1p], h0lambda * w1lambda * pos2[0]);
				atomicAdd(&pos1[h1p * Width1], h1lambda * w0lambda * pos2[0]);
				atomicAdd(&pos1[h1p * Width1 + w1p], h1lambda * w1lambda * pos2[0]);
				pos1 += Width1 * Height1;
				pos2 += Width2 * Height2;
			}
		}
	}
}





template <typename Dtype, bool packed>
void soooa_gpu_interp2_backward(const int channels, Dtype* data1, const int x1, const int y1,
		const int height1, const int width1, const int Height1, const int Width1,
		const Dtype* data2, const int x2, const int y2, const int height2, const int width2,
		const int Height2, const int Width2) {
	SASSERT0(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 &&
			x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
	SASSERT0(Width1 >= width1 + x1 && Height1 >= height1 + y1 &&
			Width2 >= width2 + x2 && Height2 >= height2 + y2);
	const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
	const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
	const int numKernels = height2 * width2;

	soooa_gpu_interp2_kernel_backward<Dtype, packed><<<SOOOA_GET_BLOCKS(numKernels), SOOOA_CUDA_NUM_THREADS>>>(
			numKernels, rheight, rwidth, channels,
			data1, x1, y1, height1, width1, Height1, Width1,
			data2, x2, y2, height2, width2, Height2, Width2);
	CUDA_POST_KERNEL_CHECK;
}

template void soooa_gpu_interp2_backward<float, false>(const int channels, float* data1,
		const int x1, const int y1,
		const int height1, const int width1, const int Height1, const int Width1,
		const float* data2, const int x2, const int y2, const int height2, const int width2,
		const int Height2, const int Width2);













template<typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = sqrt(a[index]);
	}
}

template<>
void soooa_gpu_sqrt<float>(const int N, const float* a, float* y) {
	sqrt_kernel<float> <<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(N, a, y);
}

template<>
void soooa_gpu_sqrt<double>(const int N, const double* a, double* y) {
	sqrt_kernel<double> <<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(N, a, y);
}










unsigned int soooa_rng_rand() {
#if 1
	//int result = (*soooa_rng())();
	//std::cout << "soooa_rng_rand(): " << result << std::endl;
	//return result;
	return (*soooa_rng())();;
#else
	return 2;
#endif
}


template <typename Dtype>
Dtype soooa_nextafter(const Dtype b) {
	return boost::math::nextafter<Dtype>(b, std::numeric_limits<Dtype>::max());
}

template float soooa_nextafter(const float b);


template <typename Dtype>
void soooa_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
	SASSERT0(n >= 0);
	SASSERT0(r);
	SASSERT0(a <= b);

#if 1
	// XXX: work around ...
	// release mode에서 soooa_nextafter가 제대로 작동하지 않음,
	// a == 0, b == 0 케이스에서 soooa_nextafter(b) 역시 0이 나옴.
	if (a == b) {
		for (int i = 0; i < n; i++) {
			r[i] = a;
		}
	} else {
		boost::uniform_real<Dtype> random_distribution(a, soooa_nextafter<Dtype>(b));
		boost::variate_generator<rng_t*, boost::uniform_real<Dtype>> variate_generator(
				soooa_rng(), random_distribution);

		for (int i = 0; i < n; i++) {
			r[i] = variate_generator();
		}
	}
	/*
	std::cout << "soooa_rng_uniform(): a=" << a << ", b=" << b << ": ";
	for (int i = 0; i < n; i++) {
		std::cout << r[i] << ",";
	}
	std::cout << std::endl;
	*/

#else
	for (int i = 0; i < n; i++) {
		r[i] = a + (b - a) * 1.f / 3.f;
	}
#endif
}

template void soooa_rng_uniform<float>(const int n, const float a, const float b, float* r);


template <typename Dtype>
void soooa_rng_gaussian(const int n, float a, float sigma, Dtype* r) {
	SASSERT0(n >= 0);
	SASSERT0(r);
	SASSERT0(sigma > 0);

	boost::normal_distribution<float> random_distribution(a, sigma);
	boost::variate_generator<rng_t*, boost::normal_distribution<float> >
		variate_generator(soooa_rng(), random_distribution);
	for (int i = 0; i < n; i++) {
		r[i] = variate_generator();
	}
}

template void soooa_rng_gaussian<float>(const int n, float mu, float sigma, float* r);






















