/*
 * MathFunctions.h
 *
 *  Created on: Nov 25, 2016
 *      Author: jkim
 */

#ifndef SOOOA_MATHFUNCTIONS_H_
#define SOOOA_MATHFUNCTIONS_H_

#include "common.h"
#include "Blas.h"

template <typename Dtype>
void soooa_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void soooa_set(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void soooa_gpu_set(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void soooa_gpu_axpy(const uint32_t N, const Dtype alpha, const Dtype* X,
		Dtype* Y);

template <typename Dtype>
void soooa_gpu_scal(const uint32_t N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void soooa_gpu_sub(const uint32_t N, const Dtype* a, const Dtype* b, Dtype* y);

/**
 * element-wise multiplication
 */
template <typename Dtype>
void soooa_gpu_mul(const uint32_t N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void soooa_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void soooa_gpu_dot(const uint32_t n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void soooa_gpu_axpby(const uint32_t N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
void soooa_gpu_asum(const int n, const Dtype* x, Dtype* y);





void soooa_gpu_memcpy(const size_t N, const void *X, void *Y);

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void soooa_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void soooa_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);


template <typename Dtype>
void soooa_gpu_scale(const int n, const Dtype alpha, const Dtype* x, Dtype* y);




template <typename Dtype>
void soooa_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);






template <typename Dtype>
void soooa_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void soooa_gpu_add(const int n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void soooa_gpu_square(const int N, const Dtype* a, Dtype* y);




template <typename Dtype, bool packed>
void soooa_gpu_interp2(const int channels, const Dtype* data1, const int x1, const int y1,
		const int height1, const int width1, const int Height1, const int Width1,
		Dtype* data2, const int x2, const int y2, const int height2, const int width2,
		const int Height2, const int Width2);

template <typename Dtype, bool packed>
void soooa_gpu_interp2_backward(const int channels, Dtype* data1, const int x1, const int y1,
		const int height1, const int width1, const int Height1, const int Width1,
		const Dtype* data2, const int x2, const int y2, const int height2, const int width2,
		const int Height2, const int Width2);






template <typename Dtype>
void soooa_gpu_sqrt(const int N, const Dtype* a, Dtype* y);


















unsigned int soooa_rng_rand();

template <typename Dtype>
Dtype soooa_nextafter(const Dtype b);


template <typename Dtype>
void soooa_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void soooa_rng_gaussian(const int n, float a, float sigma, Dtype* r);








#endif /* SOOOA_MATHFUNCTIONS_H_ */
