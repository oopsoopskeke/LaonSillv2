/**
 * @file MaxPooling.h
 * @date 2016/5/16
 * @author jhkim
 * @brief
 * @details
 */

#ifndef POOLING_MAXPOOLING_H_
#define POOLING_MAXPOOLING_H_

#include <limits>

#include "common.h"
#include "Pooling.h"

/**
 * @brief 최대 풀링을 구현한 Pooling 클래스
 * @details 항상 커널사이즈 절반 크기의 padding이 적용된다.
 * @todo padding 관련 파라미터를 추가하고 파라미터에 따라 padding이 적용되도록 수정한다.
 */
template <typename Dtype>
class MaxPooling : public Pooling<Dtype> {
#ifndef GPU_MODE
public:
	MaxPooling() {
		this->type = PoolingType::Max;
	}
	virtual ~MaxPooling() {}

	void forward(const pool_dim &pool_d, const rcube &input, ucube &pool_map, rcube &output) {
		UINT i, j, k, l, m;

		int left_pad = (pool_d.rows-1)/2;
		int top_pad = (pool_d.cols-1)/2;
		int in_input_row_idx;
		int in_input_col_idx;
		int pool_max_idx;
		double max;

		pool_map.zeros();

		// input image에 대해
		for(i = 0; i < input.n_slices; i++) {
			//for(j = 0; j < input.n_rows; j+=pool_d.stride) {
			//	for(k = 0; k < input.n_cols; k+=pool_d.stride) {
			for(j = 0; j/pool_d.stride < output.n_rows; j+=pool_d.stride) {
				for(k = 0; k/pool_d.stride < output.n_cols; k+=pool_d.stride) {
					max = numeric_limits<double>::min();
					pool_max_idx = 4;

					// input image의 i, j를 center로 pool 영역만큼 최대값과 위치 찾기
					for(l = 0; l < pool_d.rows; l++) {
						for(m = 0; m < pool_d.cols; m++) {
							in_input_row_idx = j-left_pad+l;
							in_input_col_idx = k-top_pad+m;

							if ((in_input_row_idx >= 0 && 
                               (UINT)in_input_row_idx < input.n_rows) &&
                               (in_input_col_idx >= 0 && 
                                (UINT)in_input_col_idx < input.n_cols)) {

								if (C_MEM(input, in_input_row_idx, in_input_col_idx, i) > 
                                    max) {
									max = C_MEM(input, in_input_row_idx, in_input_col_idx, i);
									pool_max_idx = l*pool_d.cols + m;
								}
							}
						}
					}

					C_MEMPTR(output, j/pool_d.stride, k/pool_d.stride, i) = max;
					C_MEMPTR(pool_map, j/pool_d.stride, k/pool_d.stride, i) = pool_max_idx;
				}
			}
		}

		Util::printCube(input, "input:");
		Util::printUCube(pool_map, "pool_map:");
		Util::printCube(output, "output:");
	}

	void backward(const pool_dim &pool_d, const rcube &input, ucube &pool_map,
        rcube &output) {

		UINT i, j, k;
		int pool_max_idx;
		int left_pad = (pool_d.rows-1)/2;
		int top_pad = (pool_d.cols-1)/2;

		output.zeros();

		for(i = 0; i < input.n_slices; i++) {
			for(j = 0; j < input.n_rows; j++) {
				for(k = 0; k < input.n_cols; k++) {
					pool_max_idx = C_MEM(pool_map, j, k, i);
					C_MEMPTR(output, 
                             (int)(j*pool_d.stride + pool_max_idx/pool_d.cols-left_pad),
                             (int)(k*pool_d.stride + pool_max_idx%pool_d.cols-top_pad), i)
				        = C_MEM(output, 
                                (int)(j*pool_d.stride + pool_max_idx/pool_d.cols-left_pad),
                                (int)(k*pool_d.stride + pool_max_idx%pool_d.cols-top_pad), i)
                                 + C_MEM(input, j, k, i);
				}
			}
		}

		Util::printCube(input, "input:");
		Util::printUCube(pool_map, "pool_map:");
		Util::printCube(output, "output:");
	}
#else
public:
	/**
	 * @details MaxPooling 생성자
	 * @param pool_d 풀링 연산 관련 파라미터 구조체
	 */
	MaxPooling(pool_dim pool_d) {
		this->type = PoolingType::Max;

		checkCUDNN(cudnnCreatePoolingDescriptor(&this->poolDesc));

		//int pad = (pool_d.rows-1)/2;
		checkCUDNN(cudnnSetPooling2dDescriptor(this->poolDesc,
				CUDNN_POOLING_MAX,
				CUDNN_PROPAGATE_NAN,
				pool_d.rows, pool_d.cols,
				pool_d.pad, pool_d.pad,
				pool_d.stride, pool_d.stride));
	}
	/**
	 * @details MaxPooling 소멸자
	 */
	virtual ~MaxPooling() {
		checkCUDNN(cudnnDestroyPoolingDescriptor(this->poolDesc));
	}

	void forward(const cudnnTensorDescriptor_t xDesc, const Dtype* x,
			const cudnnTensorDescriptor_t yDesc, Dtype* y) {

		checkCUDNN(cudnnPoolingForward(Cuda::cudnnHandle, this->poolDesc,
				&Cuda::alpha, xDesc, x, &Cuda::beta, yDesc, y));
	}

	void backward(const cudnnTensorDescriptor_t yDesc, const Dtype* y, const Dtype* dy,
			const cudnnTensorDescriptor_t xDesc, const Dtype* x, Dtype* dx) {
		checkCUDNN(cudnnPoolingBackward(Cuda::cudnnHandle, this->poolDesc,
				&Cuda::alpha, yDesc, y, yDesc, dy, xDesc, x,
				&Cuda::beta, xDesc, dx));
	}

#endif

};

template class MaxPooling<float>;

#endif /* POOLING_MAXPOOLING_H_ */
