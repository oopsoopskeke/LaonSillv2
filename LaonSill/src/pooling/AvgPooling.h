/**
 * @file AvgPooling.h
 * @date 2016/5/24
 * @author jhkim
 * @brief input의 좌상단을 pool 영역의 좌상단에 맞춰 pooling
 *        overlap되지 않게 pooling
 *        input.n_rows / pool_d.rows 만큼 down sample
 * @details
 */

#ifndef POOLING_AVGPOOLING_H_
#define POOLING_AVGPOOLING_H_

#include "common.h"
#include "Pooling.h"


/**
 * @brief 평균 풀링을 구현한 Pooling 클래스
 * @details 항상 padding이 적용되지 않는다.
 * @todo padding 관련 파라미터를 추가하고 파라미터에 따라 padding이 적용되도록 수정한다.
 */
template <typename Dtype>
class AvgPooling : public Pooling<Dtype> {
#ifndef GPU_MODE
public:
	AvgPooling() {
		this->type = PoolingType::Avg;
	}
	virtual ~AvgPooling() {}

	void forward(const pool_dim &pool_d, const rcube &input, ucube &pool_map, rcube &output) {
		UINT i, j, k, l, m;

		int left_pad = (pool_d.rows-1)/2;
		int top_pad = (pool_d.cols-1)/2;
		int in_input_row_idx;
		int in_input_col_idx;
		int num_pool_elem = pool_d.rows*pool_d.cols;
		double sum;

		Util::printCube(input, "input:");

		// GoogLeNet에서 Average Pooling의 경우 image의 첫 위치가 아닌 left, top pad를 
        // 포함하여 offset된 위치에서 시작되도록 계산하는 게 맞아 보여 일단 그렇게 작성
		// (stride도 left, top pad를 따르나 일단 사용자로부터 입력받도록 둠
		// 아주 일반적이지 않을 수 있음, 추후의 케이스에 수정이 필요할 수 있음.

		// input image에 대해
		for(i = 0; i < input.n_slices; i++) {
			for(j = left_pad; j < input.n_rows; j+=pool_d.stride) {
				for(k = top_pad; k < input.n_cols; k+=pool_d.stride) {
					sum = 0.0;

					// input image의 i, j를 center로 pool 영역만큼 최대값과 위치 찾기
					for(l = 0; l < pool_d.rows; l++) {
						for(m = 0; m < pool_d.cols; m++) {
							in_input_row_idx = j-left_pad+l;
							in_input_col_idx = k-top_pad+m;

							if ((in_input_row_idx >= 0 && 
                               (UINT)in_input_row_idx < input.n_rows) &&
                               (in_input_col_idx >= 0 && 
                                (UINT)in_input_col_idx < input.n_cols)) {
								//sum += input.slice(i)(in_input_row_idx, in_input_col_idx);
								sum += C_MEM(input, in_input_row_idx, in_input_col_idx, i);
							}
						}
					}
					sum /= num_pool_elem;
					//output.slice(i)(j/pool_d.rows, k/pool_d.cols) = sum;
					C_MEMPTR(output, j/pool_d.stride, k/pool_d.stride, i) = sum;
				}
			}
		}
	}

	void backward(const pool_dim &pool_d, const rcube &input, ucube &pool_map, 
        rcube &output) {
		UINT i, j, k, l, m;
		int in_output_base_row_idx, in_output_base_col_idx;
		double num_pool_elem_factor = 1.0/(pool_d.rows*pool_d.cols);
		int row, col;

		output.set_size(input.n_rows*pool_d.stride+(pool_d.rows-1)/2,
                        input.n_cols*pool_d.stride+(pool_d.cols-1)/2, input.n_slices);
		output.zeros();

		Util::printCube(input, "input:");

		// j*stride+[0:pool_d.rows-1]
		for(i = 0; i < input.n_slices; i++) {
			for(j = 0; j < input.n_rows; j++) {
				for(k = 0; k < input.n_cols; k++) {
					in_output_base_row_idx = j*pool_d.stride;
					in_output_base_col_idx = k*pool_d.stride;

					for(l = 0; l < pool_d.rows; l++) {
						for(m = 0; m < pool_d.cols; m++) {
							row = in_output_base_row_idx+l;
							col = in_output_base_col_idx+m;
							C_MEMPTR(output, row, col, i) = 
                                C_MEM(output, row, col, i) + C_MEM(input, j, k, i);
						}
					}
				}
			}
		}
		output = num_pool_elem_factor*output;
	}
#else
public:
	/**
	 * @details AvgPooling 생성자
	 * @param pool_d 풀링 연산 관련 파라미터 구조체
	 */
	AvgPooling(pool_dim pool_d) {
		this->type = PoolingType::Avg;

		checkCUDNN(cudnnCreatePoolingDescriptor(&this->poolDesc));

		//int pad = (pool_d.rows-1)/2;
		//int pad = 0;
		checkCUDNN(cudnnSetPooling2dDescriptor(this->poolDesc,
				CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
				CUDNN_PROPAGATE_NAN,
				pool_d.rows, pool_d.cols,
				pool_d.pad, pool_d.pad,
				pool_d.stride, pool_d.stride));
	}
	/**
	 * @details AvgPooling 소멸자
	 */
	virtual ~AvgPooling() {
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

template class AvgPooling<float>;

#endif /* POOLING_AVGPOOLING_H_ */
