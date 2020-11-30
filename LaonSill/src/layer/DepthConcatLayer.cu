/*
 * DepthConcatLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "DepthConcatLayer.h"

using namespace std;

//#define DEPTHCONCAT_LOG

template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
		const bool forward, const int num_concats, const int concat_size,
		const int top_concat_axis, const int bottom_concat_axis,
		const int offset_concat_axis, Dtype* out_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int total_concat_size = concat_size * bottom_concat_axis;
		const int concat_num = index / total_concat_size;
		const int concat_index = index % total_concat_size;
		const int top_index = concat_index +
				(concat_num * top_concat_axis + offset_concat_axis) * concat_size;
		if (forward) {
			out_data[top_index] = in_data[index];
		} else {
			out_data[index] = in_data[top_index];
		}
	}
}



template <typename Dtype>
DepthConcatLayer<Dtype>::DepthConcatLayer(Builder* builder)
	: Layer<Dtype>(builder) {
	initialize();
}

template <typename Dtype>
DepthConcatLayer<Dtype>::~DepthConcatLayer() {}

template <typename Dtype>
void DepthConcatLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::DepthConcat;
	this->concatAxis = 1;
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	// 입력 데이터의 shape가 변경된 것이 있는 지 확인
	bool inputShapeReshaped = false;
	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (Layer<Dtype>::_isInputShapeChanged(i)) {
			inputShapeReshaped = true;
			this->_inputShape[i] = this->_inputData[i]->getShape();
		}
	}

	if (!inputShapeReshaped) {
		return;
	}

	uint32_t batches 	= this->_inputShape[0][0];
	uint32_t channels 	= 0;
	uint32_t rows 		= this->_inputShape[0][2];
	uint32_t cols 		= this->_inputShape[0][3];

	for (uint32_t i = 0; i < this->_inputData.size(); i++) {
		channels += this->_inputData[i]->getShape()[this->concatAxis];
	}

	this->_outputData[0]->reshape({batches, channels, rows, cols});

	this->concatInputSize = this->_inputData[0]->getCountByAxis(this->concatAxis + 1);
	this->numConcats = this->_inputData[0]->getCountByAxis(0, this->concatAxis);
}


template <typename Dtype>
void DepthConcatLayer<Dtype>::feedforward() {
	reshape();

	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	int offsetConcatAxis = 0;
	const int outputConcatAxis = this->_outputData[0]->getShape()[this->concatAxis];
	const bool kForward = true;

	for (int i = 0; i < this->_inputData.size(); i++) {
		const Dtype* inputData = this->_inputData[i]->device_data();
		const int inputConcatAxis = this->_inputData[i]->getShape()[this->concatAxis];
		const int inputConcatSize = inputConcatAxis * this->concatInputSize;
		const int nThreads = inputConcatSize * this->numConcats;

		Concat<Dtype><<<SOOOA_GET_BLOCKS(nThreads), SOOOA_CUDA_NUM_THREADS>>>(
				nThreads, inputData, kForward, this->numConcats, this->concatInputSize,
				outputConcatAxis, inputConcatAxis, offsetConcatAxis, outputData);
		offsetConcatAxis += inputConcatAxis;
	}
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::backpropagation() {
	const Dtype* outputGrad = this->_outputData[0]->device_grad();
	int offsetConcatAxis = 0;
	const int outputConcatAxis = this->_outputData[0]->getShape()[this->concatAxis];
	const bool kForward = false;

	for (int i = 0; i < this->_inputData.size(); i++) {
		const int inputConcatAxis = this->_inputData[i]->getShape(this->concatAxis);
		if (this->_propDown[i]) {
			Dtype* inputGrad = this->_inputData[i]->mutable_device_grad();
			const int inputConcatSize = inputConcatAxis * this->concatInputSize;
			const int nThreads = inputConcatSize * this->numConcats;

			Concat<Dtype><<<SOOOA_GET_BLOCKS(nThreads), SOOOA_CUDA_NUM_THREADS>>>(
					nThreads, outputGrad, kForward, this->numConcats, this->concatInputSize,
					outputConcatAxis, inputConcatAxis, offsetConcatAxis, inputGrad);
		}
		offsetConcatAxis += inputConcatAxis;
	}
}



/*
template <typename Dtype>
void DepthConcatLayer<Dtype>::feedforward() {
	reshape();

	uint32_t batchOffset = 0;
	for (uint32_t i = 0; i < this->_inputs.size(); i++) {
		batchOffset += this->_inputData[i]->getCountByAxis(1);
	}

	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();
	const uint32_t batchSize = this->_inputData[0]->getShape()[0];
	uint32_t inBatchOffset = 0;
	for (uint32_t i = 0; i < this->_inputs.size(); i++) {
		const Dtype* d_inputData = this->_inputData[i]->device_data();
		const uint32_t inputCountByChannel = this->_inputData[i]->getCountByAxis(1);
		if (i > 0) {
			inBatchOffset += this->_inputData[i-1]->getCountByAxis(1);
		}
		for (uint32_t j = 0; j < batchSize; j++) {
			checkCudaErrors(cudaMemcpyAsync(
					d_outputData+batchOffset*j+inBatchOffset,
					d_inputData+inputCountByChannel*j,
					inputCountByChannel,
					cudaMemcpyDeviceToDevice));
		}
	}
}


template <typename Dtype>
void DepthConcatLayer<Dtype>::backpropagation() {
	uint32_t batchOffset = 0;
	for (uint32_t i = 0; i < this->_inputs.size(); i++) {
		batchOffset += this->_inputData[i]->getCountByAxis(1);
	}

	const Dtype* d_outputData = this->_outputData[0]->device_data();
	const uint32_t batchSize = this->_inputData[0]->getShape()[0];
	uint32_t inBatchOffset = 0;
	for (uint32_t i = 0; i < this->_inputs.size(); i++) {
		Dtype* d_inputData = this->_inputData[i]->mutable_device_data();
		const uint32_t inputCountByChannel = this->_inputData[i]->getCountByAxis(1);
		if (i > 0) {
			inBatchOffset += this->_inputData[i-1]->getCountByAxis(1);
		}
		for (uint32_t j = 0; j < batchSize; j++) {
			checkCudaErrors(cudaMemcpyAsync(
					d_inputData+inputCountByChannel*j,
					d_outputData+batchOffset*j+inBatchOffset,
					inputCountByChannel,
					cudaMemcpyDeviceToDevice));
		}
	}
}
*/

#ifndef GPU_MODE
template <typename Dtype>
void DepthConcatLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::DepthConcat;

	this->offsetIndex = 0;
	this->input.reset();
	this->delta_input.set_size(size(output));
	this->delta_input.zeros();
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::feedforward(uint32_t idx, const rcube &input,
    const char *end=0) {
	this->input = join_slices(this->input, input);
	Util::printCube(this->input, "input:");

	this->offsets.push_back(this->input.n_slices);

	if(!isLastPrevLayerRequest(idx)) return;

	this->output = this->input;

	propFeedforward(this->output, end);

	// backward pass에서 input을 사용하지 않으므로 여기서 reset할 수 있음
	this->input.reset();
	this->offsetIndex = 0;
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::backpropagation(uint32_t idx, Layer<Dtype>* next_layer) {
	Util::printCube(delta_input, "delta_input:");
	rcube w_next_delta(size(delta_input));
	Util::convertCube(next_layer->getDeltaInput(), delta_input);
	delta_input += w_next_delta;
	// delta_input = join_slices(this->delta_input, next_layer->getDeltaInput());
	if(!isLastNextLayerRequest(idx)) return;

	propBackpropagation();
	this->delta_input.zeros();
}
#endif

template class DepthConcatLayer<float>;
