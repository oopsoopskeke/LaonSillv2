/*
 * SoftmaxWithLossLayer.cpp
 *
 *  Created on: Dec 3, 2016
 *      Author: jkim
 */

#include <cfloat>
#include <vector>

#include "SoftmaxWithLossLayer.h"
#include "MathFunctions.h"


#define SOFTMAXWITHLOSSLAYER_LOG 0

using namespace std;






//#define SLPROP(layer, bar)                                                                   \
//    (((_##layer##PropLayer*)(WorkContext::curLayerProp->prop))->_##var##_)







template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(
		const int nthreads,
        const Dtype* prob_data,
        const Dtype* label,
        Dtype* loss,
        const int num,
        const int dim,
        const int spatial_dim,
        const bool has_ignore_label_,
        const int ignore_label_,
        Dtype* counts) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		//printf("SoftmaxLossForwardGPU index: %d\n", index);

		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(label[n * spatial_dim + s]);
		if (has_ignore_label_ && label_value == ignore_label_) {
			loss[index] = 0;
			counts[index] = 0;
		} else {
			loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
					Dtype(FLT_MIN)));
			counts[index] = 1;
		}
	}
}


template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::feedforward() {
	reshape();

	//InnerLayerFunc::runForward(0, -1);
	this->softmaxLayer->feedforward();

	//this->_printOn();
	//this->prob.print_data({}, false);
	//this->_printOff();

	const Dtype* probData = this->prob.device_data();
	const Dtype* label = this->_inputData[1]->device_data();
	const int dim = this->prob.getCount() / this->outerNum;
	const int nthreads = this->outerNum * this->innerNum;
	// Since this memory is not used for anything until it is overwritten
	// on the backward pass, we use it here to avoid having to allocate new GPU
	// memory to accumelate intermediate results in the kernel.
	Dtype* lossData = this->_inputData[0]->mutable_device_grad();
	// Similary, this memroy is never used elsewhere, and thus we can use it
	// to avoid having to allocated additional GPU memory.
	Dtype* counts = this->prob.mutable_device_grad();

	const bool hasIgnoreLabel = GET_PROP(prop, Loss, hasIgnoreLabel);
	const int ignoreLabel = GET_PROP(prop, Loss, ignoreLabel);
	SoftmaxLossForwardGPU<Dtype><<<SOOOA_GET_BLOCKS(nthreads), SOOOA_CUDA_NUM_THREADS>>>(
			nthreads, probData, label, lossData, this->outerNum, dim,
			this->innerNum, hasIgnoreLabel, ignoreLabel, counts);
	CUDA_POST_KERNEL_CHECK;

	NormalizationMode normalizationMode = GET_PROP(prop, Loss, normalization);
	Dtype loss;
	soooa_gpu_asum(nthreads, lossData, &loss);
	Dtype validCount = -1;
	// Only launch another CUDA kernel if we actually need the count of valid
	// outputs.
	if (normalizationMode == NormalizationMode::Valid && hasIgnoreLabel)
		soooa_gpu_asum(nthreads, counts, &validCount);

	// xxx normalizer test -> restored
	//const float lossWeight = GET_PROP(prop, Loss, lossWeight);
	this->_outputData[0]->mutable_host_data()[0] = loss
			//* type(lossWeight) /
			/ LossLayer<Dtype>::getNormalizer(normalizationMode,
					this->outerNum, this->innerNum, validCount);

}


template void SoftmaxWithLossLayer<float>::feedforward();



template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(
		const int nthreads,
		const Dtype* top,
        const Dtype* label,
        Dtype* bottom_diff,
        const int num,
        const int dim,
        const int spatial_dim,
        const bool has_ignore_label_,
        const int ignore_label_,
        Dtype* counts) {
	const int channels = dim / spatial_dim;
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(label[n * spatial_dim + s]);

		if (has_ignore_label_ && label_value == ignore_label_) {
			for (int c = 0; c < channels; ++c) {
				bottom_diff[n * dim + c * spatial_dim + s] = 0;
			}
			counts[index] = 0;
		} else {
			bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
			counts[index] = 1;
		}
	}
}


template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::backpropagation() {
	SASSERT(!GET_PROP(prop, SoftmaxWithLoss, propDown)[1],
			"SoftmaxLayer cannot backpropagate to label inputs ... ");

	if (GET_PROP(prop, SoftmaxWithLoss, propDown)[0]) {
		Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
		const Dtype* probData = this->prob.device_data();
		//soooa_gpu_memcpy(this->prob.getCount() * sizeof(Dtype), probData, inputGrad);
		soooa_copy(this->prob.getCount(), probData, inputGrad);

		const Dtype* outputData = this->_outputData[0]->device_data();
		const Dtype* label = this->_inputData[1]->device_data();
		const int dim = this->prob.getCount() / this->outerNum;
		const int nthreads = this->outerNum * this->innerNum;
		// Since this memroy is never used for anything else,
		// we use to avoid allocating new GPU memroy.
		Dtype* counts = this->prob.mutable_device_grad();

		SoftmaxLossBackwardGPU<Dtype><<<SOOOA_GET_BLOCKS(nthreads),
            SOOOA_CUDA_NUM_THREADS>>>(nthreads, outputData, label, inputGrad,
            this->outerNum, dim, this->innerNum, GET_PROP(prop, Loss, hasIgnoreLabel),
            GET_PROP(prop, Loss, ignoreLabel), counts);
		CUDA_POST_KERNEL_CHECK;

		Dtype validCount = -1;
		// Only launch another CUDA kernel if we actually need the count of valid
		// outputs.
		NormalizationMode normalizationMode = GET_PROP(prop, Loss, normalization);
		if (normalizationMode == NormalizationMode::Valid &&
				GET_PROP(prop, Loss, hasIgnoreLabel))
			soooa_gpu_asum(nthreads, counts, &validCount);

		const Dtype lossWeight = GET_PROP(prop, Loss, lossWeight) /
				LossLayer<Dtype>::getNormalizer(normalizationMode, this->outerNum,
						this->innerNum, validCount);
		soooa_gpu_scal(this->prob.getCount(), lossWeight, inputGrad);
	}
}


template void SoftmaxWithLossLayer<float>::backpropagation();





