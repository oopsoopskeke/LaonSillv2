/*
 * BatchNorm3Layer.cpp
 *
 *  Created on: Dec 18, 2017
 *      Author: jkim
 */

#include "BatchNorm3Layer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"
#include "MathFunctions.h"
#include "Updater.h"
#include "StdOutLog.h"

using namespace std;

#if 1
template <typename Dtype>
BatchNorm3Layer<Dtype>::BatchNorm3Layer()
: LearnableLayer<Dtype>(),
  mean("mean"), variance("variance"), temp("temp"), xNorm("xNorm") {
	this->type = Layer<Dtype>::BatchNorm3;

	this->movingAverageFraction = SLPROP(BatchNorm3, movingAverageFraction);
	this->useGlobalStats = SNPROP(status) == NetworkStatus::Test;
	//if (hasUseGlobalStats) {
	this->useGlobalStats = SLPROP(BatchNorm3, useGlobalStats);
	//}
	this->eps = SLPROP(BatchNorm3, eps);

	this->_params.resize(3);
	this->_paramsHistory.resize(3);
	this->_paramsHistory2.resize(3);
	this->_paramsInitialized.resize(3);

	LearnableLayer<Dtype>::initParam(0, "mean");
	LearnableLayer<Dtype>::initParam(1, "variance");
	LearnableLayer<Dtype>::initParam(2, "bias_correction");

	// Mask statistics from optimization by setting local learning rates
	// for mean, variance, and the bias correction to zero.
	this->updatePolicies.resize(3);
	for (int i = 0; i < 3; i++) {
		// set lr and decay = 0 for global mean and variance
		this->updatePolicies[i].lr_mult = 0.f;
	}
}

template <typename Dtype>
BatchNorm3Layer<Dtype>::~BatchNorm3Layer() {
	for (int i = 0; i < this->_params.size(); i++) {
		LearnableLayer<Dtype>::releaseParam(i);
	}
	this->updateParams.clear();
}

template <typename Dtype>
void BatchNorm3Layer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		// XXX
		if (this->_inputData[0]->numAxes() == 1) {
			this->channels = 1;
		} else {
			this->channels = this->_inputData[0]->getShape(1);
		}
		vector<uint32_t> sz = {1, 1, 1, (uint32_t)this->channels};
		LearnableLayer<Dtype>::reshapeParam(0, sz);
		LearnableLayer<Dtype>::reshapeParam(1, sz);
		sz[3] = 1;
		LearnableLayer<Dtype>::reshapeParam(2, sz);

		this->_params[0]->reset_host_data();
		this->_params[1]->reset_host_data();
		this->_params[2]->reset_host_data();
	}
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;
    this->_inputShape[0] = this->_inputData[0]->getShape();

	// XXX
	if (this->_inputData[0]->numAxes() > 1) {
		SASSERT0(this->_inputData[0]->getShape(1) == this->channels);
	}
	this->_outputData[0]->reshapeLike(this->_inputData[0]);

	vector<uint32_t> sz = {1, 1, 1, (uint32_t)this->channels};
	this->mean.reshape(sz);
	this->variance.reshape(sz);
	this->temp.reshapeLike(this->_inputData[0]);
	this->xNorm.reshapeLike(this->_inputData[0]);
	sz[3] = this->_inputData[0]->getShape(0);
	this->batchSumMultiplier.reshape(sz);

	int spatialDim = this->_inputData[0]->getCount() /
			(this->channels * this->_inputData[0]->getShape(0));
	if (this->spatialSumMultiplier.getShape(3) != spatialDim) {
		sz[3] = spatialDim;
		this->spatialSumMultiplier.reshape(sz);
		soooa_set(this->spatialSumMultiplier.getCount(), Dtype(1),
				this->spatialSumMultiplier.mutable_host_data());
	}

	int numbychans = this->channels * this->_inputData[0]->getShape(0);
	if (this->numByChans.getShape(3) != numbychans) {
		sz[3] = numbychans;
		this->numByChans.reshape(sz);
		soooa_set(this->batchSumMultiplier.getCount(), Dtype(1),
				this->batchSumMultiplier.mutable_host_data());
	}
}










template <typename Dtype>
void BatchNorm3Layer<Dtype>::feedforward() {
    reshape();

	this->_printOff();

	this->_inputData[0]->print_data({}, false);

	const Dtype* inputData = this->_inputData[0]->device_data();
	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	int num = this->_inputData[0]->getShape(0);
	int spatialDim = this->_inputData[0]->getCount() / (this->channels * this->_inputData[0]->getShape(0));

	if (this->_inputData[0] != this->_outputData[0]) {
		soooa_copy(this->_inputData[0]->getCount(), inputData, outputData);
	}

	this->_outputData[0]->print_data({}, false);

	if (this->useGlobalStats) {
		// use the stored mean/variance estimates.
		const Dtype scaleFactor =
				this->_params[2]->host_data()[0] == 0 ?
						0 : 1 / this->_params[2]->host_data()[0];
		soooa_gpu_scale(this->variance.getCount(), scaleFactor,
				this->_params[0]->device_data(), this->mean.mutable_device_data());
		soooa_gpu_scale(this->variance.getCount(), scaleFactor,
				this->_params[1]->device_data(), this->variance.mutable_device_data());

		this->mean.print_data({}, false);
		this->variance.print_data({}, false);
	} else {
		// compute mean
		soooa_gpu_gemv<Dtype>(CblasNoTrans, this->channels * num, spatialDim,
				1. / (num * spatialDim), inputData,
				this->spatialSumMultiplier.device_data(), 0.,
				this->numByChans.mutable_device_data());
		soooa_gpu_gemv<Dtype>(CblasTrans, num, this->channels, 1.,
				this->numByChans.device_data(), this->batchSumMultiplier.device_data(), 0.,
				this->mean.mutable_device_data());

		this->numByChans.print_data({}, false);
		this->mean.print_data({}, false);
	}

	// subtract mean
	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels, 1, 1,
			this->batchSumMultiplier.device_data(), this->mean.device_data(), 0.,
			this->numByChans.mutable_device_data());
	this->numByChans.print_data({}, false);
	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels * num,
			spatialDim, 1, -1, this->numByChans.device_data(),
			this->spatialSumMultiplier.device_data(), 1., outputData);
	this->spatialSumMultiplier.print_data({}, false);

	if (!this->useGlobalStats) {
		// compute variance using var(X) = E((X-EX)^2)
		soooa_gpu_mul(this->_outputData[0]->getCount(), this->_outputData[0]->device_data(), this->_outputData[0]->device_data(),
				this->temp.mutable_device_data());  // (X-EX)^2
		soooa_gpu_gemv<Dtype>(CblasNoTrans, this->channels * num, spatialDim,
				1. / (num * spatialDim), this->temp.device_data(),
				this->spatialSumMultiplier.device_data(), 0.,
				this->numByChans.mutable_device_data());
		soooa_gpu_gemv<Dtype>(CblasTrans, num, this->channels, Dtype(1.),
				this->numByChans.device_data(), this->batchSumMultiplier.device_data(),
				Dtype(0.), this->variance.mutable_device_data());  // E((X_EX)^2)

		// compute and save moving average
		this->_params[2]->mutable_host_data()[0] *= this->movingAverageFraction;
		this->_params[2]->mutable_host_data()[0] += 1;
		soooa_gpu_axpby(this->mean.getCount(), Dtype(1), this->mean.device_data(),
				this->movingAverageFraction, this->_params[0]->mutable_device_data());
		int m = this->_inputData[0]->getCount() / this->channels;
		Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
		soooa_gpu_axpby(this->variance.getCount(), bias_correction_factor,
				this->variance.device_data(), this->movingAverageFraction,
				this->_params[1]->mutable_device_data());
	}

	// normalize variance
	soooa_gpu_add_scalar(this->variance.getCount(), this->eps, this->variance.mutable_device_data());
	this->variance.print_data({}, false);
	soooa_gpu_sqrt(this->variance.getCount(), this->variance.device_data(),
			this->variance.mutable_device_data());
	this->variance.print_data({}, false);

	// replicate variance to input size
	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels, 1, 1,
			this->batchSumMultiplier.device_data(), this->variance.device_data(), 0.,
			this->numByChans.mutable_device_data());
	this->numByChans.print_data({}, false);
	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels * num,
			spatialDim, 1, 1., this->numByChans.device_data(),
			this->spatialSumMultiplier.device_data(), 0., this->temp.mutable_device_data());
	this->temp.print_data({}, false);
	soooa_gpu_div(this->temp.getCount(), outputData, this->temp.device_data(), outputData);
	// TODO(cdoersch): The caching is only needed because later in-place layers
	//                 might clobber the data.  Can we skip this if they won't?
	this->_outputData[0]->print_data({}, false);
	soooa_copy(this->xNorm.getCount(), outputData, this->xNorm.mutable_device_data());
	this->xNorm.print_data({}, false);

	this->_printOff();
}

template <typename Dtype>
void BatchNorm3Layer<Dtype>::backpropagation() {
	const Dtype* outputGrad;
	if (this->_inputData[0] != this->_outputData[0]) {
		outputGrad = this->_outputData[0]->device_grad();
	} else {
		soooa_copy(this->xNorm.getCount(), this->_outputData[0]->device_grad(),
				this->xNorm.mutable_device_grad());
		outputGrad = this->xNorm.device_grad();
	}
	Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
	if (this->useGlobalStats) {
		soooa_gpu_div(this->temp.getCount(), outputGrad, this->temp.device_data(), inputGrad);
		return;
	}
	const Dtype* outputData = this->xNorm.device_data();
	int num = this->_inputData[0]->getShape()[0];
	int spatialDim = this->_inputData[0]->getCount() / (this->channels * this->_inputData[0]->getShape(0));
	// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
	//
	// dE(Y)/dX =
	//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
	//     ./ sqrt(var(X) + eps)
	//
	// where \cdot and ./ are hadamard product and elementwise division,
	// respectively, dE/dY is the top diff, and mean/var/sum are all computed
	// along all dimensions except the channels dimension.  In the above
	// equation, the operations allow for expansion (i.e. broadcast) along all
	// dimensions except the channels dimension where required.

	// sum(dE/dY \cdot Y)
	soooa_gpu_mul(this->temp.getCount(), outputData, outputGrad, inputGrad);
	soooa_gpu_gemv<Dtype>(CblasNoTrans, this->channels * num, spatialDim, 1.,
			inputGrad, this->spatialSumMultiplier.device_data(), 0.,
			this->numByChans.mutable_device_data());
	soooa_gpu_gemv<Dtype>(CblasTrans, num, this->channels, 1.,
			this->numByChans.device_data(), this->batchSumMultiplier.device_data(), 0.,
			this->mean.mutable_device_data());

	// reshape (broadcast) the above
	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels, 1, 1,
			this->batchSumMultiplier.device_data(), this->mean.device_data(), 0.,
			this->numByChans.mutable_device_data());
	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels * num,
			spatialDim, 1, 1., this->numByChans.device_data(),
			this->spatialSumMultiplier.device_data(), 0., inputGrad);

	// sum(dE/dY \cdot Y) \cdot Y
	soooa_gpu_mul(this->temp.getCount(), outputData, inputGrad, inputGrad);

	// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
	soooa_gpu_gemv<Dtype>(CblasNoTrans, this->channels * num, spatialDim, 1.,
			outputGrad, this->spatialSumMultiplier.device_data(), 0.,
			this->numByChans.mutable_device_data());
	soooa_gpu_gemv<Dtype>(CblasTrans, num, this->channels, 1.,
			this->numByChans.device_data(), this->batchSumMultiplier.device_data(), 0.,
			this->mean.mutable_device_data());
	// reshape (broadcast) the above to make
	// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels, 1, 1,
			this->batchSumMultiplier.device_data(), this->mean.device_data(), 0.,
			this->numByChans.mutable_device_data());
	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * this->channels,
			spatialDim, 1, 1., this->numByChans.device_data(),
			this->spatialSumMultiplier.device_data(), 1., inputGrad);

	// dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
	soooa_gpu_axpby(this->temp.getCount(), Dtype(1), outputGrad,
			Dtype(-1. / (num * spatialDim)), inputGrad);

	// note: this->temp still contains sqrt(var(X)+eps), computed during the forward
	// pass.
	soooa_gpu_div(this->temp.getCount(), inputGrad, this->temp.device_data(), inputGrad);

}

#endif

template <typename Dtype>
void BatchNorm3Layer<Dtype>::update() {
	//const uint32_t size = this->depth;
	const Dtype weightDecay = SNPROP(weightDecay);
	const Dtype learningRate = Update<float>::calcLearningRate();
	const Dtype beta1 = SNPROP(beta1);
	const Dtype beta2 = SNPROP(beta2);

	SLPROP(BatchNorm3, decayedBeta1) *= beta1;
	SLPROP(BatchNorm3, decayedBeta2) *= beta2;
	SASSUME0(this->updateParams.size() == 3);

	for (int i = 0; i < 3; i++) {
		int paramSize = this->_params[i]->getCount();
		Dtype regScale = weightDecay * this->updatePolicies[i].decay_mult;
		Dtype learnScale = learningRate * this->updatePolicies[i].lr_mult;
		UpdateContext context = Update<Dtype>::makeContext(paramSize, regScale, learnScale);
		this->updateParams[i].context = context;
	}

	Updater::updateParams(this->updateParams);
}

template <typename Dtype>
void BatchNorm3Layer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}

template <typename Dtype>
void BatchNorm3Layer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}

























/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* BatchNorm3Layer<Dtype>::initLayer() {
	BatchNorm3Layer* layer = NULL;
	SNEW(layer, BatchNorm3Layer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void BatchNorm3Layer<Dtype>::destroyLayer(void* instancePtr) {
    BatchNorm3Layer<Dtype>* layer = (BatchNorm3Layer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void BatchNorm3Layer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    BatchNorm3Layer<Dtype>* layer = (BatchNorm3Layer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool BatchNorm3Layer<Dtype>::allocLayerTensors(void* instancePtr) {
    BatchNorm3Layer<Dtype>* layer = (BatchNorm3Layer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void BatchNorm3Layer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    BatchNorm3Layer<Dtype>* layer = (BatchNorm3Layer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void BatchNorm3Layer<Dtype>::backwardTensor(void* instancePtr) {
    BatchNorm3Layer<Dtype>* layer = (BatchNorm3Layer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void BatchNorm3Layer<Dtype>::learnTensor(void* instancePtr) {
    BatchNorm3Layer<Dtype>* layer = (BatchNorm3Layer<Dtype>*)instancePtr;
    layer->update();
}

template<typename Dtype>
bool BatchNorm3Layer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t BatchNorm3Layer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
	Optimizer opt = (Optimizer)SNPROP(optimizer);
	int paramHistoryDataCount = Update<Dtype>::getParamHistoryDataCount(opt);
    const int channels = inputShape[0].C;

    uint64_t size = 0;

    // for mean, variance, varianceCorrelation params
    size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 
        paramHistoryDataCount *  2UL;
    size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 
        paramHistoryDataCount * 2UL;
    size += ALIGNUP(sizeof(Dtype), SPARAM(CUDA_MEMPAGE_SIZE)) * paramHistoryDataCount * 2UL;

    // mean, variance, temp, xNorm
    size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    // batchSumMultiplier
    size += ALIGNUP(sizeof(Dtype) * inputShape[0].N, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    return size;
}

template class BatchNorm3Layer<float>;
