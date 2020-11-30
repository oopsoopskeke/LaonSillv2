/*
 * BiasLayer.cpp
 *
 *  Created on: Jan 6, 2018
 *      Author: jkim
 */

#include "BiasLayer.h"
#include "PropMgmt.h"
#include "Updater.h"

using namespace std;

template <typename Dtype>
int BiasLayer<Dtype>::INNER_ID = 13010;


template <typename Dtype>
BiasLayer<Dtype>::BiasLayer()
: BiasLayer(NULL) {}

template <typename Dtype>
BiasLayer<Dtype>::BiasLayer(_BiasPropLayer* prop)
: LearnableLayer<Dtype>(),
  biasMultiplier("biasMultiplier") {
	this->type = Layer<Dtype>::Bias;

	if (prop) {
		this->prop = NULL;
		SNEW(this->prop, _BiasPropLayer);
		SASSUME0(this->prop != NULL);
		*(this->prop) = *(prop);
	} else {
		this->prop = NULL;
	}

	//const int numAxes = GET_PROP(prop, Bias, numAxes);
	const int numAxes = SLPROP(Bias, numAxes);
	SASSERT(numAxes >= -1, "numAxes must be non-negative, "
			"or -1 to extend to the end of input[0]");

	this->_params.resize(1);
	this->_paramsHistory.resize(1);
	this->_paramsHistory2.resize(1);
	this->_paramsInitialized.resize(1);

	LearnableLayer<Dtype>::initParam(0, "bias");
	//cout << "in bias layer: " << endl;
	//cout << this->_params[0] << "," << this->_paramsHistory[0] << "," << this->_paramsHistory2[0] << endl;

	this->updatePolicies.resize(1);
}



template <typename Dtype>
BiasLayer<Dtype>::~BiasLayer() {
	//LearnableLayer<Dtype>::releaseParam(0);
}


template <typename Dtype>
void BiasLayer<Dtype>::reshape() {
	const int axis = GET_PROP(prop, Bias, axis);
	const int numAxes = GET_PROP(prop, Bias, numAxes);

	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		const vector<uint32_t>::const_iterator& shapeStart =
				this->_inputData[0]->getShape().begin() + axis;
		const vector<uint32_t>::const_iterator& shapeEnd =
				(numAxes == -1) ? this->_inputData[0]->getShape().end() : (shapeStart + numAxes);
		vector<uint32_t> biasShape(shapeStart, shapeEnd);
		// restore scaleShape numAxes to 4
		// guaranttes scales shape numAxes is 4
		const int biasShapeSize = biasShape.size();
		for (int i = 0; i < Data<Dtype>::SHAPE_SIZE - biasShapeSize; i++) {
			biasShape.insert(biasShape.begin(), 1);
		}
		LearnableLayer<Dtype>::reshapeParam(0, biasShape);

		param_filler<Dtype>& filler =  GET_PROP(prop, Bias, filler);
		filler.fill(this->_params[0]);
	}
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	SASSERT(this->_inputData[0]->numAxes() >= axis + numAxes,
			"bias params's shape extends past input[0]'s shape when applied "
			"starting with input[0] axis = %d", axis);
	for (int i = 0; i < numAxes; i++) {
		SASSERT(this->_inputData[0]->getShape(axis + i) ==
				this->_params[0]->getShape(Data<Dtype>::SHAPE_SIZE - numAxes + i),
				"dimension mismatch between _inputData[0]->getShape(%d) and bias->getShape(%d)",
				axis + i, Data<Dtype>::SHAPE_SIZE - numAxes + i);
	}

	this->outerDim = this->_inputData[0]->getCountByAxis(0, axis);
	this->biasDim = this->_params[0]->getCount();
	this->innerDim = this->_inputData[0]->getCountByAxis(axis + numAxes);
	this->dim = this->biasDim * this->innerDim;

	if (this->_inputData[0] != this->_outputData[0]) {
		this->_outputData[0]->reshapeLike(this->_inputData[0]);
	}
	this->biasMultiplier.reshape({1, 1, 1, (uint32_t)this->innerDim});
	if (this->biasMultiplier.host_data()[this->innerDim - 1] != Dtype(1)) {
		soooa_set(this->innerDim, Dtype(1), this->biasMultiplier.mutable_host_data());
	}
}

template<typename Dtype>
__global__ void BiasForward(const int n, const Dtype* in, const Dtype* bias,
		const int bias_dim, const int inner_dim, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n) {
		const int bias_index = (index / inner_dim) % bias_dim;
		out[index] = in[index] + bias[bias_index];
	}
}

template<typename Dtype>
void BiasLayer<Dtype>::feedforward() {
	const int count = this->_outputData[0]->getCount();
	const Dtype* inputData = this->_inputData[0]->device_data();
	const Dtype* biasData = this->_params[0]->device_data();
	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	BiasForward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(count, inputData,
			biasData, this->biasDim, this->innerDim, outputData);

}

template<typename Dtype>
void BiasLayer<Dtype>::backpropagation() {
	if (this->_inputData[0] != this->_outputData[0]) {
		const Dtype* outputGrad = this->_outputData[0]->device_grad();
		Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
		soooa_copy(this->_inputData[0]->getCount(), outputGrad, inputGrad);
	}

	// in-place, we don't need to do anything with the data diff
	const bool biasParam = (this->_inputData.size() == 1);
	//if ((!biasParam && propagate_down[1])
	//		|| (biasParam && this->param_propagate_down_[0])) {
	const Dtype* outputGrad = this->_outputData[0]->device_grad();
	Dtype* biasGrad = this->_params[0]->mutable_device_grad();
	bool accum = biasParam;
	for (int n = 0; n < this->outerDim; ++n) {
		soooa_gpu_gemv(CblasNoTrans, this->biasDim, this->innerDim, Dtype(1),
				outputGrad, this->biasMultiplier.device_data(), Dtype(accum),
				biasGrad);
		outputGrad += this->dim;
		accum = true;
	}
	//}
}





template <typename Dtype>
void BiasLayer<Dtype>::update() {
	const Dtype weightDecay = SNPROP(weightDecay);
	const Dtype learningRate = Update<float>::calcLearningRate();
	const Dtype beta1 = SNPROP(beta1);
	const Dtype beta2 = SNPROP(beta2);

	GET_PROP(prop, Bias, decayedBeta1) *= beta1;
	GET_PROP(prop, Bias, decayedBeta2) *= beta2;
	SASSUME0(this->updateParams.size() == 1);

	for (int i = 0; i < 1; i++) {
		int paramSize = this->_params[i]->getCount();
		Dtype regScale = weightDecay * this->updatePolicies[i].decay_mult;
		Dtype learnScale = learningRate * this->updatePolicies[i].lr_mult;
		UpdateContext context = Update<Dtype>::makeContext(paramSize, regScale, learnScale);
		this->updateParams[i].context = context;
	}

	Updater::updateParams(this->updateParams);
}

template <typename Dtype>
void BiasLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}

template <typename Dtype>
void BiasLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}

























/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* BiasLayer<Dtype>::initLayer() {
	BiasLayer* layer = NULL;
	SNEW(layer, BiasLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void BiasLayer<Dtype>::destroyLayer(void* instancePtr) {
	cout << "BiasLayer<Dtype>::destroyLayer" << endl;
    BiasLayer<Dtype>* layer = (BiasLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void BiasLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    BiasLayer<Dtype>* layer = (BiasLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool BiasLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    BiasLayer<Dtype>* layer = (BiasLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void BiasLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    BiasLayer<Dtype>* layer = (BiasLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void BiasLayer<Dtype>::backwardTensor(void* instancePtr) {
    BiasLayer<Dtype>* layer = (BiasLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void BiasLayer<Dtype>::learnTensor(void* instancePtr) {
    BiasLayer<Dtype>* layer = (BiasLayer<Dtype>*)instancePtr;
    layer->update();
}

template<typename Dtype>
bool BiasLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t BiasLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
	Optimizer opt = (Optimizer)SNPROP(optimizer);
	int paramHistoryDataCount = Update<Dtype>::getParamHistoryDataCount(opt);

    const int channels = inputShape[0].C;
    uint64_t size = 0;

    size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 
        paramHistoryDataCount *  2UL;

	const int axis = SLPROP(Bias, axis);
	const int numAxes = SLPROP(Bias, numAxes);
    const size_t innerDim = tensorCount(inputShape[0], axis + numAxes);

    size += ALIGNUP(sizeof(Dtype) * innerDim, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    return size;
}

template class BiasLayer<float>;
