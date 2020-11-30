/*
 * BatchNorm2Layer.cpp
 *
 *  Created on: Dec 18, 2017
 *      Author: jkim
 */

#include "EnumDef.h"
#include "BatchNorm2Layer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"
#include "MathFunctions.h"
#include "Updater.h"
#include "StdOutLog.h"
#include "Update.h"

using namespace std;

template <typename Dtype>
BatchNorm2Layer<Dtype>::BatchNorm2Layer()
: LearnableLayer<Dtype>() {
	this->type = Layer<Dtype>::BatchNorm2;

    SASSERT(SLPROP(BatchNorm2, useGlobalStats) == BatchNormUseGlobal::DependOnNetStat,
            "BatchNorm2Layer only supports BatchNormUseGlobal::DependOnNetStat.");
	this->movingAverageFraction = SLPROP(BatchNorm2, movingAverageFraction);
    this->eps = std::max<double>(SLPROP(BatchNorm2, eps), CUDNN_BN_MIN_EPSILON);
    // scaleBias 여부는 오직 BatchNorm2의 scaleBias에 의해 결정됨.
    // 별도 scaleBias에 관한 filler prop이 있어도 scaleBias가 false이면 무시함.
	this->scaleBias = SLPROP(BatchNorm2, scaleBias);

    const int numParams = this->scaleBias ? 5 : 3;
    LearnableLayer<Dtype>::resizeParam(numParams);

	LearnableLayer<Dtype>::initParam(0, "mean");
	LearnableLayer<Dtype>::initParam(1, "variance");
	LearnableLayer<Dtype>::initParam(2, "varianceCorrelation");

	if (this->scaleBias) {
		LearnableLayer<Dtype>::initParam(3, "scale");
		LearnableLayer<Dtype>::initParam(4, "bias");
	}
	this->iter = 0;

	// Mask statistics from optimization by setting local learning rates
	// for mean, variance, and the this->varcorrection to zero.
	this->updatePolicies.resize(3);
	for (int i = 0; i < 3; i++) {
		// set lr and decay = 0 for global mean and variance
		this->updatePolicies[i].lr_mult = 0.f;
		this->updatePolicies[i].decay_mult = 0.f;
	}
	// set lr for scale and bias to 1
	if (this->scaleBias) {
		this->updatePolicies.resize(5);
		for (int i = 3; i < 5; i++) {
			// set lr and decay = 1 for scale and bias
			this->updatePolicies[i].lr_mult = 1.f;
			this->updatePolicies[i].decay_mult = 1.f;
		}
	}

    checkCUDNN(cudnnCreateTensorDescriptor(&this->fwdInputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&this->fwdOutputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&this->fwdScaleBiasMeanVarDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&this->bwdInputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&this->bwdOutputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&this->bwdScaleBiasMeanVarDesc));

    this->mode = CUDNN_BATCHNORM_SPATIAL;      // only SPATIAL mode is supported

    if (!this->scaleBias) {
        this->scaleOnes = make_shared<Data<Dtype>>("scaleOnes");
        this->biasZeros = make_shared<Data<Dtype>>("biasZeros");
    }
    this->saveMean = make_shared<Data<Dtype>>("saveMean");
    this->saveInvVar = make_shared<Data<Dtype>>("saveInvVar");

    this->handlesSetup = true;
}

template <typename Dtype>
BatchNorm2Layer<Dtype>::~BatchNorm2Layer() {
	for (int i = 0; i < this->_params.size(); i++) {
		LearnableLayer<Dtype>::releaseParam(i);
	}
	this->updateParams.clear();


    if (!this->handlesSetup) {
        return;
    }
    cudnnDestroyTensorDescriptor(this->fwdInputDesc);
    cudnnDestroyTensorDescriptor(this->bwdInputDesc);
    cudnnDestroyTensorDescriptor(this->fwdOutputDesc);
    cudnnDestroyTensorDescriptor(this->bwdOutputDesc);
    cudnnDestroyTensorDescriptor(this->fwdScaleBiasMeanVarDesc);
    cudnnDestroyTensorDescriptor(this->bwdScaleBiasMeanVarDesc);

}

template <typename Dtype>
void BatchNorm2Layer<Dtype>::reshape() {
    SASSERT(this->_inputData[0] != this->_outputData[0], 
            "BatchNorm2Layer does not support In-Place operation");

	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
        // tensor가 최초 reshape() 시점에서 결정되기 때문에
        // 여기서 channels 값을 결정할 수 있음.
		this->channels = this->_inputData[0]->getShape(1);

        vector<uint32_t> shape = {1, 1, 1, (uint32_t)this->channels};

		LearnableLayer<Dtype>::reshapeParam(0, shape);
		this->_params[0]->reset_host_data();
		LearnableLayer<Dtype>::reshapeParam(1, shape);
		this->_params[1]->reset_host_data();

        shape[3] = 1;
		LearnableLayer<Dtype>::reshapeParam(2, shape);
		this->_params[2]->reset_host_data();

		if (this->scaleBias) {
            shape[3] = this->channels;
			LearnableLayer<Dtype>::reshapeParam(3, shape);
			LearnableLayer<Dtype>::reshapeParam(4, shape);

            // layerPropDef의 scaleFiller의 default값은 Constant Dtype(1)이므로
            // scaleFiller를 사용자가 별도 설정하지 않아도 기본값이 적용된다.
			param_filler<Dtype>& scaleFiller = SLPROP(BatchNorm2, scaleFiller);
			scaleFiller.fill(this->_params[3]);

            // layerPropDef의 biasFiller의 default값은 Constant Dtype(0)이므로
            // biasFiller를 사용자가 별도 설정하지 않아도 기본값이 적용된다.
			param_filler<Dtype>& biasFiller = SLPROP(BatchNorm2, biasFiller);
			biasFiller.fill(this->_params[4]);
		}

        // ====================================================
        uint32_t N = this->_inputData[0]->getShape(0);
        uint32_t C = this->_inputData[0]->getShape(1);
        uint32_t H = this->_inputData[0]->getShape(2);
        uint32_t W = this->_inputData[0]->getShape(3);

        shape[3] = C;
        if (!this->scaleBias) {
            this->scaleOnes->reshape(shape);
            this->scaleOnes->reset_host_data(false, Dtype(1));
            this->biasZeros->reshape(shape);
            this->biasZeros->reset_host_data();
        }
        this->saveMean->reshape(shape);
        this->saveInvVar->reshape(shape);
	}

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

    this->_inputShape[0] = this->_inputData[0]->getShape();

    SASSERT0(this->_inputData[0]->getShape(1) == this->channels);
	this->_outputData[0]->reshapeLike(this->_inputData[0]);

	uint32_t N = this->_inputData[0]->getShape(0);
	uint32_t C = this->_inputData[0]->getShape(1);
	uint32_t H = this->_inputData[0]->getShape(2);
	uint32_t W = this->_inputData[0]->getShape(3);

    const int stride_w = 1;
    const int stride_h = W * stride_w;
    const int stride_c = H * stride_h;
    const int stride_n = C * stride_c;

    // set up main tensors
    checkCUDNN(cudnnSetTensor4dDescriptorEx(this->fwdInputDesc, CUDNN_DATA_FLOAT,
                N, C, H, W, stride_n, stride_c, stride_h, stride_w));
    checkCUDNN(cudnnSetTensor4dDescriptorEx(this->fwdOutputDesc, CUDNN_DATA_FLOAT,
                N, C, H, W, stride_n, stride_c, stride_h, stride_w));
    checkCUDNN(cudnnSetTensor4dDescriptorEx(this->bwdInputDesc, CUDNN_DATA_FLOAT,
                N, C, H, W, stride_n, stride_c, stride_h, stride_w));
    checkCUDNN(cudnnSetTensor4dDescriptorEx(this->bwdOutputDesc, CUDNN_DATA_FLOAT,
                N, C, H, W, stride_n, stride_c, stride_h, stride_w));


    // aux tensors for caching mean & invVar from fwd to bwd pass
    vector<uint32_t> shape = {1, 1, 1, C};
    this->saveMean->reshape(shape);
    this->saveInvVar->reshape(shape);

    if (!this->scaleBias) {
        int C_old = this->scaleOnes->channels();
        if (C_old != C) {
            this->scaleOnes->reshape(shape);
            this->scaleOnes->reset_host_data(false, Dtype(1));
            this->biasZeros->reshape(shape);
            this->biasZeros->reset_host_data();
        }
    }
    checkCUDNN(cudnnDeriveBNTensorDescriptor(this->fwdScaleBiasMeanVarDesc, this->fwdInputDesc, this->mode));
    checkCUDNN(cudnnDeriveBNTensorDescriptor(this->bwdScaleBiasMeanVarDesc, this->bwdInputDesc, this->mode));

}


template <typename Dtype>
void BatchNorm2Layer<Dtype>::feedforward() {
    reshape();

    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* outputData = this->_outputData[0]->mutable_device_data();

    double epsilon = this->eps;
    const Dtype* scaleData;
    const Dtype* biasData;
    Dtype* globalMean;
    Dtype* globalVar;
    Dtype* saveMean;
    Dtype* saveInvVar;

    globalMean = this->_params[0]->mutable_device_data();
    globalVar = this->_params[1]->mutable_device_data();
    if (SNPROP(status) == NetworkStatus::Train) {
        saveMean = this->saveMean->mutable_device_data();
        saveInvVar = this->saveInvVar->mutable_device_data();
    }

    if (this->scaleBias) {
        scaleData = this->_params[3]->device_data();
        biasData = this->_params[4]->device_data();
    } else {
        scaleData = this->scaleOnes->device_data();
        biasData = this->biasZeros->device_data();
    }

    if (SNPROP(status) == NetworkStatus::Train) {
        double factor = 1. - this->movingAverageFraction; 
        if (this->iter == 0) {
            factor = 1.0;
        }
        checkCUDNN(cudnnBatchNormalizationForwardTraining(Cuda::cudnnHandle, this->mode,
            &Cuda::alpha, &Cuda::beta,
            this->fwdInputDesc, inputData, this->fwdOutputDesc, outputData,
            this->fwdScaleBiasMeanVarDesc, scaleData, biasData,
            factor, globalMean, globalVar, epsilon, saveMean, saveInvVar));

    } else if (SNPROP(status) == NetworkStatus::Test) {
        checkCUDNN(cudnnBatchNormalizationForwardInference(Cuda::cudnnHandle,
            CUDNN_BATCHNORM_SPATIAL,
            &Cuda::alpha, &Cuda::beta,
            this->fwdInputDesc, inputData, this->fwdOutputDesc, outputData,
            this->fwdScaleBiasMeanVarDesc, scaleData, biasData,
            globalMean, globalVar, epsilon));

    } else {
        SASSERT(false, "Unknown Network Status");
    }
}

template <typename Dtype>
void BatchNorm2Layer<Dtype>::backpropagation() {
    const Dtype* outputGrad = this->_outputData[0]->device_grad();
    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();

    double epsilon = this->eps;
    const Dtype* saveMean;
    const Dtype* saveInvVar;
    const Dtype* scaleData;
    Dtype* scaleGrad;
    Dtype* biasGrad;

    saveMean = this->saveMean->device_data();
    saveInvVar = this->saveInvVar->device_data();
    if (this->scaleBias) {
        scaleData = this->_params[3]->device_data();
        scaleGrad = this->_params[3]->mutable_device_grad();
        biasGrad = this->_params[4]->mutable_device_grad();
    } else {
        scaleData = this->scaleOnes->device_data();
        scaleGrad = this->scaleOnes->mutable_device_grad();
        biasGrad = this->biasZeros->mutable_device_grad();
    }

    checkCUDNN(cudnnBatchNormalizationBackward(Cuda::cudnnHandle, this->mode,
        &Cuda::alpha, &Cuda::beta,
        &Cuda::alpha, &Cuda::alpha,
        this->bwdInputDesc, inputData, this->bwdInputDesc, outputGrad, this->bwdInputDesc, inputGrad,
        this->bwdScaleBiasMeanVarDesc, scaleData, scaleGrad, biasGrad,
        epsilon, saveMean, saveInvVar));
}


template <typename Dtype>
void BatchNorm2Layer<Dtype>::update() {
	//const uint32_t size = this->depth;
	const Dtype weightDecay = SNPROP(weightDecay);
	const Dtype learningRate = Update<float>::calcLearningRate();
	const Dtype beta1 = SNPROP(beta1);
	const Dtype beta2 = SNPROP(beta2);

	SLPROP(BatchNorm2, decayedBeta1) *= beta1;
	SLPROP(BatchNorm2, decayedBeta2) *= beta2;

	if (this->scaleBias) {
		SASSUME0(this->updateParams.size() == 5);
	} else {
		SASSUME0(this->updateParams.size() == 3);
	}

	for (int i = 0; i < 5; i++) {
		if (i >= 3 && !this->scaleBias) {
			continue;
		}
		int paramSize = this->_params[i]->getCount();
		Dtype regScale = weightDecay * this->updatePolicies[i].decay_mult;
		Dtype learnScale = learningRate * this->updatePolicies[i].lr_mult;
		UpdateContext context = Update<Dtype>::makeContext(paramSize, regScale, learnScale);
		this->updateParams[i].context = context;
	}

	Updater::updateParams(this->updateParams);
}

template <typename Dtype>
void BatchNorm2Layer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}

template <typename Dtype>
void BatchNorm2Layer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}

























/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* BatchNorm2Layer<Dtype>::initLayer() {
	BatchNorm2Layer* layer = NULL;
	SNEW(layer, BatchNorm2Layer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::destroyLayer(void* instancePtr) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool BatchNorm2Layer<Dtype>::allocLayerTensors(void* instancePtr) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::backwardTensor(void* instancePtr) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::learnTensor(void* instancePtr) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    layer->update();
}

template<typename Dtype>
bool BatchNorm2Layer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t BatchNorm2Layer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
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
    
    // for scale, bias
	bool scaleBias = SLPROP(BatchNorm2, scaleBias);
    if (scaleBias) {
        size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 
            paramHistoryDataCount * 2UL;
        size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 
            paramHistoryDataCount * 2UL;
    } else {
        size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
        size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    }

    // for saveMean, saveInvVar
    size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    size += ALIGNUP(sizeof(Dtype) * channels, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    return size;
}

template class BatchNorm2Layer<float>;
