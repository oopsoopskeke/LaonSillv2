/*
 * NormalizeLayer.cpp
 *
 *  Created on: Apr 21, 2017
 *      Author: jkim
 */

#include "NormalizeLayer.h"
#include "MathFunctions.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "PropMgmt.h"
#include "Donator.h"
#include "Updater.h"
#include "MemoryMgmt.h"

#define NORMALIZELAYER_LOG	0

using namespace std;


// divid a matrix with vector
template <typename Dtype>
__global__ void DivBsx(const int nthreads, const Dtype* A,
		const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
		Dtype* B) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int c = index % cols;
		int r = (index / cols) % rows;
		if (trans == CblasNoTrans) {
			B[index] = A[index] / v[c];
		} else {
			B[index] = A[index] / v[r];
		}
	}
}

template <typename Dtype>
__global__ void MulBsx(const int nthreads, const Dtype* A,
		const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
		Dtype* B) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int c = index % cols;
		int r = (index / cols) % rows;
		if (trans == CblasNoTrans) {
			B[index] = A[index] * v[c];
		} else {
			B[index] = A[index] * v[r];
		}
	}
}




template <typename Dtype>
NormalizeLayer<Dtype>::NormalizeLayer()
: LearnableLayer<Dtype>(),
  //acrossSpatial(builder->_acrossSpatial),
  //channelShared(builder->_channelShared),
  //scaleUpdateParam(builder->_scaleUpdateParam),
  //scaleFiller(builder->_scaleFiller),
  //eps(builder->_eps),
  buffer_("buffer"),
  bufferChannel_("bufferChannel"),
  bufferSpatial_("bufferSpatial"),
  sumChannelMultiplier_("sumChannelMultiplier") {
	this->type = Layer<Dtype>::Normalize;

	this->tempCount = 0;
	//initialize();

	this->_params.resize(1);
	this->_paramsHistory.resize(1);
	this->_paramsHistory2.resize(1);

	this->_params[0] = NULL;
	SNEW(this->_params[0], Data<Dtype>, SLPROP_BASE(name) + "_scale");
	SASSUME0(this->_params[0] != NULL);

	this->_paramsHistory[0] = NULL;
	SNEW(this->_paramsHistory[0], Data<Dtype>, SLPROP_BASE(name) + "_scale_history");
	SASSUME0(this->_paramsHistory[0] != NULL);

	this->_paramsHistory2[0] = NULL;
	SNEW(this->_paramsHistory2[0], Data<Dtype>, SLPROP_BASE(name) + "_scale_history2");
	SASSUME0(this->_paramsHistory2[0] != NULL);

	this->_paramsInitialized.resize(1);
	this->_paramsInitialized[0] = false;
}

template <typename Dtype>
NormalizeLayer<Dtype>::~NormalizeLayer() {
	if (SLPROP_BASE(receive)) {
	    Donator<Dtype>::releaseReceiver(SLPROP_BASE(donatorID));
	} else {
		Util::clearVector(this->_params);
		Util::clearVector(this->_paramsHistory);
		Util::clearVector(this->_paramsHistory2);
	}
	this->updateParams.clear();
}

template <typename Dtype>
void NormalizeLayer<Dtype>::reshape() {
	SASSERT(this->_inputData[0]->numAxes() >= 2,
			"Number of axes of input data must be >= 2.");

	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	this->_outputData[0]->reshapeLike(this->_inputData[0]);

	const vector<uint32_t>& dataShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = dataShape;

	this->buffer_.reshape({1, dataShape[1], dataShape[2], dataShape[3]});
	this->bufferChannel_.reshape({1, dataShape[1], 1, 1});
	this->bufferSpatial_.reshape({1, 1, dataShape[2], dataShape[3]});

	if (SLPROP(Normalize, acrossSpatial)) {
		this->norm_.reshape({dataShape[0], 1, 1, 1});
	} else {
		this->norm_.reshape({dataShape[0], 1, dataShape[2], dataShape[3]});
	}

	uint32_t channels = dataShape[1];
	uint32_t spatialDim = dataShape[2] * dataShape[3];

	this->sumChannelMultiplier_.reshape({1, channels, 1, 1});
	this->sumChannelMultiplier_.reset_host_data(false, Dtype(1.0));
	this->sumSpatialMultiplier_.reshape({1, 1, dataShape[2], dataShape[3]});
	this->sumSpatialMultiplier_.reset_host_data(false, Dtype(1.0));

	// initialize에서 처리하는 것이 바람직하지만
	// 현재 initialize에서 channels를 알 방법이 없어서 reshape에서 처리.
	// 사실상 다시 reshape되는 상황에서 종료
	//SASSERT0(this->_paramsInitialized[0] == false);
	// channel 무관하게 single scale 사용
	if (SLPROP(Normalize, channelShared)) {
		this->_params[0]->reshape({1, 1, 1, 1});
		this->_paramsHistory[0]->reshape({1, 1, 1, 1});
		this->_paramsHistory2[0]->reshape({1, 1, 1, 1});
	}
	// channel별 별도 scale 사용
	else {
		this->_params[0]->reshape({1, 1, 1, channels});
		this->_paramsHistory[0]->reshape({1, 1, 1, channels});
		this->_paramsHistory2[0]->reshape({1, 1, 1, channels});
	}

	if (this->updateParams.size() == 0) {
		UpdateParam up;
		up.paramType = 0;
		up.paramDataPtr = (void*)this->_params[0];
		up.paramHis1Ptr = (void*)this->_paramsHistory[0];
		up.paramHis2Ptr = (void*)this->_paramsHistory2[0];
		this->updateParams.push_back(up);
	}

	if (!this->_paramsInitialized[0]) {
		SLPROP(Normalize, scaleFiller).fill(this->_params[0]);
		this->_paramsInitialized[0] = true;
	}

	if (SLPROP(Normalize, channelShared)) {
		SASSERT(this->_params[0]->getCount() == 1,
				"Scale size is inconsistent with prototxt config.");
	} else {
		SASSERT(this->_params[0]->getCount() == channels,
				"Scale size is inconsistent with prototxt config.");
	}
}


template <typename Dtype>
void NormalizeLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* inputData = this->_inputData[0]->device_data();
	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	Dtype* bufferData = this->buffer_.mutable_device_data();
	Dtype* normData;
	if (SLPROP(Normalize, acrossSpatial)) {
		// need to index it
		normData = this->norm_.mutable_host_data();
	} else {
		// add eps to avoid overflow
		normData = this->norm_.mutable_device_data();
		soooa_gpu_set<Dtype>(this->norm_.getCount(), Dtype(SLPROP(Normalize, eps)), normData);
	}

	const Dtype* scale;
	if (SLPROP(Normalize, channelShared)) {
		// vector 연산을 할 필요가 없어 host에서 scalar norm을 계산함.
		scale = this->_params[0]->host_data();
	} else {
		scale = this->_params[0]->device_data();
	}

	const Dtype* sumChannelMultiplier = this->sumChannelMultiplier_.device_data();
	int num = this->_inputData[0]->batches();
	int dim = this->_inputData[0]->getCount() / num;
	int spatialDim = this->_inputData[0]->height() * this->_inputData[0]->width();
	int channels = this->_inputData[0]->channels();

	for (int n = 0; n < num; n++) {
		// bufferData = inputData^2
		soooa_gpu_powx<Dtype>(dim, inputData, Dtype(2), bufferData);

		// 이미지 하나 전체에 대해 norm 적용
		if (SLPROP(Normalize, acrossSpatial)) {
			Dtype normsqr;
			soooa_gpu_asum<Dtype>(dim, bufferData, &normsqr);
			// add eps to avoid overflow
			normData[n] = pow(normsqr + SLPROP(Normalize, eps), Dtype(0.5));
			soooa_gpu_scale<Dtype>(dim, Dtype(1.0 / normData[n]), inputData, outputData);
		}
		// 채널간 spatialDim 단위로 norm 적용
		else {

			//this->buffer_.print_shape();
			//this->sumChannelMultiplier_.print_shape();

			// compute norm
			soooa_gpu_gemv<Dtype>(CblasTrans, channels, spatialDim, Dtype(1.0), bufferData,
					sumChannelMultiplier, Dtype(1.0), normData);
			soooa_gpu_powx<Dtype>(spatialDim, normData, Dtype(0.5), normData);
			// scale the layer
			DivBsx<Dtype><<<SOOOA_GET_BLOCKS(dim), SOOOA_CUDA_NUM_THREADS>>>(
					dim, inputData, normData, channels, spatialDim, CblasNoTrans, outputData);
			CUDA_POST_KERNEL_CHECK;
			normData += spatialDim;
		}

		// scale the output
		if (SLPROP(Normalize, channelShared)) {
			soooa_gpu_scal<Dtype>(dim, scale[0], outputData);
		} else {
			MulBsx<Dtype><<<SOOOA_GET_BLOCKS(dim), SOOOA_CUDA_NUM_THREADS>>>(
					dim, outputData, scale, channels, spatialDim, CblasTrans, outputData);
			CUDA_POST_KERNEL_CHECK;
		}
		inputData += dim;
		outputData += dim;
	}
}



template <typename Dtype>
void NormalizeLayer<Dtype>::backpropagation() {
	this->tempCount++;

	const Dtype* outputGrad = this->_outputData[0]->device_grad();
	const Dtype* outputData = this->_outputData[0]->device_data();
	const Dtype* inputData = this->_inputData[0]->device_data();
	Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();

	const Dtype* normData;
	if (SLPROP(Normalize, acrossSpatial)) {
		// need to index it
		normData = this->norm_.host_data();
	} else {
		normData = this->norm_.device_data();
	}

	const Dtype* scale;
	if (SLPROP(Normalize, channelShared)) {
		scale = this->_params[0]->host_data();
	} else {
		scale = this->_params[0]->device_data();
	}

	Dtype* bufferData = this->buffer_.mutable_device_data();
	Dtype* bufferChannel = this->bufferChannel_.mutable_device_data();
	Dtype* bufferSpatial = this->bufferSpatial_.mutable_device_data();
	const Dtype* sumChannelMultiplier = this->sumChannelMultiplier_.device_data();
	const Dtype* sumSpatialMultiplier = this->sumSpatialMultiplier_.device_data();

	int count = this->_outputData[0]->getCount();
	int num = this->_outputData[0]->batches();
	int dim = count / num;
	int spatialDim = this->_outputData[0]->height() * this->_outputData[0]->width();
	int channels = this->_outputData[0]->channels();

	// propagate to param
	if (SLPROP(Normalize, channelShared)) {
		Dtype* scaleGrad = this->_params[0]->mutable_host_grad();
		Dtype a;
		soooa_gpu_dot<Dtype>(count, outputData, outputGrad, &a);
		scaleGrad[0] += a / scale[0];
	} else {
		this->_params[0]->reset_device_grad();
		Dtype* scaleGrad = this->_params[0]->mutable_device_grad();


		for (int n = 0; n < num; n++) {
			// compute a
			soooa_gpu_mul<Dtype>(dim, outputData + n * dim, outputGrad + n *dim, bufferData);
			soooa_gpu_gemv<Dtype>(CblasNoTrans, channels, spatialDim, Dtype(1.0),
					bufferData, sumSpatialMultiplier, Dtype(0.0),
					bufferChannel);

#if NORMALIZELAYER_LOG
			if (tempCount == 2) {
				this->_printOn();
				this->buffer_.print_data({}, false);
				cout << "buffer asum: " << this->buffer_.asum_device_data() << endl;
				this->bufferChannel_.print_data({}, false);
				cout << "bufferChannel asum: " << this->bufferChannel_.asum_device_data() << endl;
				this->_printOff();
			}
#endif

			// store a / scale[i] in bufferData temporary
			soooa_gpu_div<Dtype>(channels, bufferChannel, scale, bufferChannel);
			soooa_gpu_add<Dtype>(channels, bufferChannel, scaleGrad, scaleGrad);

#if NORMALIZELAYER_LOG
			if (tempCount == 2) {
				this->_printOn();
				this->bufferChannel_.print_data({}, false);
				cout << "bufferChannel asum: " << this->bufferChannel_.asum_device_data() << endl;
				this->_params[0]->print_grad({}, false);
				cout << "params asum: " << this->_params[0]->asum_device_grad() << endl;
				this->_printOff();
				exit(1);
			}
#endif
		}
	}

	// propagate to bottom
	if (SLPROP_BASE(propDown)[0]) {
		for (int n = 0; n < num; n++) {
			if (SLPROP(Normalize, acrossSpatial)) {
				Dtype a;
				soooa_gpu_dot<Dtype>(dim, inputData, outputGrad, &a);
				soooa_gpu_scale<Dtype>(dim, a / normData[n] / normData[n], inputData,
						inputGrad);
				soooa_gpu_sub<Dtype>(dim, outputGrad, inputGrad, inputGrad);
				soooa_gpu_scale<Dtype>(dim, Dtype(1.0 / normData[n]), inputGrad, inputGrad);
			} else {
				// dot product between inputData and outputGrad
				soooa_gpu_mul<Dtype>(dim, inputData, outputGrad, bufferData);
				soooa_gpu_gemv<Dtype>(CblasTrans, channels, spatialDim, Dtype(1.0),
						bufferData, sumChannelMultiplier, Dtype(0.0),
						bufferSpatial);
				// scale bottomGrad
				MulBsx<Dtype><<<SOOOA_GET_BLOCKS(dim), SOOOA_CUDA_NUM_THREADS>>>(
						dim, inputData, bufferSpatial, channels, spatialDim,
						CblasNoTrans, inputGrad);
				CUDA_POST_KERNEL_CHECK;
				// divide by square of norm
				soooa_gpu_powx<Dtype>(spatialDim, normData, Dtype(2.0), bufferSpatial);
				DivBsx<Dtype> <<<SOOOA_GET_BLOCKS(dim), SOOOA_CUDA_NUM_THREADS>>>(
						dim, inputGrad, bufferSpatial, channels, spatialDim,
						CblasNoTrans, inputGrad);
				CUDA_POST_KERNEL_CHECK;
				soooa_gpu_sub<Dtype>(dim, outputGrad, inputGrad, inputGrad);
				// divide by norm
				DivBsx<Dtype><<<SOOOA_GET_BLOCKS(dim), SOOOA_CUDA_NUM_THREADS>>>(
						dim, inputGrad, normData, channels, spatialDim, CblasNoTrans,
						inputGrad);
				CUDA_POST_KERNEL_CHECK;
				normData += spatialDim;
			}
			// scald the grad
			if (SLPROP(Normalize, channelShared)) {
				soooa_gpu_scal<Dtype>(dim, scale[0], inputGrad);
			} else {
				MulBsx<Dtype><<<SOOOA_GET_BLOCKS(dim), SOOOA_CUDA_NUM_THREADS>>>(
						dim, inputGrad, scale, channels, spatialDim, CblasTrans,
						inputGrad);
				CUDA_POST_KERNEL_CHECK;
			}
			inputData += dim;
			outputGrad += dim;
			inputGrad += dim;
		}
	}
}

template <typename Dtype>
void NormalizeLayer<Dtype>::update() {
	const uint32_t weightSize = this->_params[0]->getCount();
	const Dtype regScale = SNPROP(weightDecay);
	const Dtype learnScale = Update<Dtype>::calcLearningRate();

	//const Dtype epsilon = SNPROP(epsilon);
	//const Dtype decayRate = SNPROP(decayRate);
	//const Dtype beta1 = SNPROP(beta1);
	//const Dtype beta2 = SNPROP(beta2);

	UpdateContext contextFilter =
			Update<Dtype>::makeContext(weightSize, regScale, learnScale);

	this->updateParams[0].context = contextFilter;

	Updater::updateParams(this->updateParams);
}


/*
template <typename Dtype>
void NormalizeLayer<Dtype>::_updateParam(const uint32_t paramSize, const Dtype regScale,
    const Dtype learnScale, const Dtype epsiloffn, const Dtype decayRate, const Dtype beta1,
    const Dtype beta2, Data<Dtype>* dataHistory, Data<Dtype>* dataHistory2,
    Data<Dtype>* data) {

	const uint32_t batches = this->_inputData[0]->batches();
	const Dtype momentum = this->networkConfig->_momentum;
	const Dtype negativeOne = Dtype(-1.0);

    if (!Worker<Dtype>::isSingle())
        data->mutable_host_grad();
	Dtype* d_paramGrad = data->mutable_device_grad();
	Dtype* d_paramData = data->mutable_device_data();
	Dtype* d_paramHistoryData = dataHistory->mutable_device_data();

    // (2) apply optimizer
    Optimizer opt = this->networkConfig->_optimizer;
    SASSERT0(opt == Optimizer::Momentum);

    //this->_printOn();


    data->print_data({}, false);
    data->print_grad({}, false);
	soooa_gpu_axpy(static_cast<int>(paramSize), regScale, d_paramData, d_paramGrad);
	data->print_grad({}, false);
	dataHistory->print_data({}, false);
	soooa_gpu_axpby(static_cast<int>(paramSize), learnScale, d_paramGrad, momentum,
			d_paramHistoryData);
	dataHistory->print_data({}, false);
	soooa_copy(static_cast<int>(paramSize), d_paramHistoryData, d_paramGrad);
	data->print_grad({}, false);
	// update
	data->print_data({}, false);
	data->print_grad({}, false);
	soooa_gpu_axpy(static_cast<int>(paramSize), negativeOne, d_paramGrad, d_paramData);
	data->print_data({}, false);


	//this->_printOff();
}
*/


template <typename Dtype>
void NormalizeLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	/*
	cout << "!!!!!!!!!! applyChanges " << endl;
    const uint32_t paramSize = this->_params[0]->getCount();
    targetLayer->_params[0]->add_device_grad(this->_params[0]);
    */
	SASSERT(false, "Not implemented yet.");
}

template <typename Dtype>
void NormalizeLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	/*
	cout << "!!!!!!!!!! syncParams " << endl;
    const uint32_t paramSize = this->_params[0]->getCount();
    NormalizeLayer<Dtype>* _targetLayer = (NormalizeLayer<Dtype>*)targetLayer;

    this->_params[0]->set_device_grad(_targetLayer->_params[0]);
    */
	SASSERT(false, "Not implemented yet.");
}















/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* NormalizeLayer<Dtype>::initLayer() {
	NormalizeLayer* layer = NULL;
	SNEW(layer, NormalizeLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void NormalizeLayer<Dtype>::destroyLayer(void* instancePtr) {
    NormalizeLayer<Dtype>* layer = (NormalizeLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void NormalizeLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	if (isInput) {
		SASSERT0(index < 1);
	} else {
		SASSERT0(index < 1);
	}

    NormalizeLayer<Dtype>* layer = (NormalizeLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool NormalizeLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    NormalizeLayer<Dtype>* layer = (NormalizeLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void NormalizeLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    NormalizeLayer<Dtype>* layer = (NormalizeLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void NormalizeLayer<Dtype>::backwardTensor(void* instancePtr) {
    NormalizeLayer<Dtype>* layer = (NormalizeLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void NormalizeLayer<Dtype>::learnTensor(void* instancePtr) {
    NormalizeLayer<Dtype>* layer = (NormalizeLayer<Dtype>*)instancePtr;
    layer->update();
}

template<typename Dtype>
bool NormalizeLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t NormalizeLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {

    int size = 0;

    // buffer
    size += inputShape[0].C * inputShape[0].H * inputShape[0].W;

    // bufferChannel
    size += inputShape[0].C;

    // bufferSpatial
    size += inputShape[0].H * inputShape[0].W;

    // norm
	if (SLPROP(Normalize, acrossSpatial)) {
        size += inputShape[0].N;
	} else {
        size += inputShape[0].N * inputShape[0].H * inputShape[0].W;
	}

    // sumChannelMultiplier
    size += inputShape[0].C;

    // sumSpatialMultiplier
    size += inputShape[0].H * inputShape[0].W;

    return ALIGNUP(sizeof(Dtype) * size, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
}

template class NormalizeLayer<float>;
