/*
 * SoftmaxLayer.cpp
 *
 *  Created on: Nov 29, 2016
 *      Author: jkim
 */

#include <vector>
#include <cfloat>

#include "SoftmaxLayer.h"
//#include "ActivationFactory.h"
#include "MathFunctions.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

#if 1
#define SOFTMAXLAYER_LOG 0


using namespace std;

template <typename Dtype>
int SoftmaxLayer<Dtype>::INNER_ID = 10010;


template <typename Dtype>
SoftmaxLayer<Dtype>::SoftmaxLayer()
: SoftmaxLayer(NULL) {}


template <typename Dtype>
SoftmaxLayer<Dtype>::SoftmaxLayer(_SoftmaxPropLayer* prop)
: Layer<Dtype>(), sumMultiplier("sumMultiplier"), scale("scale") {
	this->type = Layer<Dtype>::Softmax;
	if (prop) {
		this->prop = NULL;
		SNEW(this->prop, _SoftmaxPropLayer);
		SASSUME0(this->prop != NULL);
		*(this->prop) = *(prop);
	} else {
		this->prop = NULL;
	}

	checkCUDNN(cudnnCreateTensorDescriptor(&this->inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputTensorDesc));
}


template <typename Dtype>
SoftmaxLayer<Dtype>::~SoftmaxLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(this->inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputTensorDesc));

    if (this->prop != NULL)
        SFREE(this->prop);
}



template <typename Dtype>
void SoftmaxLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputDataShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputDataShape;

	// "prob"
	this->_outputData[0]->reshape(inputDataShape);

#if SOFTMAXLAYER_LOG
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			GET_PROP(prop, Softmax, name).c_str(), inputDataShape[0], inputDataShape[1],
			inputDataShape[2], inputDataShape[3]);
#endif


	//const uint32_t softmaxAxis = SLPROP(Softmax, softmaxAxis);
	const uint32_t softmaxAxis = GET_PROP(prop, Softmax, softmaxAxis);

	//vector<uint32_t> multDims(1, inputDataShape[softmaxAxis]);
	this->sumMultiplier.reshape({1, 1, 1, inputDataShape[softmaxAxis]});
	Dtype* multiplierData = this->sumMultiplier.mutable_host_data();
	soooa_set(this->sumMultiplier.getCount(), Dtype(1), multiplierData);

	this->outerNum = this->_inputData[0]->getCountByAxis(0, softmaxAxis);
	this->innerNum = this->_inputData[0]->getCountByAxis(softmaxAxis+1);

	vector<uint32_t> scaleDims = this->_inputData[0]->getShape();
	scaleDims[softmaxAxis] = 1;
	this->scale.reshape(scaleDims);


	//============================================================================

	const uint32_t batches = this->outerNum;
	const uint32_t channels = this->_inputData[0]->getShape(softmaxAxis);
	const uint32_t rows = this->innerNum;
	const uint32_t cols = 1;

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));


}

template <typename Dtype>
void SoftmaxLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* inputData = this->_inputData[0]->device_data();
	Dtype* outputData = this->_outputData[0]->mutable_device_data();

	checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			&Cuda::alpha,
			this->inputTensorDesc, inputData,
			&Cuda::beta,
			this->outputTensorDesc, outputData));
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::backpropagation() {
	const vector<bool>& propDown = GET_PROP(prop, Softmax, propDown);
	if (propDown[0]) {
		const Dtype* outputData = this->_outputData[0]->device_data();
		const Dtype* outputGrad = this->_outputData[0]->device_grad();
		Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();

		checkCUDNN(cudnnSoftmaxBackward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_CHANNEL,
				&Cuda::alpha,
				this->outputTensorDesc, outputData, this->outputTensorDesc, outputGrad,
				&Cuda::beta,
				this->inputTensorDesc, inputGrad));
	}
}

/*
template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}




template <typename Dtype>
SoftmaxLayer<Dtype>::SoftmaxLayer()
	: Layer<Dtype>() {
	initialize();
}

template <typename Dtype>
SoftmaxLayer<Dtype>::SoftmaxLayer(Builder* builder)
	: Layer<Dtype>(builder) {
	this->softmaxAxis = builder->_softmaxAxis;

	initialize();
}


template <typename Dtype>
SoftmaxLayer<Dtype>::~SoftmaxLayer() {
	//ActivationFactory<Dtype>::destory(activation_fn);

	//checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	//checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
}



template <typename Dtype>
void SoftmaxLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputDataShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputDataShape;

	// "prob"
	this->_outputData[0]->reshape(inputDataShape);
	//vector<uint32_t> multDims(1, inputDataShape[this->softmaxAxis]);
	this->sumMultiplier.reshape({1, 1, 1, inputDataShape[this->softmaxAxis]});
	Dtype* multiplierData = this->sumMultiplier.mutable_host_data();
	soooa_set(this->sumMultiplier.getCount(), Dtype(1), multiplierData);

	this->outerNum = this->_inputData[0]->getCountByAxis(0, this->softmaxAxis);
	this->innerNum = this->_inputData[0]->getCountByAxis(this->softmaxAxis+1);

	vector<uint32_t> scaleDims = this->_inputData[0]->getShape();
	scaleDims[this->softmaxAxis] = 1;
	this->scale.reshape(scaleDims);


	//============================================================================
	const uint32_t batches = inputDataShape[0];
	const uint32_t channels = inputDataShape[1];
	const uint32_t rows = inputDataShape[2];
	const uint32_t cols = inputDataShape[3];

#if 0
	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));
#endif

#if SOFTMAXLAYER_LOG
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			SLPROP_BASE(name).c_str(), batches, channels, rows, cols);
#endif
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* inputData = this->_inputData[0]->device_data();
	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	Dtype* scaleData = this->scale.mutable_device_data();

	const uint32_t count = this->_inputData[0]->getCount();
	const uint32_t channels = this->_outputData[0]->getShape(this->softmaxAxis);
	soooa_copy(count, inputData, outputData);
	// We need to subtract the max to avoid nemerical issues, compute the exp,
	// and then normalize.

	// compute max
	kernel_channel_max<Dtype><<<SOOOA_GET_BLOCKS(this->outerNum * this->innerNum),
			SOOOA_CUDA_NUM_THREADS>>>(this->outerNum, channels, this->innerNum, outputData,
			scaleData);
	// subtract
	kernel_channel_subtract<Dtype><<<SOOOA_GET_BLOCKS(count),
			SOOOA_CUDA_NUM_THREADS>>>(count, this->outerNum, channels, this->innerNum,
			scaleData, outputData);
	// exponentiate
	kernel_exp<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
			count, outputData, outputData);
	// sum after exp
	kernel_channel_sum<Dtype><<<SOOOA_GET_BLOCKS(this->outerNum * this->innerNum),
			SOOOA_CUDA_NUM_THREADS>>>(this->outerNum, channels, this->innerNum, outputData,
			scaleData);
	// divide
	kernel_channel_div<Dtype><<<SOOOA_GET_BLOCKS(count),
			SOOOA_CUDA_NUM_THREADS>>>(count, this->outerNum, channels, this->innerNum,
					scaleData, outputData);

}

template <typename Dtype>
void SoftmaxLayer<Dtype>::backpropagation() {
	const Dtype* outputGrad = this->_outputData[0]->device_grad();
	const Dtype* outputData = this->_outputData[0]->device_data();
	Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
	Dtype* scaleData = this->scale.mutable_device_data();

	const uint32_t count = this->_outputData[0]->getCount();
	const uint32_t channels = this->_outputData[0]->getShape(this->softmaxAxis);
	oooa_copy(count, outputData, inputGrad);
	// Compute inner1d(outputGrad, outputData) and subtract them from the bottom diff.
	kernel_channel_dot<Dtype><<<SOOOA_GET_BLOCKS(this->outerNum * this->innerNum),
			SOOOA_CUDA_NUM_THREADS>>>(this->outerNum, channels, this->innerNum,
			outputGrad, outputData, scaleData);

	kernel_channel_subtract<Dtype><<<SOOOA_GET_BLOCKS(count),
			SOOOA_CUDA_NUM_THREADS>>>(count, this->outerNum, channels, this->innerNum,
			scaleData, inputGrad);

	// elementwise multiplication
	soooa_gpu_mul<Dtype>(this->_outputData[0]->getCount(), inputGrad, outputData, inputGrad);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::Softmax;

	//this->activation_fn = ActivationFactory<Dtype>::create(Activation<Dtype>::Softmax);

	this->sumMultiplier = new Data<Dtype>("sumMultiplier");
	this->scale = new Data<Dtype>("scale");

	//checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	//checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
}
*/



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* SoftmaxLayer<Dtype>::initLayer() {
	SoftmaxLayer* layer = NULL;
	SNEW(layer, SoftmaxLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::destroyLayer(void* instancePtr) {
    SoftmaxLayer<Dtype>* layer = (SoftmaxLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    SoftmaxLayer<Dtype>* layer = (SoftmaxLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool SoftmaxLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    SoftmaxLayer<Dtype>* layer = (SoftmaxLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    SoftmaxLayer<Dtype>* layer = (SoftmaxLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::backwardTensor(void* instancePtr) {
    SoftmaxLayer<Dtype>* layer = (SoftmaxLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool SoftmaxLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t SoftmaxLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
	const uint32_t softmaxAxis = SLPROP(Softmax, softmaxAxis);

    const int sumMultiplierSize = tensorValByIndex(inputShape[0], softmaxAxis);
    size_t size = 0;
    size += ALIGNUP(sizeof(Dtype) * sumMultiplierSize, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    int scaleSize = 1;
    for (int i = 0; i < Data<Dtype>::SHAPE_SIZE; i++) {
        if (i != softmaxAxis) {
            scaleSize *= tensorValByIndex(inputShape[0], i);
        }
    } 

    size += ALIGNUP(sizeof(Dtype) * scaleSize, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    return size;
}

template class SoftmaxLayer<float>;

#endif
