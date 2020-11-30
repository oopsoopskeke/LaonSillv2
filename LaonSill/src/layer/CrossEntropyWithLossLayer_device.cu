/**
 * @file CrossEntropyWithLossLayer_device.cu
 * @date 2017-02-06
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "CrossEntropyWithLossLayer.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

#define CROSSENTROPYWITHLOSSLAYER_LOG   0

using namespace std;

// Cross Entropy : 
//  z * -log(x) + (1 - z) * -log(1 - x)
//  x : input
template <typename Dtype>
__global__ void CEForward(const Dtype* input, Dtype z, int size, Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x;
    if (input[idx] < 0.00001)
        x = 0.00001;
    else if (input[idx] > 0.99999)
        x = 0.99999;
    else
        x = input[idx];

    output[idx] += z * logf(x) + (1 - z) * logf(1 - x);
}

template <typename Dtype>
__global__ void CEForward2(const Dtype* input, const Dtype* input2, int size, Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x;
    Dtype z = input2[idx];
    if (input[idx] < 0.00001)
        x = 0.00001;
    else if (input[idx] > 0.99999)
        x = 0.99999;
    else
        x = input[idx];

    output[idx] = (-1.0) * (z * logf(x) + (1 - z) * logf(1 - x));
}

// Cross Entropy with logit(sigmoid): 
//  Loss : x - x * z + log (1 + exp(-x))        ....    x >= 0
//         -x * z + log(1 + exp(x))             ....    x < 0
//  x : input, z : target
template <typename Dtype>
__global__ void CEForwardWithSigmoid(const Dtype* input, Dtype z, int size, Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x;

    x = input[idx];
    if (x < 0) { 
        output[idx] = ((-1.0) * x * z + logf(1 + expf(x)));
    } else {
        output[idx] = (x - x * z + logf(1 + expf( (-1.0) * x)));
    }
}

// Cross Entropy with logit(sigmoid): 
//  Loss : x - x * z + log (1 + exp(-x))        ....    x >= 0
//         -x * z + log(1 + exp(x))             ....    x < 0
//  x : input, z : target
template <typename Dtype>
__global__ void CEForwardWithSigmoid2(const Dtype* input, const Dtype* input2, int size,
    Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x;
    Dtype z;

    x = input[idx];
    z = input2[idx];

    if (x < 0) { 
        output[idx] = ((-1.0) * x * z + logf(1 + expf(x)));
    } else {
        output[idx] = (x - x * z + logf(1 + expf( (-1.0) * x)));
    }
}

// gradient = x - z
// x : input
template <typename Dtype>
__global__ void CEBackward(const Dtype* input, const Dtype z, int size, Dtype* gradient) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    gradient[idx] = input[idx] - z;
}

// gradient = x - z
// x : input
template <typename Dtype>
__global__ void CEBackward2(const Dtype* input, const Dtype* input2, int size,
    Dtype* gradient) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    gradient[idx] = input[idx] - input2[idx];
}

// gradient : 1 - z - exp(-x) / (1 + exp(-x))   ....   x >= 0
//            -z + exp(x) / (1 + exp(x))        ....   x < 0 
// x : input
template <typename Dtype>
__global__ void CEBackwardWithSigmoid(const Dtype* input, const Dtype z, int size,
    Dtype* gradient) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x = input[idx];

    if (input[idx] < 0) {
        gradient[idx] = ((-1.0) * z + expf(x) / (1 + expf(x)));
    } else {
        gradient[idx] = (1 - z - expf((-1.0) * x) / (1 + expf((-1.0) * x)));
    }
}

// gradient : 1 - z - exp(-x) / (1 + exp(-x))   ....   x >= 0
//            -z + exp(x) / (1 + exp(x))        ....   x < 0 
// x : input
template <typename Dtype>
__global__ void CEBackwardWithSigmoid2(const Dtype* input, const Dtype* input2, int size,
    Dtype* gradient) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x = input[idx];
    Dtype z = input2[idx];

    if (input[idx] < 0) {
        gradient[idx] = ((-1.0) * z + expf(x) / (1 + expf(x)));
    } else {
        gradient[idx] = (1 - z - expf((-1.0) * x) / (1 + expf((-1.0) * x)));
    }
}

template <typename Dtype>
CrossEntropyWithLossLayer<Dtype>::CrossEntropyWithLossLayer()
	: LossLayer<Dtype>() {
	this->type = Layer<Dtype>::CrossEntropyWithLoss;
    this->depth = 0;
}

template<typename Dtype>
CrossEntropyWithLossLayer<Dtype>::~CrossEntropyWithLossLayer() {

}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
        const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
        const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
        assert(count == inputDataCount);
    }

    if (!Layer<Dtype>::_isInputShapeChanged(0))
        return;

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();

	uint32_t batches = inputShape[0];
	uint32_t channels = inputShape[1];
	uint32_t rows = inputShape[2];
	uint32_t cols = inputShape[3];
    uint32_t depth = this->_inputData[0]->getCountByAxis(1);

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({batches, 1, depth, 1});

	STDOUT_COND_LOG(CROSSENTROPYWITHLOSSLAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);
	STDOUT_COND_LOG(CROSSENTROPYWITHLOSSLAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);

    if (this->depth == 0) {
        this->depth = depth;
    } else {
        SASSERT(this->depth == depth, "old depth=%d, depth=%d", this->depth, depth);
    }

    SASSERT0(this->_inputData.size() <= 2);
    if (this->_inputData.size() == 2) {
        // target value가 아닌 target values가 지정이 된 경우 
        const vector<uint32_t>& inputShape2 = this->_inputData[1]->getShape();
        SASSERT0(inputShape2[0] == inputShape[0]);
        SASSERT0(this->depth == this->_inputData[0]->getCountByAxis(1));
    }
}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::feedforward() {
	reshape();

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];

    const Dtype *inputData = this->_inputData[0]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();

    int count = this->depth * batchCount;

    if (SLPROP(CrossEntropyWithLoss, withSigmoid)) {
        if (this->_inputData.size() == 2) {
            const Dtype *inputData2 = this->_inputData[1]->device_data();
            CEForwardWithSigmoid2<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, inputData2, count, outputData);
        } else {
            CEForwardWithSigmoid<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, (Dtype)SLPROP(CrossEntropyWithLoss, targetValue), count,
                outputData);
        }
    } else {
        if (this->_inputData.size() == 2) {
            const Dtype *inputData2 = this->_inputData[1]->device_data();
            CEForward2<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, inputData2, count, outputData);
        } else {
            CEForward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, (Dtype)SLPROP(CrossEntropyWithLoss, targetValue), count,
                outputData);
        }
    }
}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];

    const Dtype *inputData = this->_inputData[0]->device_data();
	Dtype* inputGrads = this->_inputData[0]->mutable_device_grad();

    int count = batchCount * this->depth;

    if (SLPROP(CrossEntropyWithLoss, withSigmoid)) {
        if (this->_inputData.size() == 2) {
            const Dtype *inputData2 = this->_inputData[1]->device_data();
            CEBackwardWithSigmoid2<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, inputData2, count, inputGrads);
        } else {
            CEBackwardWithSigmoid<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, (Dtype)SLPROP(CrossEntropyWithLoss, targetValue), count,
                inputGrads);
        }
    } else {
        if (this->_inputData.size() == 2) {
            const Dtype *inputData2 = this->_inputData[1]->device_data();
            CEBackward2<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, inputData2, count, inputGrads);
        } else {
            CEBackward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, (Dtype)SLPROP(CrossEntropyWithLoss, targetValue), count,
                inputGrads);
        }

    }
}

template <typename Dtype>
Dtype CrossEntropyWithLossLayer<Dtype>::cost() {
    const Dtype* outputData = this->_outputData[0]->host_data();
    int batchCount = (int)this->_inputShape[0][0];

    Dtype avg = 0.0;
    for (int i = 0; i < this->depth * batchCount; i++) {
        avg += outputData[i];
    }
	return avg / ((Dtype)this->depth * (Dtype)batchCount);
}

template<typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::setTargetValue(Dtype value) {
    SLPROP(CrossEntropyWithLoss, targetValue) = value;
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* CrossEntropyWithLossLayer<Dtype>::initLayer() {
	CrossEntropyWithLossLayer* layer = NULL;
	SNEW(layer, CrossEntropyWithLossLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::destroyLayer(void* instancePtr) {
    CrossEntropyWithLossLayer<Dtype>* layer = (CrossEntropyWithLossLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

    CrossEntropyWithLossLayer<Dtype>* layer = (CrossEntropyWithLossLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(index < 2);
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(index == 0);
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool CrossEntropyWithLossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    CrossEntropyWithLossLayer<Dtype>* layer = (CrossEntropyWithLossLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    CrossEntropyWithLossLayer<Dtype>* layer = (CrossEntropyWithLossLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::backwardTensor(void* instancePtr) {
    CrossEntropyWithLossLayer<Dtype>* layer = (CrossEntropyWithLossLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool CrossEntropyWithLossLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if(inputShape.size() != 2) {
        return false;
    }

	uint32_t batches = inputShape[0].N;
	uint32_t channels = inputShape[0].C;
	uint32_t rows = inputShape[0].H;
	uint32_t cols = inputShape[0].W;
    uint32_t depth = channels * rows * cols;

    TensorShape outputShape1;
    outputShape1.N = batches;
    outputShape1.C = 1;
    outputShape1.H = depth;
    outputShape1.W = 1;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t CrossEntropyWithLossLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class CrossEntropyWithLossLayer<float>;
