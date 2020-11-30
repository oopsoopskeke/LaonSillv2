/**
 * @file YOLOPassThruLayer_device.cu
 * @date 2018-01-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "YOLOPassThruLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

#define EPSILON                 0.000001

template <typename Dtype>
__global__ void reorgKernel(const Dtype* input, int size, int channels, int rows, int cols,
        int stride, bool forward, Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    int curBatch = idx / channels;
    int curChannel = idx % channels;

    int topChannels = channels / (stride * stride);
    int topRows = rows * stride;
    int topCols = cols * stride;

    for (int h = 0; h < cols; h++) {
        for (int w = 0; w < rows; w++) {
            int bottomIndex = w + rows * (h + cols * (curChannel + channels * curBatch));
            int c2 = curChannel % topChannels;
            int offset = curChannel / topChannels;
            int w2 = w * stride + offset % stride;
            int h2 = h * stride + offset / stride;
            int topIndex = w2 + topRows * (h2 + topCols * (c2 + topChannels * curBatch));
            if (forward)
                output[topIndex] = input[bottomIndex];
            else
                output[bottomIndex] = input[topIndex];
        }
    }
}

template <typename Dtype>
void YOLOPassThruLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	this->_inputShape[0] = {batches, channels, rows, cols};

    int strideInt = SLPROP(YOLOPassThru, stride);
    bool reverseBool = SLPROP(YOLOPassThru, stride);

    if (reverseBool) {
        SASSERT0(rows % strideInt == 0);
        SASSERT0(cols % strideInt == 0);

        this->_outputData[0]->reshape({batches, uint32_t(channels * strideInt * strideInt),
                uint32_t(rows / strideInt), uint32_t(cols / strideInt)});
    } else {
        SASSERT0(channels % (strideInt * strideInt) == 0);

        this->_outputData[0]->reshape({batches, uint32_t(channels / strideInt / strideInt),
                uint32_t(rows * strideInt), uint32_t(cols * strideInt)});
    }
}

template <typename Dtype>
void YOLOPassThruLayer<Dtype>::feedforward() {
	reshape();

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int channelCount = inputShape[1];
    int rowCount = inputShape[2];
    int colCount = inputShape[3];
    int size = batchCount * channelCount;

    const Dtype *inputData = this->_inputData[0]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();

    int strideInt = SLPROP(YOLOPassThru, stride);
    bool reverseBool = SLPROP(YOLOPassThru, reverse);

    reorgKernel<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, size, channelCount, rowCount, colCount, strideInt, reverseBool, outputData);
}

template <typename Dtype>
void YOLOPassThruLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int channelCount = inputShape[1];
    int rowCount = inputShape[2];
    int colCount = inputShape[3];

    int size = batchCount * channelCount;

    const Dtype *outputGrad = this->_outputData[0]->device_grad();
    Dtype *inputGrad = this->_inputData[0]->mutable_device_grad();

    int strideInt = SLPROP(YOLOPassThru, stride);
    bool reverseBool = SLPROP(YOLOPassThru, reverse);

    int outChannels = channelCount * strideInt * strideInt;
    int outRows = rowCount / strideInt;
    int outCols = colCount / strideInt;

    reorgKernel<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        outputGrad, size, outChannels, outRows, outCols, strideInt, !reverseBool, inputGrad);
}

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* YOLOPassThruLayer<Dtype>::initLayer() {
	YOLOPassThruLayer* layer = NULL;
	SNEW(layer, YOLOPassThruLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool YOLOPassThruLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::backwardTensor(void* instancePtr) {
	YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool YOLOPassThruLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    /* reverse가 false일 때 row, col이 stride의 배수여야 함
       reverse가 true일 때 channel이 stride의 제곱의 배수여야 함
       */

    // input tensor 1개
    if (SLPROP_BASE(input).size() != 1) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Passthrough Layer should have only 1 input tensors but it has %d tensors",
            (int)SLPROP_BASE(input).size());
        return false;
    }

    // output tensor 1개
    if (SLPROP_BASE(output).size() != 1) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Passthrough Layer should have only 1 output tensors but it has %d tensors",
            (int)SLPROP_BASE(output).size());
        return false;
    }

    if (inputShape[0].N <= 0 || inputShape[0].C <= 0 ||
            inputShape[0].H <= 0 || inputShape[0].W <= 0)
        return false;

    const int strideInt = SLPROP(YOLOPassThru, stride);
    const bool reverseBool = SLPROP(YOLOPassThru, reverse);

    int outChannel;
    int outHeight;
    int outWidth;

    if (reverseBool) {
        if (inputShape[0].H % strideInt != 0 || inputShape[0].W % strideInt != 0) {
            SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Passthrough Layer input tensor's rows and cols should be multiple of "
            "stride if reverse case. but rows=%d, cols=%d and stride=%d.",
            (int)inputShape[0].W, (int)inputShape[0].H, strideInt);
            return false;
        } else {
            outChannel = inputShape[0].C * (strideInt * strideInt);
            outHeight = inputShape[0].H / strideInt;
            outWidth = inputShape[0].W / strideInt;
        }

    } else {
        if (inputShape[0].C % (strideInt * strideInt) != 0) {
            SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Passthrough Layer input tensor's channels should be multiple of "
            "(stride ** 2) if not reverse case. but channels=%d and stride=%d.",
            (int)inputShape[0].C, strideInt);
            return false;
        } else {
            outChannel = inputShape[0].C / (strideInt * strideInt);
            outHeight = inputShape[0].H * strideInt;
            outWidth = inputShape[0].W * strideInt;
        }
    }

    if (outChannel <= 0 || outHeight <= 0 || outWidth <= 0)
        return false;

    TensorShape outputShape1;
    outputShape1.N = inputShape[0].N;
    outputShape1.C = outChannel;
    outputShape1.H = outHeight;
    outputShape1.W = outWidth;

    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t YOLOPassThruLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class YOLOPassThruLayer<float>;
