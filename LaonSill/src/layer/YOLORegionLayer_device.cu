/**
 * @file YOLORegionLayer_device.cu
 * @date 2018-01-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "cuda_runtime.h"

#include "YOLORegionLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

#define EPSILON                 0.000001

template <typename Dtype>
__global__ void YoloRegionForward(const Dtype* input, int size,
        const int side1, const int side2, const int dim, int classNum,
        const Dtype* anchorVals, Dtype* output) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= size)
		return;

    int elemPerAnchorBox = classNum + 4;
    int anchorBoxCount = (int)(dim / elemPerAnchorBox);
    int gridCount = side1 * side2;
    int gridElemShift = gridCount * elemPerAnchorBox;

    int curBatch = idx / gridCount;
    int curGrid = idx % gridCount;

    for (int i = 0; i < anchorBoxCount; i++) {
        int inBoxIndex = gridElemShift * (curBatch * 5 + i) + curGrid;
        int outBoxIndex = idx * dim + i * elemPerAnchorBox;

        Dtype x1 = input[inBoxIndex + 0 * gridCount];
        Dtype y1 = input[inBoxIndex + 1 * gridCount];
        Dtype w1 = input[inBoxIndex + 2 * gridCount];
        Dtype h1 = input[inBoxIndex + 3 * gridCount];
        Dtype c1 = input[inBoxIndex + 4 * gridCount];

        output[outBoxIndex + 0] = 1.0 / (1.0 + expf((-1.0) * x1));
        output[outBoxIndex + 1] = 1.0 / (1.0 + expf((-1.0) * y1));

        output[outBoxIndex + 2] = 
            anchorVals[i * 2 + 0]  * expf(w1) / (Dtype)(side1);
        output[outBoxIndex + 3] = 
            anchorVals[i * 2 + 1] * expf(h1) / (Dtype)(side2);

        output[outBoxIndex + 4] = 1.0 / (1.0 + expf((-1.0) * c1));

        // softmax
        // exponential 함수에서 매우 큰값이 나오는 것을 막기 위해서..
        Dtype sum = 0.0;
        Dtype maxVal = input[inBoxIndex + (5 + 0) * gridCount];
        for (int j = 1; j < classNum - 1; j++) {
            if (input[inBoxIndex + (5 + j) * gridCount] > maxVal)
                maxVal = input[inBoxIndex + (5 + j) * gridCount];
        }

        for (int j = 0; j < classNum - 1; j++) {
            Dtype class1 = input[inBoxIndex + (5 + j) * gridCount] - maxVal;

            output[outBoxIndex + 5 + j] = expf(class1);
            sum += output[outBoxIndex + 5 + j];
        }

        for (int j = 0; j < classNum - 1; j++) {
            output[outBoxIndex + 5 + j] = output[outBoxIndex + 5 + j] / (sum + EPSILON);
        }
    }
}

template <typename Dtype>
__global__ void YoloRegionBackward(const Dtype* outputGrad, const Dtype* output, int size,
        const int side1, const int side2, const int dim, int classNum, Dtype* inputGrad) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    int elemPerAnchorBox = classNum + 4;
    int anchorBoxCount = (int)(dim / elemPerAnchorBox);
    int gridCount = side1 * side2;
    int gridElemShift = gridCount * elemPerAnchorBox;

    int curBatch = idx / gridCount;
    int curGrid = idx % gridCount;

    for (int i = 0; i < anchorBoxCount; i++) {
        int inBoxIndex = gridElemShift * (curBatch * 5 + i) + curGrid;
        int outBoxIndex = idx * dim + i * elemPerAnchorBox;

        Dtype x1 = output[outBoxIndex + 0];
        Dtype y1 = output[outBoxIndex + 1];
        Dtype w1 = output[outBoxIndex + 2];
        Dtype h1 = output[outBoxIndex + 3];
        Dtype c1 = output[outBoxIndex + 4];

        inputGrad[inBoxIndex + 0 * gridCount] = x1 * (1.0 - x1) *
            outputGrad[outBoxIndex + 0];
        inputGrad[inBoxIndex + 1 * gridCount] = y1 * (1.0 - y1) *
            outputGrad[outBoxIndex + 1];
        inputGrad[inBoxIndex + 2 * gridCount] = w1 * outputGrad[outBoxIndex + 2];
        inputGrad[inBoxIndex + 3 * gridCount] = h1 * outputGrad[outBoxIndex + 3];
        inputGrad[inBoxIndex + 4 * gridCount] = c1 * (1.0 - c1) * outputGrad[outBoxIndex + 4];

        for (int j = 0; j < classNum - 1; j++) {
            inputGrad[inBoxIndex + (5 + j) * gridCount] = 
                output[outBoxIndex + 5 + j] * (1.0 - output[outBoxIndex + 5 + j]) *
                outputGrad[outBoxIndex + 5 + j];
        }
    }
}


template <typename Dtype>
void YOLORegionLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({batches, rows, cols, channels});

    int classCount = SLPROP(YOLORegion, numClasses);
    SASSERT0(channels % (classCount + 4) == 0);

    uint32_t anchorBoxTwice = 2 * channels / (classCount + 4);
    this->anchorSet->reshape({1, 1, 1, anchorBoxTwice});

    Dtype* anchorData = (Dtype*)this->anchorSet->mutable_host_data();
    for (int i = 0; i < SLPROP(YOLORegion, anchors).size(); i++) {
        anchorData[i] = SLPROP(YOLORegion, anchors)[i];
    }

    for (int i = SLPROP(YOLORegion, anchors).size(); i < anchorBoxTwice; i++) {
        anchorData[i] = 0.5;
    }

}

template <typename Dtype>
void YOLORegionLayer<Dtype>::feedforward() {
	reshape();

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int gridCountElem = inputShape[1];
    int gridCountY = inputShape[2];
    int gridCountX = inputShape[3];
    int size = batchCount * gridCountX * gridCountY;

    const Dtype* anchorVals = this->anchorSet->device_data();
    const Dtype *inputData = this->_inputData[0]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();

    YoloRegionForward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, size, gridCountX, gridCountY, gridCountElem,
        SLPROP(YOLORegion, numClasses), anchorVals, outputData);
}

template <typename Dtype>
void YOLORegionLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int gridCountElem = inputShape[1];
    int gridCountY = inputShape[2];
    int gridCountX = inputShape[3];
    int size = batchCount * gridCountX * gridCountY;

    const Dtype *outputGrad = this->_outputData[0]->device_grad();
    const Dtype *outputData = this->_outputData[0]->device_data();
    Dtype *inputGrad = this->_inputData[0]->mutable_device_grad();

    YoloRegionBackward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        outputGrad, outputData, size, gridCountX, gridCountY, gridCountElem,
        SLPROP(YOLORegion, numClasses), inputGrad);
}

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* YOLORegionLayer<Dtype>::initLayer() {
	YOLORegionLayer* layer = NULL;
	SNEW(layer, YOLORegionLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool YOLORegionLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::backwardTensor(void* instancePtr) {
	YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool YOLORegionLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    if (SLPROP_BASE(input).size() != 1) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Region Layer should have only 1 input tensors but it has %d tensors",
            (int)SLPROP_BASE(input).size());
        return false;
    }
    if (SLPROP_BASE(output).size() != 1) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Region Layer should have only 1 output tensors but it has %d tensors",
            (int)SLPROP_BASE(output).size());
        return false;
    }

    if (inputShape[0].N <= 0 || inputShape[0].C <= 0 ||
           inputShape[0].H <= 0 || inputShape[0].W <= 0)
        return false;

    int classNumber = (int)(SLPROP(YOLORegion, numClasses));
    if (inputShape[0].C % (classNumber + 4) != 0) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Region Layer input channel should be multiple of (number of classes + 4). "
            "but input's channel is %d, number of classes is %d.",
            inputShape[0].C, classNumber);
        return false;
    }

    TensorShape outputShape1;
    outputShape1.N = inputShape[0].N;
    outputShape1.C = inputShape[0].H;
    outputShape1.H = inputShape[0].W;
    outputShape1.W = inputShape[0].C;

    if (outputShape1.N <= 0 || outputShape1.C <= 0 ||
           outputShape1.H <= 0 || outputShape1.W <= 0)
        return false;

    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t YOLORegionLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class YOLORegionLayer<float>;
