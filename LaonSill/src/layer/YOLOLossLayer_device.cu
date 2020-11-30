/**
 * @file YOLOLossLayer_device.cu
 * @date 2017-04-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "cuda_runtime.h"

#include "YOLOLossLayer.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

#define YOLOLOSSLAYER_LOG       1
#define EPSILON                 0.000001
#define NUM_THREADS             128

/**
 * tensor는 다음과 같이 구성이 됩니다.
 *
 * input[0] : Data
 *
 * +---+---+---+---+------------+---------+---------+-----+----------+
 * | x | y | w | h | confidence | class#0 | class#1 | ... | class#19 | -> batch=0, grid=0,0
 * +---+---+---+---+------------+---------+---------+-----+----------+    anchorbox=0
 * | x | y | w | h | confidence | class#0 | class#1 | ... | class#19 | -> batch=0, grid=0,0
 * +---+---+---+---+------------+---------+---------+-----+----------+    anchorbox=1
 * |                       ...                                       |
 * +---+---+---+---+------------+---------+---------+-----+----------+
 * | x | y | w | h | confidence | class#0 | class#1 | ... | class#19 | -> batch=7, grid=12,12
 * +---+---+---+---+------------+---------+---------+-----+----------+    anchorbox=4'
 *
 * input[1] : Label
 *
 * +----0----+------1------+------2------+---3--+--4---+---5--+---6--+-----7-----+
 * | item_id | group_label | instance_id | xmin | ymin | xmax | ymax | difficult |
 * +---------+-------------+-------------+------+------+------+------+-----------+
 *
 * output[0] tenser
 *
 * +------------------------+------------------------+------+---------------------------+
 * | batch#0 grid(0,0) loss | batch#0 grid(1,0) loss | ...  | batch#7 grid(12, 12) loss |
 * +------------------------+------------------------+------+---------------------------+
 */


// YOLO Loss Layer forward & Backward
// 소스가 너무 길다.. 정리해야 할꺼 같다.. 그런데 지금하기는 귀찮다.. ㅜㅜ
template <typename Dtype>
__global__ void YoloLoss(const Dtype* input, const Dtype* input2, int size, 
    const int side1, const int side2, const int dim, const int gtBoxCount,
    int classNum, Dtype noobjVal, Dtype coordVal, Dtype objVal, Dtype classVal,
    Dtype reward, Dtype iouThresh, Dtype* output, Dtype* inputGrad, Dtype* objIndex) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    int curBatch = (int)(idx / (side1 * side2));
    int curGrid = (int)(idx % (side1 * side2));

    int elemPerAnchorBox = classNum + 4;
    int anchorBoxCount = (int)(dim / elemPerAnchorBox);

    output[idx] = 0.0;

    for (int i = 0; i < anchorBoxCount; i++) {
        objIndex[i] = 0.0;
        // confidence backward
        int backBaseIndex = idx * dim + i * elemPerAnchorBox;
        for (int j = 0; j < elemPerAnchorBox; j++) {
            if (j == 4) {
                Dtype c1 = input[backBaseIndex + j];
                inputGrad[backBaseIndex + j] = noobjVal * (c1 - 0.0);
            } else {
                inputGrad[backBaseIndex + j] = 0.0;
            }
        }
    }

    for (int t = 0; t < gtBoxCount; t++) {
        // ground truth Box

        // batch check
        int labelBatch = input2[t * 8 + 0];
        if (labelBatch != curBatch)
            continue;

        // background check
        int labelClass = input2[t * 8 + 1];
        if (labelClass == 0)
            continue;

        // grid check
        Dtype minX = input2[t * 8 + 3];
        Dtype minY = input2[t * 8 + 4];
        Dtype maxX = input2[t * 8 + 5];
        Dtype maxY = input2[t * 8 + 6];

        Dtype normedX = (maxX + minX) / 2.0;
        Dtype normedY = (maxY + minY) / 2.0;

        int gridX = (int)(normedX * side1);
        int gridY = (int)(normedY * side2);

        if (curGrid != gridX * side1 + gridY)
            continue;

        Dtype x = normedX * (Dtype)(side1) - gridX;
        Dtype y = normedY * (Dtype)(side2) - gridY;
        Dtype w = (maxX - minX) * (Dtype)(side1);
        Dtype h = (maxY - minY) * (Dtype)(side2);

        // anchor Boxes
        int bestBoxIndex = 0;
        Dtype bestBoxIOU = 0.0;
        Dtype bestDist = -1.0;

        for (int i = 0; i < anchorBoxCount; i++) {
            if (objIndex[i] == 1.0)
                continue;

            int boxBaseIndex = idx * dim + i * elemPerAnchorBox;

            Dtype x1 = input[boxBaseIndex + 0];
            Dtype y1 = input[boxBaseIndex + 1];
            Dtype w1 = input[boxBaseIndex + 2] * side1;
            Dtype h1 = input[boxBaseIndex + 3] * side2;

            // calc box iou
            Dtype left = max(x1 - w1 / 2.0, x - w / 2.0);
            Dtype right = min(x1 + w1 / 2.0, x + w / 2.0);
            Dtype top = max(y1 - h1 / 2.0, y - h / 2.0);
            Dtype bottom = min(y1 + h1 / 2.0, y + h / 2.0);

            Dtype ov_w = right - left;
            Dtype ov_h = bottom - top;

            Dtype b_inter;
            if (ov_w <= 0 || ov_h <= 0)
                b_inter = 0.0;
            else
                b_inter = ov_w * ov_h;

            Dtype b_union;
            b_union = w1 * h1 + w * h - b_inter;

            Dtype box_iou = 0.0;
            if (b_union > 0)
                box_iou = b_inter / b_union;

            if (box_iou > bestBoxIOU) {
                bestBoxIndex = i;
                bestBoxIOU = box_iou;
            } else if (bestBoxIOU == 0.0) {
                Dtype box_dist = (x - x1) * (x - x1) + (y - y1) * (y - y1) +
                    (w - w1) * (w - w1) + (h - h1) * (h - h1);
                if (bestDist == -1.0) {
                    bestBoxIndex = i;
                    bestDist = box_dist;
                } else if (box_dist < bestDist) {
                    bestBoxIndex = i;
                    bestDist = box_dist;
                }
            }
        }

        if (bestBoxIOU > reward)
            reward = bestBoxIOU;

        for (int i = 0; i < anchorBoxCount; i++) {

            if (objIndex[i] == 1.0)
                continue;

            int boxBaseIndex = idx * dim + i * elemPerAnchorBox;
            Dtype x1 = input[boxBaseIndex + 0];
            Dtype y1 = input[boxBaseIndex + 1];
            Dtype w1 = input[boxBaseIndex + 2] * side1;
            Dtype h1 = input[boxBaseIndex + 3] * side2;
            Dtype c1 = input[boxBaseIndex + 4];

            if (bestBoxIOU > reward) {
                inputGrad[boxBaseIndex + 4] = 0.0;
            }

            if (bestBoxIndex != i)
                continue;

            objIndex[i] = 1.0;

            // forward coords, confidence
            output[idx] = output[idx] + coordVal * (x1 - x) * (x1 - x);
            output[idx] = output[idx] + coordVal * (y1 - y) * (y1 - y);
            output[idx] = output[idx] + coordVal * (sqrtf(w1) - sqrtf(w)) * \
                          (sqrtf(w1) - sqrtf(w)) / (Dtype)(side1);
            output[idx] = output[idx] + coordVal * (sqrtf(h1) - sqrtf(h)) * \
                          (sqrtf(h1) - sqrtf(h)) / (Dtype)(side2);
            output[idx] = output[idx] + objVal * (c1 - reward) * (c1 - reward);

            for (int j = 0; j < classNum - 1; j++) {
                if (j == labelClass - 1) {
                    // forward class
                    output[idx] = output[idx] + classVal * \
                  (input[boxBaseIndex + 5 + j] - 1.0) * (input[boxBaseIndex + 5 + j] - 1.0);
                    // backward class
                    inputGrad[boxBaseIndex + 5 + j] = classVal *
                        (input[boxBaseIndex + 5 + j] - 1.0);
                } else {
                    // forward class
                    output[idx] = output[idx] + classVal * \
                  (input[boxBaseIndex + 5 + j] - 0.0) * (input[boxBaseIndex + 5 + j] - 0.0);
                    // backward class
                    inputGrad[boxBaseIndex + 5 + j] = classVal *
                        (input[boxBaseIndex + 5 + j] - 0.0);
                }
            }

            // backward coords, confidence
            inputGrad[boxBaseIndex + 0] = coordVal * (x1 - x);
            inputGrad[boxBaseIndex + 1] = coordVal * (y1 - y);
            inputGrad[boxBaseIndex + 2] = coordVal * log(w1 / w);
            inputGrad[boxBaseIndex + 3] = coordVal * log(h1 / h);
            inputGrad[boxBaseIndex + 4] = objVal * (c1 - reward);

        }
    }

    // no obj confidence forward
    for (int b = 0; b < anchorBoxCount; b++) {
        if (objIndex[b] == 1.0)
            continue;

        int noObjBaseIdx = idx * dim + b * elemPerAnchorBox;
        Dtype c1 = input[noObjBaseIdx + 4];
        output[idx] = output[idx] + noobjVal * (c1 - 0.0) * (c1 - 0.0);
    }
}

template <typename Dtype>
YOLOLossLayer<Dtype>::YOLOLossLayer() : LossLayer<Dtype>() {
	this->type = Layer<Dtype>::YOLOLoss;

    SNEW(this->objIdxVec, Data<Dtype>, SLPROP_BASE(name) + "_objIdxVec");
    SASSUME0(this->objIdxVec != NULL);
}

template<typename Dtype>
YOLOLossLayer<Dtype>::~YOLOLossLayer() {
    SFREE(this->objIdxVec);

}

template <typename Dtype>
void YOLOLossLayer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
        const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
        const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
        assert(count == inputDataCount);
    }

    if (!Layer<Dtype>::_isInputShapeChanged(0))
        return;

    SASSERT0(this->_inputData.size() == 2);

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputShape;
	this->_outputData[0]->reshape(this->_inputShape[0]);

    const vector<uint32_t>& inputShape2 = this->_inputData[1]->getShape();
	this->_inputShape[1] = inputShape2;

    uint32_t channels = inputShape[3];

    uint32_t anchorElement = SLPROP(YOLOLoss, numClasses) + 4;
    uint32_t anchorNumber = channels / anchorElement;
    this->objIdxVec->reshape({1, 1, 1, anchorNumber});

    Dtype* objIdxData = (Dtype*)this->objIdxVec->mutable_host_data();
    for (int i = 0; i < anchorNumber; i++) {
        objIdxData[i] = 0.0;
    }

	STDOUT_COND_LOG(YOLOLOSSLAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
	STDOUT_COND_LOG(YOLOLOSSLAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
}

template <typename Dtype>
void YOLOLossLayer<Dtype>::feedforward() {
	reshape();

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    const int gridCountY = inputShape[1];
    const int gridCountX = inputShape[2];
    const int gridCountElem = inputShape[3];
    int size = batchCount * gridCountX * gridCountY;

    const vector<uint32_t>& labelShape = this->_inputData[1]->getShape();
    const int groundTruthBox = labelShape[2];

    const Dtype *inputData = this->_inputData[0]->device_data();
    const Dtype *inputData2 = this->_inputData[1]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();
    Dtype *inputGrad = this->_inputData[0]->mutable_device_grad();

    Dtype* objIndex = this->objIdxVec->mutable_device_data();

    int blockNum = (size + NUM_THREADS - 1) / NUM_THREADS;
    YoloLoss<Dtype><<<blockNum, NUM_THREADS>>>(
        inputData, inputData2, size, gridCountX, gridCountY, gridCountElem,
        groundTruthBox, (int)SLPROP(YOLOLoss, numClasses),
        (Dtype)SLPROP(YOLOLoss, noobj), (Dtype)SLPROP(YOLOLoss, coord),
        (Dtype)SLPROP(YOLOLoss, obj), (Dtype)SLPROP(YOLOLoss, class),
        (Dtype)SLPROP(YOLOLoss, reward), (Dtype)SLPROP(YOLOLoss, iouThresh),
        outputData, inputGrad, objIndex);
}

template <typename Dtype>
void YOLOLossLayer<Dtype>::backpropagation() {
}

template <typename Dtype>
Dtype YOLOLossLayer<Dtype>::cost() {
    const Dtype* outputData = this->_outputData[0]->host_data();
    Dtype avg = 0.0;

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int gridCountY = inputShape[1];
    int gridCountX = inputShape[2];
    int count = batchCount * gridCountX * gridCountY;


    for (int i = 0; i < count; i++) {
        avg += outputData[i];
    }
	return avg / (Dtype)batchCount;
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* YOLOLossLayer<Dtype>::initLayer() {
	YOLOLossLayer* layer = NULL;
	SNEW(layer, YOLOLossLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;

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
bool YOLOLossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::backwardTensor(void* instancePtr) {
    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool YOLOLossLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    if (SLPROP_BASE(input).size() != 2) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Region Layer should have 2 input tensors but it has %d tensors",
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

    TensorShape outputShape1;
    outputShape1.N = inputShape[0].N;
    outputShape1.C = inputShape[0].C;
    outputShape1.H = inputShape[0].H;
    outputShape1.W = inputShape[0].W;

    if (outputShape1.N <= 0 || outputShape1.C <= 0 ||
           outputShape1.H <= 0 || outputShape1.W <= 0)
        return false;

    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t YOLOLossLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class YOLOLossLayer<float>;
