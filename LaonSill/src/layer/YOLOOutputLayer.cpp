 /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * @ file      YOLOOutputLayer.cpp
 * @ date      2018-02-06
 * @ author    SUN
 * @ brief
 * @ details
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <vector>
#include <array>

#include "YOLOOutputLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"
#include "ImageUtil.h"

#include "frcnn_common.h"

using namespace std;

/*******************************************************************************************
 * YOLO Output layer : YOLO 커스텀 데이터 inference 지원과 mAP계산을 위한 레이어.
 *
 *  << input tensor format >>
 *  +---+---+---+---+---+-------------+-----+-------------+
 *  | x | y | w | h | c | class0 prob | ... | classN prob | -> batch0, grid(0,0), anchor#0
 *  +---+---+---+---+---+-------------+-----+-------------+
 *  | x | y | w | h | c | class0 prob | ... | classN prob | -> batch0, grid(0,0), anchor#1
 *  +---+---+---+---+---+-------------+-----+-------------+
 *  |                     ...                             |
 *  +---+---+---+---+---+-------------+-----+-------------+
 *  | x | y | w | h | c | class0 prob | ... | classN prob | -> batch0, grid(1,0), anchor#0
 *  +---+---+---+---+---+-------------+-----+-------------+
 *  |                     ...                             |
 *  +---+---+---+---+---+-------------+-----+-------------+
 *  | x | y | w | h | c | class0 prob | ... | classN prob | -> batch0, grid(0,1), anchor#0
 *  +---+---+---+---+---+-------------+-----+-------------+
 *  |                     ...                             |
 *  +---+---+---+---+---+-------------+-----+-------------+
 *  | x | y | w | h | c | class0 prob | ... | classN prob | -> batch1, grid(0,0), anchor#0
 *  +---+---+---+---+---+-------------+-----+-------------+
 *  |                     ...                             |
 *  +---+---+---+---+---+-------------+-----+-------------+
 *
 *
 *  << output tensor format >>
 *  +------------------+-------+-------+------+------+------+------+
 *  | itemID(batchIdx) | label | score | xmin | ymin | xmax | ymax |
 *  +------------------+-------+-------+------+------+------+------+
 *  | itemID(batchIdx) | label | score | xmin | ymin | xmax | ymax |
 *  +------------------+-------+-------+------+------+------+------+
 *  |                             ...                              |
 *  +------------------+-------+-------+------+------+------+------+
 *  | itemID(batchIdx) | label | score | xmin | ymin | xmax | ymax |
 *  +------------------+-------+-------+------+------+------+------+
 *
 *******************************************************************************************/

template <typename Dtype>
YOLOOutputLayer<Dtype>::YOLOOutputLayer()
: Layer<Dtype>() {
    this->type = Layer<Dtype>::YOLOOutput;
}

template <typename Dtype>
YOLOOutputLayer<Dtype>::~YOLOOutputLayer() {
}

template <typename Dtype>
void YOLOOutputLayer<Dtype>::YOLOOutputForward(const Dtype* inputData, const int batch,
        const int side1, const int side2, const int dim) {

    int gridCount = side1 * side2;
    int classNum = (int)(SLPROP(YOLOOutput, numClasses));
    int elemPerAnchorBox = classNum + 4;
    int anchorBoxCount = (int)(dim / elemPerAnchorBox);
    float confThresh = (float)(SLPROP(YOLOOutput, scoreThresh));

    int resultCount = 0;
    Dtype left, top, right, bottom;

    vector<vector<Dtype>> output;
    vector<vector<Dtype>> result;

    for (int n = 0; n < batch; n++) {
        for (int i = 0; i < gridCount; i++){
            int gridX = i % side2;
            int gridY = i / side2;

            for (int j = 0; j < anchorBoxCount; j++){
                //int bboxesIdx = i * dim + j * elemPerAnchorBox;
                int bboxesIdx = (n * gridCount + i) * dim + j * elemPerAnchorBox;
                float x = inputData[bboxesIdx + 0];
                float y = inputData[bboxesIdx + 1];
                float w = inputData[bboxesIdx + 2];
                float h = inputData[bboxesIdx + 3];
                float c = inputData[bboxesIdx + 4];

                float maxProbability = inputData[bboxesIdx + 5];
                int labelIdx = 0;

                for (int k = 1; k < classNum - 1; k++){
                    if (maxProbability < inputData[bboxesIdx + 5 + k]){
                        labelIdx = k;
                        maxProbability = inputData[bboxesIdx + 5 + k];
                    }
                }

                float score = c * maxProbability;

                if (score <= confThresh) {
                    continue;
                }

                resultCount++;

                top = (float)(((float)gridY + y) / (float)side2 - 0.5 * h);
                bottom = (float)(((float)gridY + y) / (float)side2 + 0.5 * h);
                left = (float)(((float)gridX + x) / (float)side1 - 0.5 * w);
                right = (float)(((float)gridX + x) / (float)side1 + 0.5 * w);

                vector<Dtype> bbox(6);
                bbox[0] = score;
                bbox[1] = labelIdx;
                bbox[2] = max(Dtype(0.0), left);
                bbox[3] = max(Dtype(0.0), top);
                bbox[4] = min((Dtype)side1, right);
                bbox[5] = min((Dtype)side2, bottom);

                output.push_back(bbox);
            }

        }
        // NMS ***
        float nmsThresh = (float)(SLPROP(YOLOOutput, nmsIOUThresh));

        for (int i = 0; i < classNum - 1; i++) {
            vector<uint32_t> keep;
            vector<vector<float>> bboxes;
            vector<float> scores;

            for (int j = 0; j < resultCount; j++) {
                if (output[j][1] != i) {
                    continue;
                }
                vector<float> coord = {output[j][2], output[j][3],
                   output[j][4], output[j][5]};
                bboxes.push_back(coord);
                scores.push_back(output[j][0]);
            }

            if (bboxes.size() == 0)
                continue;

            ImageUtil<Dtype>::nms(bboxes, scores, nmsThresh, keep);

            for (int k = 0; k < keep.size(); k++) {
                vector<Dtype> pred(7);
                pred[0] = float(n);
                pred[1] = float(i) + 1.0;
                pred[2] = scores[keep[k]];
                pred[3] = bboxes[keep[k]][0];
                pred[4] = bboxes[keep[k]][1];
                pred[5] = bboxes[keep[k]][2];
                pred[6] = bboxes[keep[k]][3];

                result.push_back(pred);
            }

        }
    }

    if (result.size() > 0) {
        fillDataWith2dVec(result, this->_outputData[0]);
    } else {
        this->_outputData[0]->reshape({1, 1, 1, 7});
        this->_outputData[0]->mutable_host_data()[1] = -1;
    }
}

template <typename Dtype>
void YOLOOutputLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    const int batchSize = inputShape[0];

    this->_outputData[0]->reshape({1, 1, 1, 7});

}

template <typename Dtype>
void YOLOOutputLayer<Dtype>::feedforward(){
    reshape();

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    const int batchSize = inputShape[0];
    const int gridCountY = inputShape[1];
    const int gridCountX = inputShape[2];
    const int gridCountElem = inputShape[3];

    const Dtype* inputData = this->_inputData[0]->host_data();

    YOLOOutputForward(inputData, batchSize, gridCountX, gridCountY, gridCountElem);

}

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* YOLOOutputLayer<Dtype>::initLayer() {
	YOLOOutputLayer* layer = NULL;
	SNEW(layer, YOLOOutputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool YOLOOutputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::backwardTensor(void* instancePtr) {
	YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;
    //layer->backpropagation();
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool YOLOOutputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    if (SLPROP_BASE(input).size() != 1) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Output Layer should have only 1 input tensors but it has %d tensors",
            (int)SLPROP_BASE(input).size());
        return false;
    }
    if (SLPROP_BASE(output).size() != 1) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Output Layer should have only 1 output tensors but it has %d tensors",
            (int)SLPROP_BASE(output).size());
        return false;
    }

    if (inputShape[0].N <= 0 || inputShape[0].C <= 0 ||
           inputShape[0].H <= 0 || inputShape[0].W <= 0)
        return false;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 7;

    if (outputShape1.N <= 0 || outputShape1.C <= 0 ||
           outputShape1.H <= 0 || outputShape1.W <= 0)
        return false;

    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t YOLOOutputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class YOLOOutputLayer<float>;
