/*
 * FrcnnTestLiveOutputLayer.cpp
 *
 *  Created on: Jul 13, 2017
 *      Author: jkim
 */

#include <vector>
#include <array>

#include "FrcnnTestLiveOutputLayer.h"
#include "BboxTransformUtil.h"
#include "PropMgmt.h"
#include "StdOutLog.h"
#include "MemoryMgmt.h"

using namespace std;


template <typename Dtype>
FrcnnTestLiveOutputLayer<Dtype>::FrcnnTestLiveOutputLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::FrcnnTestLiveOutput;

	SASSERT(SNPROP(useCompositeModel) == true || SNPROP(status) == NetworkStatus::Test,
				"FrcnnTestLiveOutputLayer can be run only in Test Status");
}

template <typename Dtype>
FrcnnTestLiveOutputLayer<Dtype>::~FrcnnTestLiveOutputLayer() {}


template <typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;
	}
}


template <typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::feedforward() {
	reshape();

	vector<vector<Dtype>> scores;
	vector<vector<Dtype>> predBoxes;

	imDetect(scores, predBoxes);
	testNet(scores, predBoxes);
}


template <typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::imDetect(vector<vector<Dtype>>& scores,
		vector<vector<Dtype>>& predBoxes) {

	// im_info (1, 1, 1, 3-[height, width, scale])
	const Dtype imHeight = this->_inputData[1]->host_data()[0];
	const Dtype imWidth = this->_inputData[1]->host_data()[1];
	const Dtype imScale = this->_inputData[1]->host_data()[2];

	// rois (1, 1, #rois, 5-[batch index, x1, y1, x2, y2])
	const uint32_t numRois = this->_inputData[0]->getShape(2);
	vector<vector<Dtype>> boxes(numRois);
	const Dtype* rois = this->_inputData[0]->host_data();

	for (uint32_t i = 0; i < numRois; i++) {
		boxes[i].resize(4);
		// unscale back to raw image space
		boxes[i][0] = rois[5 * i + 1] / imScale;
		boxes[i][1] = rois[5 * i + 2] / imScale;
		boxes[i][2] = rois[5 * i + 3] / imScale;
		boxes[i][3] = rois[5 * i + 4] / imScale;
	}

	fill2dVecWithData(this->_inputData[2], scores);

	// bbox_pred (#rois, 4 * num classes)
	BboxTransformUtil::bboxTransformInv(boxes, this->_inputData[3], predBoxes);
	BboxTransformUtil::clipBoxes(predBoxes,
			{round(imHeight/imScale), round(imWidth/imScale)});
}

template <typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::testNet(vector<vector<Dtype>>& scores,
		vector<vector<Dtype>>& boxes) {
	const Dtype confThresh = Dtype(SLPROP(FrcnnTestLiveOutput, confThresh));
	const Dtype nmsThresh = Dtype(SLPROP(FrcnnTestLiveOutput, nmsThresh));

	vector<vector<Dtype>> result;

	vector<uint32_t> keep;
	vector<Dtype> clsScores;
	vector<vector<Dtype>> clsBoxes;
	vector<uint32_t> inds;

	const int numClasses = SLPROP(FrcnnTestLiveOutput, numClasses);
	for (int clsInd = 1; clsInd < numClasses; clsInd++) {
		//int clsInd = 15;		// Person Only
		//const string& cls = "person";
		fillClsScores(scores, clsInd, clsScores);
		fillClsBoxes(boxes, clsInd, clsBoxes);

		//cout << cls << "\t\tboxes before nms: " << scores.size();
		nms(clsBoxes, clsScores, nmsThresh, keep);
		//cout << " , after nms: " << keep.size() << endl;

		clsBoxes = vec_keep_by_index(clsBoxes, keep);
		clsScores = vec_keep_by_index(clsScores, keep);

		// score 중 confThresh 이상인 것에 대해
		np_where_s(clsScores, GE, confThresh, inds);

		if (inds.size() == 0)
			continue;

		int offset = result.size();
		for (int i = 0; i < inds.size(); i++) {
			vector<Dtype> temp(7);
			temp[0] = 0.f;
			temp[1] = float(clsInd);
			temp[2] = clsScores[inds[i]];
			temp[3] = clsBoxes[inds[i]][0];
			temp[4] = clsBoxes[inds[i]][1];
			temp[5] = clsBoxes[inds[i]][2];
			temp[6] = clsBoxes[inds[i]][3];

			result.push_back(temp);
		}

		/*
		int numBBoxes = inds.size();
		if (numBBoxes == 0) {
			this->_outputData[0]->reshape({1, 1, 1, 1});
		} else {
			this->_outputData[0]->reshape({1, 1, (uint32_t)numBBoxes, 5});
			Dtype* bboxResult = this->_outputData[0]->mutable_host_data();
			for (int i = 0; i < numBBoxes; i++) {
				bboxResult[i * 5 + 0] = clsBoxes[inds[i]][0];
				bboxResult[i * 5 + 1] = clsBoxes[inds[i]][1];
				bboxResult[i * 5 + 2] = clsBoxes[inds[i]][2];
				bboxResult[i * 5 + 3] = clsBoxes[inds[i]][3];
				bboxResult[i * 5 + 4] = clsScores[inds[i]];
			}
		}
		*/
	}

	if (result.size() > 0) {
		fillDataWith2dVec(result, this->_outputData[0]);
	} else {
		this->_outputData[0]->reshape({1, 1, 1, 7});
		this->_outputData[0]->mutable_host_data()[1] = -1;
	}
}

template <typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::fillClsScores(vector<vector<Dtype>>& scores, int clsInd,
		vector<Dtype>& clsScores) {

	const int size = scores.size();
	clsScores.resize(size);
	for (int i = 0; i < size; i++) {
		clsScores[i] = scores[i][clsInd];
	}
}

template <typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::fillClsBoxes(vector<vector<Dtype>>& boxes, int clsInd,
		vector<vector<Dtype>>& clsBoxes) {
	const int size = boxes.size();
	clsBoxes.resize(size);

	for (int i = 0; i < size; i++) {
		clsBoxes[i].resize(4);

		int base = clsInd * 4;
		clsBoxes[i][0] = boxes[i][base + 0];
		clsBoxes[i][1] = boxes[i][base + 1];
		clsBoxes[i][2] = boxes[i][base + 2];
		clsBoxes[i][3] = boxes[i][base + 3];
	}
}



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* FrcnnTestLiveOutputLayer<Dtype>::initLayer() {
	FrcnnTestLiveOutputLayer* layer = NULL;
	SNEW(layer, FrcnnTestLiveOutputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::destroyLayer(void* instancePtr) {
    FrcnnTestLiveOutputLayer<Dtype>* layer = (FrcnnTestLiveOutputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 4);
	} else {
		SASSERT0(index < 1);
	}

    FrcnnTestLiveOutputLayer<Dtype>* layer = (FrcnnTestLiveOutputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool FrcnnTestLiveOutputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    FrcnnTestLiveOutputLayer<Dtype>* layer = (FrcnnTestLiveOutputLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	FrcnnTestLiveOutputLayer<Dtype>* layer = (FrcnnTestLiveOutputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::backwardTensor(void* instancePtr) {
	FrcnnTestLiveOutputLayer<Dtype>* layer = (FrcnnTestLiveOutputLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void FrcnnTestLiveOutputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool FrcnnTestLiveOutputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 4)
        return false;

    // 사실상 _outputData를 사용하지 않음.
    // inputShape로 초기화해준다.
    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t FrcnnTestLiveOutputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class FrcnnTestLiveOutputLayer<float>;
