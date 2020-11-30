/*
 * FrcnnTestOutputLayer.cpp
 *
 *  Created on: Dec 16, 2016
 *      Author: jkim
 */

#include <vector>
#include <array>


#include "FrcnnTestOutputLayer.h"
#include "BboxTransformUtil.h"
#include "PropMgmt.h"
#include "StdOutLog.h"
#include "MemoryMgmt.h"

#define FRCNNTESTOUTPUTLAYER_LOG 0

using namespace std;


template <typename Dtype>
FrcnnTestOutputLayer<Dtype>::FrcnnTestOutputLayer()
: Layer<Dtype>(),
  labelMap(SLPROP(FrcnnTestOutput, labelMapPath)) {
	this->type = Layer<Dtype>::FrcnnTestOutput;

	//SASSERT(SNPROP(status) == NetworkStatus::Test,
	//		"FrcnnTestOutputLayer can be run only in Test Status");
	/*
	this->classes = {"__background__", "aeroplane", "bicycle", "bird", "boat", "bottle",
			"bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
			"person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
			*/

    this->numClasses = SLPROP(FrcnnTestOutput, numClasses);
    SASSERT(labelMap.isValid() || this->numClasses > 1,
            "provide valid labelmap or numClasses larger than 1");

    if (this->labelMap.isValid()) {
	    this->labelMap.build();
    }

	this->boxColors.push_back(cv::Scalar(10, 163, 240));
	this->boxColors.push_back(cv::Scalar(44, 90, 130));
	this->boxColors.push_back(cv::Scalar(239, 80, 0));
	this->boxColors.push_back(cv::Scalar(37, 0, 162));
	this->boxColors.push_back(cv::Scalar(226, 161, 27));

	this->boxColors.push_back(cv::Scalar(115, 0, 216));
	this->boxColors.push_back(cv::Scalar(0, 196, 164));
	this->boxColors.push_back(cv::Scalar(255, 0, 106));
	this->boxColors.push_back(cv::Scalar(23, 169, 96));
	this->boxColors.push_back(cv::Scalar(0, 138, 0));

	this->boxColors.push_back(cv::Scalar(138, 96, 118));
	this->boxColors.push_back(cv::Scalar(100, 135, 109));
	this->boxColors.push_back(cv::Scalar(0, 104, 250));
	this->boxColors.push_back(cv::Scalar(208, 114, 244));
	this->boxColors.push_back(cv::Scalar(0, 20, 229));

	this->boxColors.push_back(cv::Scalar(63, 59, 122));
	this->boxColors.push_back(cv::Scalar(135, 118, 100));
	this->boxColors.push_back(cv::Scalar(169, 171, 0));
	this->boxColors.push_back(cv::Scalar(255, 0, 170));
	this->boxColors.push_back(cv::Scalar(0, 193, 216));
}

template <typename Dtype>
FrcnnTestOutputLayer<Dtype>::~FrcnnTestOutputLayer() {}


template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		this->_outputData[0]->reshape({1, 1, 1, 7});

		// mAP를 측정하기 위해 label을 만드는 경우에 해당
		if (this->_outputData.size() > 1) {
			this->_outputData[1]->reshape({1, 1, 1, 8});
		}
	}

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;
	}
}





template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::feedforward() {
	reshape();

	vector<vector<Dtype>> scores;
	vector<vector<Dtype>> predBoxes;

	imDetect(scores, predBoxes);
	testNet(scores, predBoxes);
}


template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::imDetect(vector<vector<Dtype>>& scores, vector<vector<Dtype>>& predBoxes) {

#if FRCNNTESTOUTPUTLAYER_LOG
	this->_printOn();
	this->_inputData[0]->print_data({}, false, -1);		// rois
	this->_inputData[1]->print_data({}, false, -1);		// im_info
	this->_inputData[2]->print_data({}, false, -1);		// cls_prob
	this->_inputData[3]->print_data({}, false, -1);		// bbox_pred
	this->_printOff();
#endif
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

#if FRCNNTESTOUTPUTLAYER_LOG
	print2dArray("boxes", boxes);
	this->_printOn();
	this->_inputData[3]->print_data({}, false);
	this->_printOff();
#endif
	fill2dVecWithData(this->_inputData[2], scores);


	if (SLPROP(FrcnnTestOutput, needNorm)) {
		//this->_inputData[3]->print_shape();
		const int numBoxes = this->_inputData[3]->getShape(0);
		const int numClasses = this->_inputData[3]->getShape(2) / 4;
		Dtype* bboxPred = this->_inputData[3]->mutable_host_data();

		int offset = 0;
		for (int i = 0; i < numBoxes; i++) {
			for (int j = 0; j < numClasses; j++) {
				offset = i * numClasses * 4 + j * 4;
				bboxPred[offset + 0] = bboxPred[offset + 0] * TRAIN_BBOX_NORMALIZE_STDS[0] +
						TRAIN_BBOX_NORMALIZE_MEANS[0];
				bboxPred[offset + 1] = bboxPred[offset + 1] * TRAIN_BBOX_NORMALIZE_STDS[1] +
						TRAIN_BBOX_NORMALIZE_MEANS[1];
				bboxPred[offset + 2] = bboxPred[offset + 2] * TRAIN_BBOX_NORMALIZE_STDS[2] +
						TRAIN_BBOX_NORMALIZE_MEANS[2];
				bboxPred[offset + 3] = bboxPred[offset + 3] * TRAIN_BBOX_NORMALIZE_STDS[3] +
						TRAIN_BBOX_NORMALIZE_MEANS[3];
			}
		}
	}

	// bbox_pred (#rois, 4 * num classes)
	BboxTransformUtil::bboxTransformInv(boxes, this->_inputData[3], predBoxes);
	BboxTransformUtil::clipBoxes(predBoxes,
			{round(imHeight/imScale), round(imWidth/imScale)});
	//BboxTransformUtil::clipBoxes(predBoxes,
	//		{round(imHeight), round(imWidth)});

#if FRCNNTESTOUTPUTLAYER_LOG
	print2dArray("boxes", boxes);
	//print2dArray("scores", scores);
	print2dArray("predBoxes", predBoxes);
#endif
}

template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::testNet(vector<vector<Dtype>>& scores,
		vector<vector<Dtype>>& boxes) {

	const Dtype confThresh = Dtype(SLPROP(FrcnnTestOutput, confThresh));
	const Dtype nmsThresh = Dtype(SLPROP(FrcnnTestOutput, nmsThresh));

	vector<vector<float>> result;

	vector<uint32_t> keep;
	vector<Dtype> clsScores;
	vector<vector<Dtype>> clsBoxes;
	vector<uint32_t> inds;
	vector<vector<Dtype>> detectionOut;

    const int classSize = (this->labelMap.isValid()) ? this->labelMap.getCount() : this->numClasses;
	for (int clsInd = 1; clsInd < classSize; clsInd++) {

		fillClsScores(scores, clsInd, clsScores);
		fillClsBoxes(boxes, clsInd, clsBoxes);
		nms(clsBoxes, clsScores, nmsThresh, keep);

		clsBoxes = vec_keep_by_index(clsBoxes, keep);
		clsScores = vec_keep_by_index(clsScores, keep);

		// score 중 confThresh 이상인 것에 대해
		np_where_s(clsScores, GE, confThresh, inds);
		//np_where_s(clsScores, GE, 0.01, inds);

		if (inds.size() == 0)
			continue;

		int offset = result.size();
		for (int i = 0; i < inds.size(); i++) {
			vector<float> temp(7);
			temp[0] = 0.f;
			temp[1] = float(clsInd);
			temp[2] = clsScores[inds[i]];
			temp[3] = clsBoxes[inds[i]][0];
			temp[4] = clsBoxes[inds[i]][1];
			temp[5] = clsBoxes[inds[i]][2];
			temp[6] = clsBoxes[inds[i]][3];
			result.push_back(temp);
		}
	}

	if (SLPROP(FrcnnTestOutput, outputResult)) {
		cv::Mat im = cv::imread(Util::imagePath, CV_LOAD_IMAGE_COLOR);
		uint32_t numBoxes = result.size();

		for (uint32_t i = 0; i < numBoxes; i++) {
			int clsInd = round(result[i][1]);
			cv::rectangle(im, cv::Point(result[i][3], result[i][4]),
				cv::Point(result[i][5], result[i][6]),
				boxColors[clsInd-1], 2);

            if (this->labelMap.isValid()) {
                cv::putText(im, this->labelMap.convertIndToLabel(clsInd) , cv::Point(result[i][3],
                        result[i][4]+15.0f), 2, 0.5f, boxColors[clsInd-1]);
            } else {
                cv::putText(im, std::to_string(clsInd), cv::Point(result[i][3],
                        result[i][4]+15.0f), 2, 0.5f, boxColors[clsInd-1]);
            }
		}

		if (SLPROP(FrcnnTestOutput, savePath) == "") {
			const string windowName = "result";
			cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
			cv::imshow(windowName, im);

			if (true) {
				cv::waitKey(0);
				cv::destroyAllWindows();
			}
		} else {
			cv::imwrite(SLPROP(FrcnnTestOutput, savePath) + "/" + Util::imagePath.substr(Util::imagePath.length()-10), im);
		}
	}

	if (result.size() > 0) {
		fillDataWith2dVec(result, this->_outputData[0]);
	} else {
		this->_outputData[0]->reshape({1, 1, 1, 7});
		this->_outputData[0]->mutable_host_data()[1] = -1;
	}

	// mAP 측정을 위해 gt_boxes Data를 받아 label로 변환하는 경우
	if (this->_inputData.size() > 4 && this->_outputData.size() > 1) {
		uint32_t numGtBoxes = this->_inputData[4]->getShape(2);
		this->_outputData[1]->reshape({1, 1, numGtBoxes, 8});
		const Dtype*gtBoxes = this->_inputData[4]->host_data();

		int offset = 0;
		vector<vector<Dtype>> labels(numGtBoxes);
		for (int i = 0; i < numGtBoxes; i++) {
			offset = i * this->_inputData[4]->getShape(3);

			labels[i].resize(8);
			labels[i][0] = 0.f;
			labels[i][1] = gtBoxes[offset + 4];;
			labels[i][2] = 0.f;
			labels[i][3] = gtBoxes[offset + 0];
			labels[i][4] = gtBoxes[offset + 1];
			labels[i][5] = gtBoxes[offset + 2];
			labels[i][6] = gtBoxes[offset + 3];
			labels[i][7] = 0.f;
		}
		fillDataWith2dVec(labels, this->_outputData[1]);
	}
	//displayBoxesOnImage("TEST_RESULT", Util::imagePath, 1, restoredBoxes, boxLabels, {},
	//		boxColors);
	//cout << "end object detection result ... " << endl;
}

template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::fillClsScores(vector<vector<Dtype>>& scores, int clsInd,
		vector<Dtype>& clsScores) {

	const int size = scores.size();
	clsScores.resize(size);
	for (int i = 0; i < size; i++) {
		clsScores[i] = scores[i][clsInd];
	}
}

template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::fillClsBoxes(vector<vector<Dtype>>& boxes, int clsInd,
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



template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::backpropagation() {

}



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* FrcnnTestOutputLayer<Dtype>::initLayer() {
	FrcnnTestOutputLayer* layer = NULL;
	SNEW(layer, FrcnnTestOutputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::destroyLayer(void* instancePtr) {
    FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 5);
	} else {
		SASSERT0(index < 2);
	}

    FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool FrcnnTestOutputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::backwardTensor(void* instancePtr) {
	FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool FrcnnTestOutputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 5)
        return false;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 7;
    outputShape.push_back(outputShape1);

    TensorShape outputShape2;
    outputShape2.N = 1;
    outputShape2.C = 1;
    outputShape2.H = 1;
    outputShape2.W = 8;
    outputShape.push_back(outputShape2);

    return true;
}

template<typename Dtype>
uint64_t FrcnnTestOutputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class FrcnnTestOutputLayer<float>;


