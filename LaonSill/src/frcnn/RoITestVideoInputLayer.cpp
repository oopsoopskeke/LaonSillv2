/*
 * RoITestVideoInputLayer.cpp
 *
 *  Created on: May 30, 2017
 *      Author: jkim
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "RoITestVideoInputLayer.h"
#include "PascalVOC.h"
#include "frcnn_common.h"
#include "RoIDBUtil.h"
#include "MockDataSet.h"
#include "SysLog.h"
#include "Util.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"


#define ROITESTVIDEOINPUTLAYER_LOG 0

using namespace std;

template <typename Dtype>
RoITestVideoInputLayer<Dtype>::RoITestVideoInputLayer()
: InputLayer<Dtype>(),
  cap(SLPROP(RoITestVideoInput, videoPath)) {
	this->type = Layer<Dtype>::RoITestVideoInput;

	SASSERT(SNPROP(status) == NetworkStatus::Test,
			"RoITestVideoInputLayer can be run only in Test Status");

	SASSERT(this->cap.isOpened(), "Could not open video %s", SLPROP(RoITestVideoInput, videoPath).c_str());
	this->_dataSet = NULL;
	SNEW(this->_dataSet, MockDataSet<Dtype>, 1, 1, 1, 1, 50, 1);
	SASSUME0(this->_dataSet != NULL);

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
RoITestVideoInputLayer<Dtype>::~RoITestVideoInputLayer() {

}


template <typename Dtype>
void RoITestVideoInputLayer<Dtype>::reshape() {
	// 입력 레이어의 경우 outputs만 사용자가 설정,
	// inputs에 대해 outputs와 동일하게 Data 참조하도록 강제한다.
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}

	Layer<Dtype>::_adjustInputShape();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		// "data"
		if (i == 0) {
			const vector<uint32_t> dataShape =
					{1, 3, vec_max(TEST_SCALES), TEST_MAX_SIZE};
			this->_inputData[0]->reshape(dataShape);
			this->_inputShape[0] = dataShape;

#if ROITESTVIDEOINPUTLAYER_LOG
			printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(),
					dataShape[0], dataShape[1], dataShape[2], dataShape[3]);
#endif
		}
		// "im_info"
		else if (i == 1) {
			const vector<uint32_t> iminfoShape = {1, 1, 1, 3};
			this->_inputShape[1] = iminfoShape;
			this->_inputData[1]->reshape(iminfoShape);

#if ROITESTVIDEOINPUTLAYER_LOG
			printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(),
					iminfoShape[0], iminfoShape[1], iminfoShape[2], iminfoShape[3]);
#endif
		}
		// "raw_im"
		else if (i == 2) {
			const vector<uint32_t> rawImShape = {1, 1, 1, 1};
			this->_inputShape[2] = rawImShape;
			this->_inputData[2]->reshape(rawImShape);
		}
	}
}



template <typename Dtype>
void RoITestVideoInputLayer<Dtype>::feedforward() {
	reshape();
	getNextMiniBatch();
}

template <typename Dtype>
void RoITestVideoInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();
	getNextMiniBatch();
}


template <typename Dtype>
void RoITestVideoInputLayer<Dtype>::getNextMiniBatch() {
	cv::Mat frame;
	for (int i = 0; i < 3; i++) {
		if (!this->cap.grab()) {
			cout << "END OF VIDEO ... " << endl;
			exit(1);
		}
	}
	this->cap.retrieve(frame);
	getImageBlob(frame);
}


template <typename Dtype>
float RoITestVideoInputLayer<Dtype>::getImageBlob(cv::Mat& im) {
	cv::Mat imOrig;
	im.copyTo(imOrig);
	imOrig.convertTo(imOrig, CV_32F);


	im.convertTo(im, CV_32FC3);
	const vector<uint32_t> rawImShape = {1, (uint32_t)im.rows, (uint32_t)im.cols, 3};
	this->_inputData[2]->reshape(rawImShape);
	this->_inputData[2]->set_host_data((Dtype*)im.data);


	float* imPtr = (float*)imOrig.data;

	int n = imOrig.rows * imOrig.cols * imOrig.channels();
	for (int i = 0; i < n; i+=3) {
		imPtr[i+0] -= SLPROP(RoITestVideoInput, pixelMeans)[0];
		imPtr[i+1] -= SLPROP(RoITestVideoInput, pixelMeans)[1];
		imPtr[i+2] -= SLPROP(RoITestVideoInput, pixelMeans)[2];
	}

	const vector<uint32_t> imShape = {(uint32_t)imOrig.cols, (uint32_t)imOrig.rows,
			(uint32_t)imOrig.channels()};
	uint32_t imSizeMin = np_min(imShape, 0, 2);
	uint32_t imSizeMax = np_max(imShape, 0, 2);

	const float targetSize = TEST_SCALES[0];
	float imScale = targetSize / float(imSizeMin);
	// Prevent the biggest axis from being more than MAX_SIZE
	if (np_round(imScale * imSizeMax) > TEST_MAX_SIZE)
		imScale = float(TEST_MAX_SIZE) / float(imSizeMax);

	cv::resize(imOrig, im, cv::Size(), imScale, imScale, CV_INTER_LINEAR);

	// 'data'
	const vector<uint32_t> inputShape = {1, (uint32_t)im.rows, (uint32_t)im.cols, 3};
	this->_inputData[0]->reshape(inputShape);
	this->_inputData[0]->set_host_data((Dtype*)im.data);

	// Move channels (axis 3) to axis 1
	// Axis order will become: (batch elem, channel, height, width)
	const vector<uint32_t> channelSwap = {0, 3, 1, 2};
	this->_inputData[0]->transpose(channelSwap);
	this->_inputShape[0] = this->_inputData[0]->getShape();

	// 'im_info'
	this->_inputData[1]->reshape({1, 1, 1, 3});
	this->_inputData[1]->mutable_host_data()[0] = Dtype(im.rows);
	this->_inputData[1]->mutable_host_data()[1] = Dtype(im.cols);
	this->_inputData[1]->mutable_host_data()[2] = Dtype(imScale);

	return imScale;
}



template<typename Dtype>
int RoITestVideoInputLayer<Dtype>::getNumTrainData() {
    return this->_dataSet->getNumTrainData();
}

template<typename Dtype>
int RoITestVideoInputLayer<Dtype>::getNumTestData() {
    return this->_dataSet->getNumTestData();
}

template<typename Dtype>
void RoITestVideoInputLayer<Dtype>::shuffleTrainDataSet() {
    return this->_dataSet->shuffleTrainDataSet();
}











/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* RoITestVideoInputLayer<Dtype>::initLayer() {
	RoITestVideoInputLayer* layer = NULL;
	SNEW(layer, RoITestVideoInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void RoITestVideoInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    RoITestVideoInputLayer<Dtype>* layer = (RoITestVideoInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void RoITestVideoInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    RoITestVideoInputLayer<Dtype>* layer = (RoITestVideoInputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool RoITestVideoInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    RoITestVideoInputLayer<Dtype>* layer = (RoITestVideoInputLayer<Dtype>*)instancePtr;
    layer->reshape();

    if (SNPROP(miniBatch) == 0) {
		int trainDataNum = layer->getNumTrainData();
		if (trainDataNum % SNPROP(batchSize) == 0) {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize);
		} else {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize) + 1;
		}
		WorkContext::curPlanInfo->miniBatchCount = SNPROP(miniBatch);
	}

    return true;
}

template<typename Dtype>
void RoITestVideoInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	RoITestVideoInputLayer<Dtype>* layer = (RoITestVideoInputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void RoITestVideoInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
void RoITestVideoInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool RoITestVideoInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    // data
    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 3;
    outputShape1.H = vec_max(TEST_SCALES);
    outputShape1.W = TEST_MAX_SIZE;
    outputShape.push_back(outputShape1);

    // im_info
    TensorShape outputShape2;
    outputShape2.N = 1;
    outputShape2.C = 1;
    outputShape2.H = 1;
    outputShape2.W = 3;
    outputShape.push_back(outputShape2);

    // raw_im 
    TensorShape outputShape3;
    outputShape3.N = 1;
    outputShape3.C = 1;
    outputShape3.H = 1;
    outputShape3.W = 1;
    outputShape.push_back(outputShape3);

    return true;
}

template<typename Dtype>
uint64_t RoITestVideoInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}





template class RoITestVideoInputLayer<float>;
