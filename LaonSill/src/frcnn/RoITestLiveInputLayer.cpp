/*
 * RoITestLiveInputLayer.cpp
 *
 *  Created on: Jul 13, 2017
 *      Author: jkim
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "RoITestLiveInputLayer.h"
#include "common.h"
#include "frcnn_common.h"
#include "PropMgmt.h"
#include "StdOutLog.h"
#include "MemoryMgmt.h"

using namespace std;


template <typename Dtype>
RoITestLiveInputLayer<Dtype>::RoITestLiveInputLayer()
: InputLayer<Dtype>() {
	this->type = Layer<Dtype>::RoITestLiveInput;

	SASSERT((SNPROP(useCompositeModel) == true) || (SNPROP(status) == NetworkStatus::Test),
			"RoITestLiveInputLayer can be run only in Test Status");
}

template <typename Dtype>
RoITestLiveInputLayer<Dtype>::~RoITestLiveInputLayer() {

}


template <typename Dtype>
void RoITestLiveInputLayer<Dtype>::reshape() {
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
		}
		// "im_info"
		else if (i == 1) {
			const vector<uint32_t> iminfoShape = {1, 1, 1, 3};
			this->_inputShape[1] = iminfoShape;
			this->_inputData[1]->reshape(iminfoShape);
		}
	}
}

template <typename Dtype>
void RoITestLiveInputLayer<Dtype>::feedImage(const int channels, const int height,
		const int width, float* image) {
	SASSERT0(channels == 3);
	SASSERT0(height > 0);
	SASSERT0(width > 0);
	SASSERT0(image != NULL);

	this->channels = channels;
	this->height = height;
	this->width = width;
	this->image = image;
}

template <typename Dtype>
void RoITestLiveInputLayer<Dtype>::getNextMiniBatch() {
	cv::Mat cv_img(this->height, this->width, CV_32FC3, this->image);
	SASSERT(cv_img.data, "Could not decode datum.");

	// Mat으로 input data에 해당하는 'data'와 'im_info' Data 초기화
	// feedforward 준비 완료
	getBlobs(cv_img);
}




template <typename Dtype>
void RoITestLiveInputLayer<Dtype>::feedforward() {
	reshape();
	getNextMiniBatch();
}


template <typename Dtype>
float RoITestLiveInputLayer<Dtype>::getBlobs(cv::Mat& im) {
	return getImageBlob(im);
}

template <typename Dtype>
float RoITestLiveInputLayer<Dtype>::getImageBlob(cv::Mat& im) {
	cv::Mat imOrig;
	im.copyTo(imOrig);
	//imOrig.convertTo(imOrig, CV_32F);

	float* imPtr = (float*)imOrig.data;

	int n = imOrig.rows * imOrig.cols * imOrig.channels();
	for (int i = 0; i < n; i+=3) {
		imPtr[i+0] -= SLPROP(RoITestLiveInput, pixelMeans)[0];
		imPtr[i+1] -= SLPROP(RoITestLiveInput, pixelMeans)[1];
		imPtr[i+2] -= SLPROP(RoITestLiveInput, pixelMeans)[2];
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
int RoITestLiveInputLayer<Dtype>::getNumTrainData() {
    return 1;
}




/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* RoITestLiveInputLayer<Dtype>::initLayer() {
	RoITestLiveInputLayer* layer = NULL;
	SNEW(layer, RoITestLiveInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void RoITestLiveInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    RoITestLiveInputLayer<Dtype>* layer = (RoITestLiveInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void RoITestLiveInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(false);
	} else {
		SASSERT0(index < 2);
	}

    RoITestLiveInputLayer<Dtype>* layer = (RoITestLiveInputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool RoITestLiveInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    RoITestLiveInputLayer<Dtype>* layer = (RoITestLiveInputLayer<Dtype>*)instancePtr;
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
void RoITestLiveInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	RoITestLiveInputLayer<Dtype>* layer = (RoITestLiveInputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void RoITestLiveInputLayer<Dtype>::backwardTensor(void* instancePtr) {
}

template<typename Dtype>
void RoITestLiveInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool RoITestLiveInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
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

    return true;
}

template<typename Dtype>
uint64_t RoITestLiveInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}


template class RoITestLiveInputLayer<float>;

