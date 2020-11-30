/*
 * AnnotatedLiveDataLayer.cpp
 *
 *  Created on: Sep 4, 2017
 *      Author: jkim
 */

#include <opencv2/highgui/highgui.hpp>

#include "AnnotatedLiveDataLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "IO.h"
#include "WorkContext.h"
#include "Param.h"
#include "Perf.h"
#include "EnumDef.h"
#include "MathFunctions.h"
#include "Sampler.h"
#include "ImageUtil.h"
#include "MemoryMgmt.h"

using namespace std;



template <typename Dtype>
AnnotatedLiveDataLayer<Dtype>::AnnotatedLiveDataLayer()
: InputLayer<Dtype>(),
  dataTransformer(&SLPROP(AnnotatedLiveData, dataTransformParam)),
  videoCapture(SLPROP(AnnotatedLiveData, camIndex)) {
	this->type = Layer<Dtype>::AnnotatedLiveData;

	DataTransformParam& dataTransformParam = this->dataTransformer.param;
	dataTransformParam.resizeParam = SLPROP(AnnotatedLiveData, resizeParam);
	dataTransformParam.resizeParam.updateInterpMode();

	// Make sure dimension is consistent within batch.
	if (this->dataTransformer.param.resizeParam.prob >= 0.f) {
		if (this->dataTransformer.param.resizeParam.resizeMode == ResizeMode::FIT_SMALL_SIZE) {
			SASSERT(SNPROP(batchSize) == 1, "Only support batch size of 1 for FIT_SMALL_SIZE.");
		}
	}

	SASSERT(this->videoCapture.isOpened(), "video device is not opened ... ");

	this->videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, SLPROP(AnnotatedLiveData, camResWidth));
	this->videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, SLPROP(AnnotatedLiveData, camResHeight));
}

template <typename Dtype>
AnnotatedLiveDataLayer<Dtype>::~AnnotatedLiveDataLayer() {}

template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP(AnnotatedLiveData, output).size(); i++) {
			SLPROP(AnnotatedLiveData, input).push_back(SLPROP(AnnotatedLiveData, output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	Layer<Dtype>::_adjustInputShape();

    const int batchSize = SNPROP(batchSize);



    /*
    // XXX: trace하면서 확인.
    // 여기는 mock data로 임시로 넘어가도록 수정
    AnnotatedDatum* annoDatum = new AnnotatedDatum();
    annoDatum->encoded;
    annoDatum->channels = 3;

    // Use data transformer to infer the expected data shape from annoDatum.
    vector<uint32_t> outputShape = this->dataTransformer.inferDataShape(annoDatum);
    outputShape[0] = batchSize;
    this->_outputData[0]->reshape(outputShape);
    */

    const uint32_t height = this->dataTransformer.param.resizeParam.height;
    const uint32_t width = this->dataTransformer.param.resizeParam.width;
    vector<uint32_t> outputShape = {1, 3, height, width};
    this->_outputData[0]->reshape(outputShape);

}

template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::feedforward() {
	reshape();
	load_batch();
}

template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
	load_batch();
}



template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::load_batch() {
	cv::Mat frame;
	bool frameValid = false;

	while (!frameValid) {
		try {
			this->videoCapture >> frame;
			frameValid = true;
		} catch (cv::Exception& e) {
			cout << "frame invalid ...  skip current frame ... " << endl;
		}
	}




	// 1. data
	const ResizeParam& resizeParam = this->dataTransformer.param.resizeParam;
	cv::Mat resizedFrame;
	cv::resize(frame, resizedFrame, cv::Size(resizeParam.width, resizeParam.height),
			0, 0, cv::INTER_LINEAR);

	// DataTransformer::transform에서 이미지 정보를 CHW 기준으로 조회해서
	// 아래와 같이 shape 생성
	vector<uint32_t> dataShape = {1, (uint32_t)resizedFrame.channels(),
			(uint32_t)resizedFrame.rows, (uint32_t)resizedFrame.cols};
	//vector<uint32_t> dataShape = {1, frame.rows, frame.cols, frame.channels()};
	this->_outputData[0]->reshape(dataShape);

	Datum datum;
	// channel_separated false -> opencv의 (b, g, r), (b, g, r) ... 을 그대로 저장
	// channel_separated true  -> 실상황 데이터를 전송하기 위해
	CVMatToDatum(resizedFrame, true, &datum);
	this->dataTransformer.transform(&datum, this->_outputData[0], 0);





	// 2. data_org
	frame.convertTo(frame, CV_32F);
	vector<uint32_t> dataOrgShape = {1, (uint32_t)frame.channels(),
			(uint32_t)frame.rows, (uint32_t)frame.cols};
	this->_outputData[1]->reshape(dataOrgShape);
	Dtype* data_org = this->_outputData[1]->mutable_host_data();
	ConvertHWCCVToHWC(frame, data_org);



}














template <typename Dtype>
int AnnotatedLiveDataLayer<Dtype>::getNumTrainData() {
	// XXX: int max로 설정해 두면 될 듯
	return INT_MAX;
}

template <typename Dtype>
int AnnotatedLiveDataLayer<Dtype>::getNumTestData() {
	return 0;
}

template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::shuffleTrainDataSet() {

}








/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* AnnotatedLiveDataLayer<Dtype>::initLayer() {
	AnnotatedLiveDataLayer* layer = NULL;
	SNEW(layer, AnnotatedLiveDataLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::destroyLayer(void* instancePtr) {
    AnnotatedLiveDataLayer<Dtype>* layer = (AnnotatedLiveDataLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	SASSERT0(!isInput);
	SASSERT0(index < 2);

    AnnotatedLiveDataLayer<Dtype>* layer = (AnnotatedLiveDataLayer<Dtype>*)instancePtr;
	SASSERT0(layer->_outputData.size() == index);
	layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool AnnotatedLiveDataLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    AnnotatedLiveDataLayer<Dtype>* layer = (AnnotatedLiveDataLayer<Dtype>*)instancePtr;
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
void AnnotatedLiveDataLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	AnnotatedLiveDataLayer<Dtype>* layer = (AnnotatedLiveDataLayer<Dtype>*)instancePtr;
	layer->feedforward();

}

template<typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool AnnotatedLiveDataLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    ResizeParam& resizeParam = SLPROP(AnnotatedData, resizeParam);
    const int batchSize = SNPROP(batchSize);
    const uint32_t height = resizeParam.height;
    const uint32_t width = resizeParam.width;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 3;
    outputShape1.H = height;
    outputShape1.W = width;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t AnnotatedLiveDataLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class AnnotatedLiveDataLayer<float>;
