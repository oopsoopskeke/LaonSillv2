/**
 * @file LiveDataInputLayer.cpp
 * @date 2017-12-11
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <unistd.h>
#include <vector>

#include "LiveDataInputLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "WorkContext.h"
#include "Param.h"
#include "Perf.h"
#include "MemoryMgmt.h"

#if LIVEDATAINPUTLAYER_TEST
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

using namespace std;


template <typename Dtype>
LiveDataInputLayer<Dtype>::LiveDataInputLayer()
: InputLayer<Dtype>(),
  dataTransformer(&SLPROP(LiveDataInput, dataTransformParam)) {
	this->type = Layer<Dtype>::LiveDataInput;

	SASSERT(SLPROP(LiveDataInput, rows) != 0 || SLPROP(LiveDataInput, resizeParam).height != 0,
			"one of rows or resizeParam.height should be larger than 0.");
	SASSERT(SLPROP(LiveDataInput, cols) != 0 || SLPROP(LiveDataInput, resizeParam).width != 0,
			"one of cols or resizeParam.width should be larger than 0.");

	this->height = SLPROP(LiveDataInput, resizeParam).height > 0 ?
			SLPROP(LiveDataInput, resizeParam).height :
			SLPROP(LiveDataInput, rows);

	this->width = SLPROP(LiveDataInput, resizeParam).width > 0 ?
			SLPROP(LiveDataInput, resizeParam).width :
			SLPROP(LiveDataInput, cols);

	DataTransformParam& dataTransformParam = this->dataTransformer.param;
	dataTransformParam.resizeParam = SLPROP(LiveDataInput, resizeParam);
	dataTransformParam.resizeParam.updateInterpMode();

#if LIVEDATAINPUTLAYER_TEST
	testList.push_back("/home/jkim/Dev/data/image/ilsvrc12_train/images/n15075141/n15075141_158.JPEG");
	testList.push_back("/home/jkim/Dev/data/image/ilsvrc12_train/images/n09193705/n09193705_9517.JPEG");
	testList.push_back("/home/jkim/Dev/data/image/ilsvrc12_train/images/n07753113/n07753113_9411.JPEG");
	cur = 0;
	cv::namedWindow("input");
#endif
}

template <typename Dtype>
LiveDataInputLayer<Dtype>::~LiveDataInputLayer() {
	// TODO Auto-generated destructor stub
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
    Layer<Dtype>::_adjustInputShape();

    this->_inputShape[0][0] = 1;
    this->_inputShape[0][1] = SLPROP(LiveDataInput, channels);
    this->_inputShape[0][2] = this->height;
    this->_inputShape[0][3] = this->width;
    this->_inputData[0]->reshape(this->_inputShape[0]);
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::feedImage(const int channels, const int height,
		const int width, float* image) {
    //SASSERT0(height == SLPROP(LiveDataInput, rows));
    //SASSERT0(width == SLPROP(LiveDataInput, cols));
	SASSERT0(image != NULL);

	// XXX: 이문헌 연구원님과 협의 후, image 자체를 CV_8U로 받도록 수정
	// 데이터 전송량도 많아지고 다시 CV_8U로 변환해야 하는 오버헤드 발생
    if (channels == 1) {
        cv::Mat img(height, width, CV_32FC1, image);
        img.convertTo(img, CV_8UC1);
        this->_inputShape[0] = this->dataTransformer.inferDataShape(img);
        this->_inputData[0]->reshape(this->_inputShape[0]);
        this->dataTransformer.transform(img, this->_inputData[0], 0);
    } else {
        SASSUME0(channels == 3);
        cv::Mat img(height, width, CV_32FC3, image);
        img.convertTo(img, CV_8UC3);
        this->_inputShape[0] = this->dataTransformer.inferDataShape(img);
        this->_inputData[0]->reshape(this->_inputShape[0]);
        this->dataTransformer.transform(img, this->_inputData[0], 0);
    }
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::feedforward() {
	reshape();
#if LIVEDATAINPUTLAYER_TEST
	string test = testList[cur];
	cur++;
	if (cur >= testList.size()) {
		cur = 0;
	}
	cv::Mat img = cv::imread(test);
	img.convertTo(img, CV_32FC3);
	feedImage(img.channels(), img.rows, img.cols, (float*)img.data);
#endif
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
}

template <typename Dtype>
int LiveDataInputLayer<Dtype>::getNumTrainData() {
    return 1;
}

template <typename Dtype>
int LiveDataInputLayer<Dtype>::getNumTestData() {
	return 0;
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::shuffleTrainDataSet() {

}

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* LiveDataInputLayer<Dtype>::initLayer() {
	LiveDataInputLayer* layer = NULL;
	SNEW(layer, LiveDataInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void LiveDataInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    LiveDataInputLayer<Dtype>* layer = (LiveDataInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void LiveDataInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	if (isInput) {
		SASSERT0(false);
	} else {
		SASSERT0(index < 2);
	}

    LiveDataInputLayer<Dtype>* layer = (LiveDataInputLayer<Dtype>*)instancePtr;
    if (!isInput) {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool LiveDataInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    LiveDataInputLayer<Dtype>* layer = (LiveDataInputLayer<Dtype>*)instancePtr;
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
void LiveDataInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	LiveDataInputLayer<Dtype>* layer = (LiveDataInputLayer<Dtype>*)instancePtr;
	layer->feedforward();

}

template<typename Dtype>
void LiveDataInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void LiveDataInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool LiveDataInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    if (inputShape.size() != 0) {
        return false;
    }

    if (SLPROP_BASE(output).size() != 1) {
        return false;
    }

    // XXX: 일단 Lenet을 테스트하는 선에서만 동작할 수 있도록 제한적으로 구현하였다.
    //      추후에 보강이 필요하다.
    TensorShape outputShape1;
    outputShape1.N = SNPROP(batchSize);
    outputShape1.C = SLPROP(LiveDataInput, channels);
    outputShape1.H = SLPROP(LiveDataInput, resizeParam).height;
    outputShape1.W = SLPROP(LiveDataInput, resizeParam).width;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t LiveDataInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class LiveDataInputLayer<float>;
