/*
 * RecallLayer.cpp
 *
 *  Created on: Aug 9, 2017
 *      Author: jkim
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "RecallLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "IO.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
RecallLayer<Dtype>::RecallLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Recall;

	int numClasses = SLPROP(Recall, numClasses);
	SASSERT(numClasses == 2, "only supports two classes case.");
	float confidence = SLPROP(Recall, confidence);
	SASSERT(confidence > 0.f && confidence <= 1.f, "confidence should be > 0.f and < 1.f");
	int targetClass = SLPROP(Recall, targetClass);
	SASSERT(targetClass == 0 || targetClass == 1, "targetClass should be 0 or 1");

	for (int i = 0; i < 2; i++) {
		this->sampleCount[i] = 0;
		this->ttCount[i] = 0;
		//this->tfCount[i] = 0;
		this->missCount[i] = 0;
	}
}

template <typename Dtype>
RecallLayer<Dtype>::~RecallLayer() {

}

template <typename Dtype>
void RecallLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	// label shape는 변하지 않음.
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	this->_inputShape[0] = this->_inputData[0]->getShape();
	this->_inputShape[1] = this->_inputData[1]->getShape();

	vector<uint32_t> outputShape({1, 1, 1, 1});
	this->_outputData[0]->reshape(outputShape);
}

template <typename Dtype>
void RecallLayer<Dtype>::feedforward() {
	reshape();

	const int batchSize = SNPROP(batchSize);
	const int numClasses = SLPROP(Recall, numClasses);
	const float confidence = SLPROP(Recall, confidence);
	const int targetClass = SLPROP(Recall, targetClass);
	const int otherClass = (targetClass == 1) ? 0 : 1;

#if 0
	cout << "batchSize: " << batchSize << ", numClasses: " << numClasses << ", confidence: " <<
			confidence << ", targetClass: " << targetClass << ", otherClass: " << otherClass << endl;
#endif

	const Dtype* inputData = this->_inputData[0]->host_data();
	const Dtype* inputLabel = this->_inputData[1]->host_data();

	for (int i = 0; i < batchSize; i++) {
		int label = (int)inputLabel[i];
		this->sampleCount[label]++;

#if 0
		cout << i << "th batch ... " << endl;
		cout << "label: " << label << endl;
		cout << "conf0: " << inputData[i * numClasses] << ", conf1: " <<
				inputData[i * numClasses + 1] << ", sum: " <<
				inputData[i * numClasses] + inputData[i * numClasses + 1] << endl;
#endif

		// confidence 0.5인 경우 -> 일반 케이스
		// confidence 0.5 이하인 경우
		Dtype targetScore = inputData[i * numClasses + targetClass];
		if (targetScore > confidence) {
			//cout << "classfied as target class ... ";
			if (label == targetClass) {
				// 정상이라고 예측했고 정상인 경우
				//cout << "correct ... " << endl;
				this->ttCount[targetClass]++;
			} else {
				// 정상이라고 예측했는데 정상이 아닌 경우
				//cout << "incorrect ... " << endl;
				this->missCount[targetClass]++;
				//save(targetScore, i);
			}
		} else {
			//cout << "classfied as other class ... ";
			if (label == otherClass) {
				// 비정상이라고 예측했고 비정상인 경우
				//cout << "correct ... " << endl;
				this->ttCount[otherClass]++;
			} else {
				// 비정상이라고 예측했는데 정상인 경우
				//cout << "incorrect ... " << endl;
			}
		}
	}

	if (SNPROP(iterations) == SNPROP(testInterval) - 1) {

		int totalSampleCount = this->sampleCount[0] + this->sampleCount[1];
		int totalTtCount = this->ttCount[0] + this->ttCount[1];

		STDOUT_LOG("sample count[%s] : [0] - %d, [1] - %d", SLPROP_BASE(name).c_str(),
				this->sampleCount[0], this->sampleCount[1]);
		STDOUT_LOG("tt count[%s] : [0] - %d, [1] - %d", SLPROP_BASE(name).c_str(),
				this->ttCount[0], this->ttCount[1]);

		float acc = totalTtCount / float(totalSampleCount);
		STDOUT_LOG("average accuracy[%s] : %f", SLPROP_BASE(name).c_str(), acc);

		float recall = this->ttCount[targetClass] / float(this->sampleCount[targetClass]);
		STDOUT_LOG("average recall[%s] : %f", SLPROP_BASE(name).c_str(), recall);
	}

}

template <typename Dtype>
void RecallLayer<Dtype>::backpropagation() {
	//SASSERT(false, "Not implemented yet.");
}


template <typename Dtype>
void RecallLayer<Dtype>::save(const float score, const int batchIndex) {
	const vector<float>& mean = SLPROP(Recall, mean);
	const string outputPath = SLPROP(Recall, outputPath);

	const int channels = this->_inputData[2]->channels();
	const int height = this->_inputData[2]->height();
	const int width = this->_inputData[2]->width();

	cv::Mat mask(height, width, CV_32FC3, cv::Scalar(mean[0], mean[1], mean[2]));
	string path = outputPath + "/" + to_string(this->missCount[SLPROP(Recall, targetClass)]) + ".jpg";
	cout << "path: " << path << endl;

	const Dtype* src = this->_inputData[2]->host_data();
	Dtype* dst = NULL;
	SMALLOC(dst, Dtype, channels * height * width * sizeof(Dtype));
	SASSUME0(dst != NULL);

	int offset = this->_inputData[2]->offset(batchIndex);
	ConvertCHWToHWC(channels, height, width, src + offset, dst);

	cv::Mat cv_img(height, width, CV_32FC3, dst);
	cv_img += mask;
	cv_img.convertTo(cv_img, CV_8UC3);

	char scoreBuf[100];
	sprintf(scoreBuf, "%f", score);
	cv::putText(cv_img, string(scoreBuf), cv::Point(0, height), 2, 0.5f, cv::Scalar(0, 0, 255));
	cv::imwrite(path, cv_img);

	SDELETE(dst);
}




/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* RecallLayer<Dtype>::initLayer() {
	RecallLayer* layer = NULL;
	SNEW(layer, RecallLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void RecallLayer<Dtype>::destroyLayer(void* instancePtr) {
    RecallLayer<Dtype>* layer = (RecallLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void RecallLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	if (isInput) {
		SASSERT0(index < 3);
	} else {
		SASSERT0(index < 1);
	}

    RecallLayer<Dtype>* layer = (RecallLayer<Dtype>*)instancePtr;
    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool RecallLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    RecallLayer<Dtype>* layer = (RecallLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void RecallLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	RecallLayer<Dtype>* layer = (RecallLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void RecallLayer<Dtype>::backwardTensor(void* instancePtr) {
	//RecallLayer<Dtype>* layer = (RecallLayer<Dtype>*)instancePtr;
	//layer->backpropagation();
	SASSERT0(false);
}

template<typename Dtype>
void RecallLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool RecallLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 1;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t RecallLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}


template class RecallLayer<float>;
