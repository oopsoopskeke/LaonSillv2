/*
 * DetectionOutputLiveLayer.cpp
 *
 *  Created on: May 15, 2017
 *      Author: jkim
 */

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "DetectionOutputLiveLayer.h"
#include "BBoxUtil.h"
#include "MathFunctions.h"
#include "StdOutLog.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"
#include "IO.h"
#include "MathFunctions.h"

using namespace std;
using namespace boost::property_tree;

template <typename Dtype>
DetectionOutputLiveLayer<Dtype>::DetectionOutputLiveLayer()
: Layer<Dtype>(),
  bboxPreds("bboxPreds"),
  bboxPermute("bboxPermute"),
  confPermute("confPermute"),
  temp("temp"),
  dispName("LAONADE AUTOTUNED"),
  wtLabel("LAONADE AUTOTUNED"),
  woLabel("HUMAN TRAINED") {
	this->type = Layer<Dtype>::DetectionOutputLive;
	this->dispMode = 0;

	SASSERT(SLPROP(DetectionOutputLive, numClasses) > 0, "Must specify numClasses.");
	SASSERT(SLPROP(DetectionOutputLive, nmsParam).nmsThreshold >= 0, "nmsThreshold must be non negative.");
	SASSERT0(SLPROP(DetectionOutputLive, nmsParam).eta > 0.f && SLPROP(DetectionOutputLive, nmsParam).eta <= 1.f);
	this->numLocClasses = SLPROP(DetectionOutputLive, shareLocation) ? 1 : SLPROP(DetectionOutputLive, numClasses);

	cv::namedWindow(this->dispName, CV_WINDOW_NORMAL);
	cv::setWindowProperty(this->dispName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

}

template <typename Dtype>
DetectionOutputLiveLayer<Dtype>::~DetectionOutputLiveLayer() {

}

template <typename Dtype>
void DetectionOutputLiveLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();
	bool inputShapeChanged = false;
	for (int i = 0; i < this->_inputData.size(); i++) {
		if (Layer<Dtype>::_isInputShapeChanged(0)) {
			inputShapeChanged = true;
			break;
		}
	}
	if (!inputShapeChanged) return;

	this->bboxPreds.reshapeLike(this->_inputData[0]);
	if (!SLPROP(DetectionOutputLive, shareLocation)) {
		this->bboxPermute.reshapeLike(this->_inputData[0]);
	}
	this->confPermute.reshapeLike(this->_inputData[1]);

	SASSERT0(this->_inputData[0]->batches() == this->_inputData[1]->batches());
	if (this->bboxPreds.batches() != this->_inputData[0]->batches() ||
			this->bboxPreds.getCountByAxis(1) != this->_inputData[0]->getCountByAxis(1)) {
		this->bboxPreds.reshapeLike(this->_inputData[0]);
	}
	if (!SLPROP(DetectionOutputLive, shareLocation) && (this->bboxPermute.batches() != this->_inputData[0]->batches()
			|| this->bboxPermute.getCountByAxis(1) != this->_inputData[0]->getCountByAxis(1))) {
		this->bboxPermute.reshapeLike(this->_inputData[0]);
	}
	if (this->confPermute.batches() != this->_inputData[1]->batches() ||
			this->confPermute.getCountByAxis(1) != this->_inputData[1]->getCountByAxis(1)) {
		this->confPermute.reshapeLike(this->_inputData[1]);
	}

	this->numPriors = this->_inputData[2]->channels() / 4;

	SASSERT(this->numPriors * this->numLocClasses * 4 == this->_inputData[0]->channels(),
			"Number of priors must match number of location predictions.");

	SASSERT(this->numPriors * SLPROP(DetectionOutputLive, numClasses) == this->_inputData[1]->channels(),
			"Number of priors must match number of confidence predictions.");
	// num() and channels() are 1.
	vector<uint32_t> outputShape(4, 1);
	// Since the number of bboxes to be kept is unknown before nms, we manually
	// set it to (fake) 1.
	outputShape[2] = 1;
	// Each orw is a 7 dimension vector, which stores
	// [image_id, label, confidence, xmin, ymin, xmax, ymax]
	outputShape[3] = 7;
	this->_outputData[0]->reshape(outputShape);
}



void setLabel(const string& label, cv::Point org, int fontface, float scale, int thickness,
		cv::Scalar& bgColor, cv::Mat& im) {
	int baseline = 0;
	cv::Scalar fgColor = cv::Scalar(255, 255, 255);

	cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
	cv::rectangle(im, org, org + cv::Point(text.width + text.height, text.height * 2), bgColor, CV_FILLED);
	cv::putText(im, label, org + cv::Point((int)(text.height * 0.5f), (int)(text.height * 1.5f)), fontface, scale,
			//fgColor, thickness, CV_AA);
			fgColor, thickness, CV_AA);
}



template <typename Dtype>
void DetectionOutputLiveLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* locData = this->_inputData[0]->device_data();
	const Dtype* priorData = this->_inputData[2]->device_data();
	const int num = this->_inputData[0]->batches();

	// Decode predictions.
	Dtype* bboxData = this->bboxPreds.mutable_device_data();
	const int locCount = this->bboxPreds.getCount();
	const bool clipBBox = false;
	DecodeBBoxesGPU<Dtype>(locCount, locData, priorData, SLPROP(DetectionOutputLive, codeType),
			SLPROP(DetectionOutputLive, varianceEncodedInTarget), this->numPriors, SLPROP(DetectionOutputLive, shareLocation),
			this->numLocClasses, SLPROP(DetectionOutputLive, backgroundLabelId), clipBBox, bboxData);

	// Retrieve all decoded location predictions.
	const Dtype* bboxHostData;
	if (!SLPROP(DetectionOutputLive, shareLocation)) {
		Dtype* bboxPermuteData = this->bboxPermute.mutable_device_data();
		PermuteDataGPU<Dtype>(locCount, bboxData, this->numLocClasses, this->numPriors, 4,
				bboxPermuteData);
		bboxHostData = this->bboxPermute.host_data();
	} else {
		bboxHostData = this->bboxPreds.host_data();
	}

	// Retrieve all confidences.
	Dtype* confPermuteData = this->confPermute.mutable_device_data();
	PermuteDataGPU<Dtype>(this->_inputData[1]->getCount(), this->_inputData[1]->device_data(),
			SLPROP(DetectionOutputLive, numClasses), this->numPriors, 1, confPermuteData);
	const Dtype* confHostData = this->confPermute.host_data();

	int numKept = 0;
	vector<map<int, vector<int>>> allIndices;
	for (int i = 0; i < num; i++) {
		map<int, vector<int>> indices;
		int numDet = 0;
		const int confIdx = i * SLPROP(DetectionOutputLive, numClasses) * this->numPriors;
		int bboxIdx;
		if (SLPROP(DetectionOutputLive, shareLocation)) {
			bboxIdx = i * this->numPriors * 4;
		} else {
			bboxIdx = confIdx * 4;
		}
		for (int c = 0; c < SLPROP(DetectionOutputLive, numClasses); c++) {
			if (c == SLPROP(DetectionOutputLive, backgroundLabelId)) {
				// Ignore background class.
				continue;
			}
			const Dtype* curConfData = confHostData + confIdx + c * this->numPriors;
			const Dtype* curBBoxData = bboxHostData + bboxIdx;
			if (!SLPROP(DetectionOutputLive, shareLocation)) {
				curBBoxData += c * this->numPriors * 4;
			}
			ApplyNMSFast(curBBoxData, curConfData, this->numPriors, SLPROP(DetectionOutputLive, confidenceThreshold),
					SLPROP(DetectionOutputLive, nmsParam).nmsThreshold, SLPROP(DetectionOutputLive, nmsParam).eta, SLPROP(DetectionOutputLive, nmsParam).topK, &(indices[c]));
			numDet += indices[c].size();
		}
		if (SLPROP(DetectionOutputLive, keepTopK) > -1 && numDet > SLPROP(DetectionOutputLive, keepTopK)) {
			vector<pair<float, pair<int, int>>> scoreIndexPairs;
			for (map<int, vector<int>>::iterator it = indices.begin();
					it != indices.end(); it++) {
				int label = it->first;
				const vector<int>& labelIndices = it->second;
				for (int j = 0; j < labelIndices.size(); j++) {
					int idx = labelIndices[j];
					float score = confHostData[confIdx + label * this->numPriors + idx];
					scoreIndexPairs.push_back(
							std::make_pair(score, std::make_pair(label, idx)));
				}
			}

			// Keep top k results per image.
			std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(),
					SortScorePairDescend<pair<int, int>>);
			scoreIndexPairs.resize(SLPROP(DetectionOutputLive, keepTopK));
			// Store the new indices.
			map<int, vector<int>> newIndices;
			for (int j = 0; j < scoreIndexPairs.size(); j++) {
				int label = scoreIndexPairs[j].second.first;
				int idx = scoreIndexPairs[j].second.second;
				newIndices[label].push_back(idx);
			}
			allIndices.push_back(newIndices);
			numKept += SLPROP(DetectionOutputLive, keepTopK);
		} else {
			allIndices.push_back(indices);
			numKept += numDet;
		}
	}

	vector<uint32_t> outputShape(4, 1);
	outputShape[2] = numKept;
	outputShape[3] = 7;
	Dtype* outputData;
	if (numKept == 0) {
        if (SNPROP(status) == NetworkStatus::Test) {
		    STDOUT_LOG("Couldn't find any detections.");
        }
		outputShape[2] = num;
		this->_outputData[0]->reshape(outputShape);
		outputData = this->_outputData[0]->mutable_host_data();
		soooa_set<Dtype>(this->_outputData[0]->getCount(), -1, outputData);
		// Generate fake results per image.
		for (int i = 0; i < num; i++) {
			outputData[0] = i;
			outputData += 7;
		}
	} else {
		this->_outputData[0]->reshape(outputShape);
		outputData = this->_outputData[0]->mutable_host_data();
	}

	int count = 0;
	for (int i = 0; i < num; i++) {
		const int confIdx = i * SLPROP(DetectionOutputLive, numClasses) * this->numPriors;
		int bboxIdx;
		if (SLPROP(DetectionOutputLive, shareLocation)) {
			bboxIdx = i * this->numPriors * 4;
		} else {
			bboxIdx = confIdx * 4;
		}
		for (map<int, vector<int>>::iterator it = allIndices[i].begin();
				it != allIndices[i].end(); it++) {
			int label = it->first;
			vector<int>& indices = it->second;

			const Dtype* curConfData = confHostData + confIdx + label * this->numPriors;
			const Dtype* curBBoxData = bboxHostData + bboxIdx;
			if (!SLPROP(DetectionOutputLive, shareLocation)) {
				curBBoxData += label * this->numPriors * 4;
			}
			for (int j = 0; j < indices.size(); j++) {
				int idx = indices[j];
				outputData[count * 7] = i;
				outputData[count * 7 + 1] = label;
				outputData[count * 7 + 2] = curConfData[idx];
				for (int k = 0; k < 4; k++) {
					outputData[count * 7 + 3 + k] = curBBoxData[idx * 4 + k];
				}
				count++;
			}
		}
	}


	/*
	// opencv pixel format으로 복구
	// 원본 사이즈로 resize
	vector<cv::Mat> cvImgs;
	const int singleImageSize = this->_inputData[0]->getCountByAxis(1);
	const int imageHeight = 300;		// network image size
	const int imageWidth = 300;
	const int height = 480;				// final image size
	const int width = 640;

	const vector<Dtype> pixelMeans = {104.0, 117.0, 123.0};
	const Dtype* dataData = this->_inputData[0]->host_data();
	cv::Mat result;
	transformInv(1, singleImageSize, imageHeight, imageWidth, height, width,
			pixelMeans, dataData, this->temp, result);

	cv::imshow("cam", result);
	if (cv::waitKey(30) > 0) {
		cv::destroyAllWindows();
		exit(1);
	}
	*/






	const int numDet = this->_outputData[0]->height();
	const int height = this->_inputData[3]->height();
	const int width = this->_inputData[3]->width();

	cv::Mat cv_img_wt(height, width, CV_32FC3, this->_inputData[3]->mutable_host_data());
	cv_img_wt.convertTo(cv_img_wt, CV_8UC3);
	cv::Mat cv_img_wo;
	cv_img_wt.copyTo(cv_img_wo);

	int xmin, ymin, xmax, ymax;
	float fxmin, fymin, fxmax, fymax;
	float xshift, yshift, wmg, hmg;
	int xdir, ydir;




	const float wtThresh = SLPROP(DetectionOutputLive, wtThresh);
	const float woThresh = SLPROP(DetectionOutputLive, woThresh);
	const float woMargin = SLPROP(DetectionOutputLive, woMargin);
	const int woRand = SLPROP(DetectionOutputLive, woRand);

	cv::Scalar tColor = cv::Scalar(0, 0, 255);
	int tFontFace = cv::FONT_HERSHEY_DUPLEX;
	float tScale = 0.5f;
	int tThickness = 1;

	cv::Scalar bColor = cv::Scalar(255, 0, 255);
	int bFontFace = cv::FONT_HERSHEY_DUPLEX;
	float bScale = 0.5f;
	int bThickness = 1;

	int baseline = 0;


	char scoreBuf[100];
	// W/T Autotune Loop
	for (int i = 0; i < numDet; i++) {
		const int label = (int)outputData[i * 7 + 1];
		// 사람 필터링
		if (label != 15) continue;

		const float score = outputData[i * 7 + 2];
		// 낮은 스코어 필터링
		if (score < wtThresh) continue;

		xmin = (int)(outputData[i * 7 + 3] * width);
		ymin = (int)(outputData[i * 7 + 4] * height);
		xmax = (int)(outputData[i * 7 + 5] * width);
		ymax = (int)(outputData[i * 7 + 6] * height);

		cv::rectangle(cv_img_wt, cv::Point(xmin, ymin), cv::Point(xmax, ymax), bColor, 2);
		sprintf(scoreBuf, "%.2f", score);
		setLabel(string(scoreBuf), cv::Point(xmin, ymin), bFontFace, bScale, bThickness, bColor, cv_img_wt);
	}
	setLabel(this->wtLabel, cv::Point(0, 0), tFontFace, tScale, tThickness, tColor, cv_img_wt);

	// W/O Autotune Loop
	for (int i = 0; i < numDet; i++) {
		const int label = (int)outputData[i * 7 + 1];
		if (label != 15) continue;

		const float score = outputData[i * 7 + 2];
		if (score < woThresh) continue;

		// 랜덤 필터링
		if (soooa_rng_rand() % woRand == 0) continue;

		fxmin = outputData[i * 7 + 3] ;
		fymin = outputData[i * 7 + 4] ;
		fxmax = outputData[i * 7 + 5] ;
		fymax = outputData[i * 7 + 6] ;

		xshift = (fxmax - fxmin) * woMargin;
		yshift = (fymax - fymin) * woMargin;

		soooa_rng_uniform(1, -xshift, xshift, &wmg);
		soooa_rng_uniform(1, -yshift, yshift, &hmg);

		fxmin = soooa_rng_rand() % 2 ? std::max<float>(0.f, fxmin + wmg) : std::max<float>(0.f, fxmin - wmg);
		fymin = soooa_rng_rand() % 2 ? std::max<float>(0.f, fymin + hmg) : std::max<float>(0.f, fymin - hmg);
		fxmax = soooa_rng_rand() % 2 ? std::min<float>(1.f, fxmax + wmg) : std::min<float>(1.f, fxmax - wmg);
		fymax = soooa_rng_rand() % 2 ? std::min<float>(1.f, fymax + hmg) : std::min<float>(1.f, fymax - hmg);

		if (fxmin >= fxmax || fymin >= fymax) continue;

		const int xmin = (int)(fxmin * width);
		const int ymin = (int)(fymin * height);
		const int xmax = (int)(fxmax * width);
		const int ymax = (int)(fymax * height);

		cv::rectangle(cv_img_wo, cv::Point(xmin, ymin), cv::Point(xmax, ymax), bColor, 2);
		sprintf(scoreBuf, "%.2f", score);
		setLabel(string(scoreBuf), cv::Point(xmin, ymin), bFontFace, bScale, bThickness, bColor, cv_img_wo);
	}
	setLabel(this->woLabel, cv::Point(0, 0), tFontFace, tScale, tThickness, tColor, cv_img_wo);


	// side-by-side
	if (this->dispMode == 0) {
		const int x2Width = width * 2;
		cv::Mat bg = cv::Mat::zeros(cv::Size(x2Width, height), CV_8UC3);
		cv::Rect wtRoi = cv::Rect(0, 0, cv_img_wt.cols, cv_img_wt.rows);
		cv::Rect woRoi = cv::Rect(cv_img_wt.cols, 0, cv_img_wo.cols, cv_img_wo.rows);
		cv::Mat wtSubView = bg(wtRoi);
		cv::Mat woSubView = bg(woRoi);

		cv_img_wt.copyTo(wtSubView);
		cv_img_wo.copyTo(woSubView);

		cv::imshow(this->dispName, bg);
	}
	// picture in picture
	else if (this->dispMode == 1) {
		// wo image를 pip로 wt 이미지에 넣어야 함
		//cv::Mat fg = cv::imread("/home/jkim/Pictures/1480152332917.png");

		const int pipWidth = width / 4;
		const int pipHeight = height / 4;

		cv::resize(cv_img_wo, cv_img_wo, cv::Size(pipWidth, pipHeight), 0, 0, cv::INTER_LINEAR);
		cv::Rect roi = cv::Rect(cv_img_wt.cols - pipWidth, cv_img_wt.rows - pipHeight,
				pipWidth, pipHeight);
		cv::Mat subView = cv_img_wt(roi);
		cv_img_wo.copyTo(subView);

		cv::imshow(this->dispName, cv_img_wt);
	}
	// autotuned
	else if (this->dispMode == 2) {
		cv::imshow(this->dispName, cv_img_wt);
	}
	// human trained
	else if (this->dispMode == 3) {
		cv::imshow(this->dispName, cv_img_wo);
	}



	char key = cv::waitKey(5);
	if (key == 't') {
		this->dispMode++;
		if (this->dispMode > 3) {
			this->dispMode = 0;
		}
	} else if (key > 0){
		cv::destroyAllWindows();
		exit(1);
	}
}

template <typename Dtype>
void DetectionOutputLiveLayer<Dtype>::backpropagation() {

}








/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* DetectionOutputLiveLayer<Dtype>::initLayer() {
	DetectionOutputLiveLayer* layer = NULL;
	SNEW(layer, DetectionOutputLiveLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void DetectionOutputLiveLayer<Dtype>::destroyLayer(void* instancePtr) {
    DetectionOutputLiveLayer<Dtype>* layer = (DetectionOutputLiveLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void DetectionOutputLiveLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 4);
	} else {
		SASSERT0(index < 1);
	}

    DetectionOutputLiveLayer<Dtype>* layer = (DetectionOutputLiveLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DetectionOutputLiveLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DetectionOutputLiveLayer<Dtype>* layer = (DetectionOutputLiveLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void DetectionOutputLiveLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	DetectionOutputLiveLayer<Dtype>* layer = (DetectionOutputLiveLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void DetectionOutputLiveLayer<Dtype>::backwardTensor(void* instancePtr) {
	DetectionOutputLiveLayer<Dtype>* layer = (DetectionOutputLiveLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void DetectionOutputLiveLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool DetectionOutputLiveLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 3)
        return false;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 7;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t DetectionOutputLiveLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    size_t size = 0;
    const int input0Count = tensorCount(inputShape[0]);
    const int input1Count = tensorCount(inputShape[1]);
    
    // bboxPreds
    size += ALIGNUP(sizeof(Dtype) * input0Count, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    // bboxPermute
	if (!SLPROP(DetectionOutput, shareLocation)) {
        size += ALIGNUP(sizeof(Dtype) * input0Count, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
	}
    
    // confPermute
    size += ALIGNUP(sizeof(Dtype) * input1Count, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    return size;
}

template class DetectionOutputLiveLayer<float>;
