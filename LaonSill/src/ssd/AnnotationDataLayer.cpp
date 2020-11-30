/*
 * AnnotationDataLayer.cpp
 *
 *  Created on: Apr 19, 2017
 *      Author: jkim
 */

#include "AnnotationDataLayer.h"
#include "tinyxml2/tinyxml2.h"
#include "StdOutLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;


template <typename Dtype>
AnnotationDataLayer<Dtype>::AnnotationDataLayer()
: InputLayer<Dtype>(),
  /*
  flip(builder->_flip),
  imageHeight(builder->_imageHeight),
  imageWidth(builder->_imageWidth),
  imageSetPath(builder->_imageSetPath),
  baseDataPath(builder->_baseDataPath),
  labelMap(builder->_labelMapPath),
  pixelMeans(builder->_pixelMeans),
  bShuffle(builder->_shuffle),
  */
  labelMap(SLPROP(AnnotationData, labelMapPath)),
  data("data", true) {
	this->type = Layer<Dtype>::AnnotationData;

	this->data.reshape({1, SLPROP(AnnotationData, imageHeight),
		SLPROP(AnnotationData, imageWidth), 3});
	const bool live = SLPROP(AnnotationData, live);

	if (!live) {
		this->labelMap.build();

		loadODRawDataPath();
		//loadODRawDataIm();
		loadODRawDataAnno();

		loadODMetaData();

		this->perm.resize(this->odMetaDataList.size());
		std::iota(this->perm.begin(), this->perm.end(), 0);
		shuffle();
	} else {
		SASSERT(SNPROP(useCompositeModel) || SNPROP(status) == NetworkStatus::Test,
				"AnnotationDataLayer in live mode can be run only in Test Status");
	}
}

template <typename Dtype>
AnnotationDataLayer<Dtype>::~AnnotationDataLayer() {

}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		// AnnotationDataLayer에서 inputShape를 확인하지 않을 것이므로
		// inputShape를 별도로 갱신하지는 않음.
		// for "data" Data
		this->_inputData[0]->reshape({SNPROP(batchSize), 3, SLPROP(AnnotationData, imageHeight), SLPROP(AnnotationData, imageWidth)});

		if (!SLPROP(AnnotationData, live)) {
			// for "label" Data
			// 1. item_id 2. group_label 3. instance_id
			// 4. xmin 5. ymin 6. xmax 7. ymax 8. difficult
			// cf. [2]: numBBs
			// numBBs를 max 추정하여 미리 충분히 잡아 두면
			// 실행시간에 추가 메모리 할당이 없을 것. batch * 10으로 추정
			//this->_inputData[1]->reshape({1, 1, SNPROP(batchSize) * 10, 8});
			this->_inputData[1]->reshape({1, 1, 2, 8});
		}
	}
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::feedImage(const int channels, const int height,
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
int AnnotationDataLayer<Dtype>::getNumTrainData() {
	if (!SLPROP(AnnotationData, live)) {
		return this->odMetaDataList.size();
	} else {
		return 1;
	}
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::shuffleTrainDataSet() {
	return;
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::feedforward() {
	reshape();
	getNextMiniBatch();
	//verifyData();
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();
	getNextMiniBatch();
	//verifyData();
}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::loadODRawDataPath() {
	ifstream ifs(SLPROP(AnnotationData, imageSetPath).c_str(), ios::in);
	if (!ifs.is_open()) {
		cout << "no such file: " << SLPROP(AnnotationData, imageSetPath) << endl;
		exit(1);
	}

	ODRawData<Dtype> odRawData;
	string imPath;
	string annoPath;
	while (ifs >> imPath >> annoPath) {
		odRawData.imPath = imPath;
		odRawData.annoPath = annoPath;
		this->odRawDataList.push_back(odRawData);
	}

	ifs.close();
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::loadODRawDataIm() {
	const int numODRawData = this->odRawDataList.size();
	for (int i = 0; i < numODRawData; i++) {
		if (i % 1000 == 0) {
			cout << "loadODRawDataIm(): " << i << endl;
		}

		ODRawData<Dtype>& odRawData = this->odRawDataList[i];
		cv::Mat im = cv::imread(SLPROP(AnnotationData, baseDataPath) + odRawData.imPath);
		im.convertTo(im, CV_32FC3);

		//float imHeightScale = float(SLPROP(AnnotationData, imageHeight)) / float(im.rows);
		//float imWidthScale = float(SLPROP(AnnotationData, imageWidth)) / float(im.cols);
		cv::resize(im, im, cv::Size(SLPROP(AnnotationData, imageWidth), SLPROP(AnnotationData, imageHeight)), 0, 0, CV_INTER_LINEAR);

		odRawData.im = im;
	}
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::loadODRawDataAnno() {
	const int numODRawData = this->odRawDataList.size();
	for (int i = 0; i < numODRawData; i++) {
		ODRawData<Dtype>& odRawData = this->odRawDataList[i];
		readAnnotation(odRawData);

		//normalize bounding box coordinates
		for (int j = 0; j < odRawData.boundingBoxes.size(); j++) {
			BoundingBox<Dtype>& bb = odRawData.boundingBoxes[j];
			bb.buf[0] = 0;									// item_id
			bb.buf[1] = Dtype(bb.label);					// group_label
			bb.buf[2] = 0;									// instance_id
			bb.buf[3] = Dtype(bb.xmin) / odRawData.width;	// xmin
			bb.buf[4] = Dtype(bb.ymin) / odRawData.height;	// ymin
			bb.buf[5] = Dtype(bb.xmax) / odRawData.width;	// xmax
			bb.buf[6] = Dtype(bb.ymax) / odRawData.height;	// ymax
			bb.buf[7] = Dtype(bb.diff);						// difficult
		}
	}
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::readAnnotation(ODRawData<Dtype>& odRawData) {
	tinyxml2::XMLDocument annotationDocument;
	tinyxml2::XMLNode* annotationNode;

	const string filePath = SLPROP(AnnotationData, baseDataPath) + odRawData.annoPath;
	annotationDocument.LoadFile(filePath.c_str());
	annotationNode = annotationDocument.FirstChild();

	// filename
	//tinyxml2::XMLElement* filenameElement = annotationNode->FirstChildElement("filename");
	//annotation.filename = filenameElement->GetText();

	// size
	tinyxml2::XMLElement* sizeElement = annotationNode->FirstChildElement("size");
	sizeElement->FirstChildElement("width")->QueryIntText((int*)&odRawData.width);
	sizeElement->FirstChildElement("height")->QueryIntText((int*)&odRawData.height);
	sizeElement->FirstChildElement("depth")->QueryIntText((int*)&odRawData.depth);

	// object
	for (tinyxml2::XMLElement* objectElement =
			annotationNode->FirstChildElement("object"); objectElement != 0;
			objectElement = objectElement->NextSiblingElement("object")) {
		BoundingBox<Dtype> boundingBox;
		boundingBox.name = objectElement->FirstChildElement("name")->GetText();
		boundingBox.label = this->labelMap.convertLabelToInd(boundingBox.name);
		objectElement->FirstChildElement("difficult")
					 ->QueryIntText((int*)&boundingBox.diff);

		tinyxml2::XMLElement* bndboxElement = objectElement->FirstChildElement("bndbox");
		bndboxElement->FirstChildElement("xmin")->QueryIntText((int*)&boundingBox.xmin);
		bndboxElement->FirstChildElement("ymin")->QueryIntText((int*)&boundingBox.ymin);
		bndboxElement->FirstChildElement("xmax")->QueryIntText((int*)&boundingBox.xmax);
		bndboxElement->FirstChildElement("ymax")->QueryIntText((int*)&boundingBox.ymax);

		//if (boundingBox.diff == 0) {
			odRawData.boundingBoxes.push_back(boundingBox);
		//}
	}
}



template <typename Dtype>
void AnnotationDataLayer<Dtype>::loadODMetaData() {
	const int numODRawData = this->odRawDataList.size();

	ODMetaData<Dtype> odMetaData;
	for (int i = 0; i < numODRawData; i++) {
		odMetaData.rawIdx = i;
		odMetaData.flip = false;

		this->odMetaDataList.push_back(odMetaData);

		if (SLPROP(AnnotationData, flip)) {
			odMetaData.flip = true;
			this->odMetaDataList.push_back(odMetaData);
		}
	}
}



template <typename Dtype>
void AnnotationDataLayer<Dtype>::shuffle() {
	if (SLPROP(AnnotationData, shuffle)) {
		std::random_shuffle(this->perm.begin(), this->perm.end());
	}
	//cout << "***shuffle is temporaray disabled ... " << endl;
	this->cur = 0;
}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::getNextMiniBatch() {
	if (!SLPROP(AnnotationData, live)) {
		vector<int> inds;
		getNextMiniBatchInds(inds);
		getMiniBatch(inds);
	} else {
		cv::Mat cv_img(this->height, this->width, CV_32FC3, this->image);
		SASSERT(cv_img.data, "Could not decode datum.");

		int resizeHeight = SLPROP(AnnotationData, imageHeight);
		int resizeWidth = SLPROP(AnnotationData, imageWidth);

		Dtype* dataData = this->_outputData[0]->mutable_host_data();
		const vector<uint32_t> dataShape = {1, (uint32_t)resizeHeight, (uint32_t)resizeWidth, 3};

		cv::resize(cv_img, cv_img, cv::Size(resizeWidth, resizeHeight), 0, 0, CV_INTER_LINEAR);

		// subtract mean
		float* imPtr = (float*)cv_img.data;
		int n = cv_img.rows * cv_img.cols * cv_img.channels();
		for (int j = 0; j < n; j += 3) {
			imPtr[j + 0] -= SLPROP(AnnotationData, mean)[0];
			imPtr[j + 1] -= SLPROP(AnnotationData, mean)[1];
			imPtr[j + 2] -= SLPROP(AnnotationData, mean)[2];
		}

		// data
		this->data.reshape(dataShape);
		this->data.set_host_data((Dtype*)cv_img.data);
		this->data.transpose({0, 3, 1, 2}); // BGR,BGR,... to BB..,GG..,RR..

		std::copy(this->data.host_data(), this->data.host_data() + this->data.getCount(),
				dataData);
	}
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::getNextMiniBatchInds(vector<int>& inds) {
	if (this->cur + SNPROP(batchSize) > this->odMetaDataList.size()) {
		shuffle();
	}

	inds.clear();
	inds.insert(inds.end(), this->perm.begin() + this->cur,
			this->perm.begin() + this->cur + SNPROP(batchSize));

	this->cur += SNPROP(batchSize);
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::getMiniBatch(const vector<int>& inds) {

	// Count total bounding boxes in this batch.
	uint32_t totalBBs = 0;
	for (int i = 0; i < inds.size(); i++) {
		ODMetaData<Dtype>& odMetaData = this->odMetaDataList[inds[i]];
		ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];
		totalBBs += odRawData.boundingBoxes.size();
	}

	// "data"의 경우 shape가 초기에 결정, reshape가 필요없음.
	//this->_outputData[0]->reshape({SNPROP(batchSize), 3, SLPROP(AnnotationData, imageHeight),
	//	SLPROP(AnnotationData, imageWidth)});
	// "label"의 경우 현재 batch의 bounding box 수에 따라 shape 변동.
	this->_outputData[1]->reshape({1, 1, totalBBs, 8});

	Dtype* dataData = this->_outputData[0]->mutable_host_data();
	Dtype* labelData = this->_outputData[1]->mutable_host_data();

	// "data" 1개에 대한 shape. cv::Mat의 경우 BGR, BGR, ... 의 구조.
	const vector<uint32_t> dataShape = {1, SLPROP(AnnotationData, imageHeight), SLPROP(AnnotationData, imageWidth), 3};
	int bbIdx = 0;
	Dtype buf[8];
	for (int i = 0; i < inds.size(); i++) {
		ODMetaData<Dtype>& odMetaData = this->odMetaDataList[inds[i]];
		ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];


		if (odRawData.im.empty()) {
			cv::Mat im = cv::imread(SLPROP(AnnotationData, baseDataPath) + odRawData.imPath);
			im.convertTo(im, CV_32FC3);
			cv::resize(im, im, cv::Size(SLPROP(AnnotationData, imageWidth), SLPROP(AnnotationData, imageHeight)), 0, 0, CV_INTER_LINEAR);
			odRawData.im = im;
		}

		// transform image ... 추가 transform option 추가될 경우 여기서 처리.
		// 현재는 flip만 적용되어 있음.
		// flip image
		cv::Mat copiedIm;
		if (odMetaData.flip) {
			// 1 means flipping around y-axis
			cv::flip(odRawData.im, copiedIm, 1);
		} else {
			odRawData.im.copyTo(copiedIm);
		}

		// subtract mean
		float* imPtr = (float*)copiedIm.data;
		int n = copiedIm.rows * copiedIm.cols * copiedIm.channels();
		for (int j = 0; j < n; j += 3) {
			imPtr[j + 0] -= SLPROP(AnnotationData, mean)[0];
			imPtr[j + 1] -= SLPROP(AnnotationData, mean)[1];
			imPtr[j + 2] -= SLPROP(AnnotationData, mean)[2];
		}

		// data
		this->data.reshape(dataShape);
		this->data.set_host_data((Dtype*)copiedIm.data);
		this->data.transpose({0, 3, 1, 2}); // BGR,BGR,... to BB..,GG..,RR..

		std::copy(this->data.host_data(), this->data.host_data() + this->data.getCount(),
				dataData + i * this->data.getCount());

		// label
		const int numBBs = odRawData.boundingBoxes.size();
		for (int j = 0; j < numBBs; j++) {
			buildLabelData(odMetaData, j, buf);
			buf[0] = i;
			// XXX: for debugging.
			//buf[2] = inds[i];
			std::copy(buf, buf + 8, labelData + bbIdx * 8);
			bbIdx++;
		}
	}

	//this->_printOn();
	//this->_outputData[0]->print_data({}, false);
	//this->_outputData[1]->print_data({}, false, -1);
	//this->_printOff();
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::buildLabelData(ODMetaData<Dtype>& odMetaData, int bbIdx,
		Dtype buf[8]) {
	ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];
	BoundingBox<Dtype>& bb = odRawData.boundingBoxes[bbIdx];

	std::copy(bb.buf, bb.buf + 8, buf);

	// flip horizontally only
	if (odMetaData.flip) {
		buf[3] = 1.0 - bb.buf[5];				// xmin
		buf[5] = 1.0 - bb.buf[3];				// xmax
	}
}




template <typename Dtype>
void AnnotationDataLayer<Dtype>::verifyData() {
	STDOUT_LOG("VERIFYING DATA ... ");
	const string windowName = "AnnotationDataLayer::verifyData()";
	const int batches = SNPROP(batchSize);
	const int singleImageSize = this->_outputData[0]->getCountByAxis(1);
	const Dtype* dataData = this->_outputData[0]->host_data();
	const Dtype* labelData = this->_outputData[1]->host_data();
	const uint32_t numBBs = this->_outputData[1]->getShape(2);

	Dtype* data = this->data.mutable_host_data();
	Dtype buf[8];
	int bbIdx = 0;
	for (int i = 0; i < batches; i++) {
		int idx = int(labelData[bbIdx * 8 + 2]);
		ODMetaData<Dtype>& odMetaData = this->odMetaDataList[idx];
		ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];
		//odRawData.displayBoundingBoxes(SLPROP(AnnotationData, baseDataPath), this->labelMap.colorList);

		this->data.reshape({1, 3, SLPROP(AnnotationData, imageHeight), SLPROP(AnnotationData, imageWidth)});
		std::copy(dataData + i * singleImageSize, dataData + (i + 1) * singleImageSize, data);

		// transpose
		this->data.transpose({0, 2, 3, 1});

		// pixel mean
		for (int j = 0; j < singleImageSize; j += 3) {
			data[j + 0] += SLPROP(AnnotationData, mean)[0];
			data[j + 1] += SLPROP(AnnotationData, mean)[1];
			data[j + 2] += SLPROP(AnnotationData, mean)[2];
		}

		cv::Mat im = cv::Mat(SLPROP(AnnotationData, imageHeight), SLPROP(AnnotationData, imageWidth), CV_32FC3, data);
		cv::resize(im, im, cv::Size(odRawData.width, odRawData.height), 0, 0, CV_INTER_LINEAR);


		while (bbIdx < numBBs) {
			idx = int(labelData[bbIdx * 8 + 0]);
			if (idx == i) {
				std::copy(labelData + bbIdx * 8, labelData + (bbIdx + 1) * 8, buf);
				//printArray(buf, 8);

				// 1. label 4. xmin 5. ymin 6. xmax 7. ymax
				int label = int(buf[1]);
				int xmin = int(buf[3] * odRawData.width);
				int ymin = int(buf[4] * odRawData.height);
				int xmax = int(buf[5] * odRawData.width);
				int ymax = int(buf[6] * odRawData.height);

				cv::rectangle(im, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
						this->labelMap.colorList[label], 2);
				cv::putText(im, this->labelMap.convertIndToLabel(label),
						cv::Point(xmin, ymin+15.0f), 2, 0.5f,
						this->labelMap.colorList[label]);
				bbIdx++;
			} else {
				break;
			}
		}

		im.convertTo(im, CV_8UC3);
		cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName, im);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::printMat(cv::Mat& im, int type) {

	if (type == CV_32F) {
		float* data = (float*)im.data;

		//for (int r = 0; r < )

		for (int i = 0; i < 9; i++) {
			cout << data[i] << ", ";
		}
		cout << endl;
	} else if (type == CV_8U) {
		uchar* data = (uchar*)im.data;
		for (int i = 0; i < 9; i++) {
			cout << uint32_t(data[i]) << ", ";
		}
		cout << endl;
	}
}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::printArray(Dtype* array, int n) {

	for (int i = 0; i < n; i++) {
		cout << array[i] << ", ";
	}
	cout << endl;
}















/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* AnnotationDataLayer<Dtype>::initLayer() {
	AnnotationDataLayer* layer = NULL;
	SNEW(layer, AnnotationDataLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void AnnotationDataLayer<Dtype>::destroyLayer(void* instancePtr) {
    AnnotationDataLayer<Dtype>* layer = (AnnotationDataLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void AnnotationDataLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
    SASSERT0(index < 3);

    AnnotationDataLayer<Dtype>* layer = (AnnotationDataLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool AnnotationDataLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    AnnotationDataLayer<Dtype>* layer = (AnnotationDataLayer<Dtype>*)instancePtr;
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
void AnnotationDataLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	AnnotationDataLayer<Dtype>* layer = (AnnotationDataLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void AnnotationDataLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void AnnotationDataLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool AnnotationDataLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    TensorShape outputShape1;
    outputShape1.N = SNPROP(batchSize);
    outputShape1.C = 3;
    outputShape1.H = SLPROP(AnnotationData, imageHeight);
    outputShape1.W = SLPROP(AnnotationData, imageWidth);

    if (!SLPROP(AnnotationData, live)) {
        outputShape1.N = 1;
        outputShape1.C = 1;
        outputShape1.H = 2;
        outputShape1.W = 8;
    }
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t AnnotationDataLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class AnnotationDataLayer<float>;
