/*
 * RoIDataLayer.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: jkim
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "frcnn_common.h"
#include "BaseLayer.h"
#include "RoIDataLayer.h"
#include "ImagePackDataSet.h"
#include "PascalVOC.h"
#include "RoIDBUtil.h"
#include "MockDataSet.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "MemoryMgmt.h"

#define ROIDATALAYER_LOG 0

using namespace std;


template <typename Dtype>
RoIDataLayer<Dtype>::RoIDataLayer()
: InputLayer<Dtype>(),
  dataReader(SLPROP(RoIData, source)),
  dataTransformer(&SLPROP(RoIData, dataTransformParam)) {
	this->type = Layer<Dtype>::RoIData;
	const string dataSetName = SLPROP(RoIData, dataSetName);
	if (dataSetName.empty()) {
		this->dataReader.selectDataSetByIndex(0);
	} else {
		this->dataReader.selectDataSetByName(dataSetName);
	}

	// roidb 벡터속의 하나의 roidb는 하나의 이미지 정보에 해당
	cout << this->dataReader.getNumData() << " roidb entries ... " << endl;

	int numClasses = 0;
	if (SLPROP(RoIData, numClasses) > 0) {
		numClasses = SLPROP(RoIData, numClasses);
	} else if (this->dataReader.getHeader().labelItemList.size() > 0) {
		numClasses = this->dataReader.getHeader().labelItemList.size();
	} else {
		SASSERT(false, "one of numClasses and labelItemList size should be larger than 0.");
	}

	SASSERT0(TRAIN_BBOX_NORMALIZE_TARGETS_PRECOMPUTED);
	np_tile(TRAIN_BBOX_NORMALIZE_MEANS, numClasses, this->bboxMeans);
	np_tile(TRAIN_BBOX_NORMALIZE_STDS, numClasses, this->bboxStds);

	DataTransformParam& dataTransformParam = this->dataTransformer.param;
	dataTransformParam.resizeParam = SLPROP(RoIData, resizeParam);
	dataTransformParam.resizeParam.updateInterpMode();
}

template <typename Dtype>
RoIDataLayer<Dtype>::~RoIDataLayer() {}



template <typename Dtype>
void RoIDataLayer<Dtype>::reshape() {
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
					{TRAIN_IMS_PER_BATCH, 3, vec_max(TRAIN_SCALES), TRAIN_MAX_SIZE};
			this->_inputData[0]->reshape(dataShape);
			this->_inputShape[0] = dataShape;
		}
		// "im_info"
		else if (i == 1) {
			const vector<uint32_t> iminfoShape = {1, 1, 1, 3};
			this->_inputShape[1] = iminfoShape;
			this->_inputData[1]->reshape(iminfoShape);
		}
		// "gt_boxes"
		else if (i == 2) {
			const vector<uint32_t> gtboxesShape = {1, 1, 1, 5};
			this->_inputShape[2] = gtboxesShape;
			this->_inputData[2]->reshape(gtboxesShape);
		}
	}
}




template <typename Dtype>
void RoIDataLayer<Dtype>::feedforward() {
	reshape();
	load_batch();
}

template <typename Dtype>
void RoIDataLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();
	load_batch();
}

template <typename Dtype>
void RoIDataLayer<Dtype>::load_batch() {
	AnnotatedDatum* annoDatum;
	struct timespec startTime;
	if (SPARAM(INPUT_DATA_PROVIDER_MEASURE_PERFORMANCE)) {
		SPERF_START(DATAINPUT_ACCESS_TIME, &startTime);
	}
	if ((WorkContext::curBootMode == BootMode::ServerClientMode) &&
		SPARAM(USE_INPUT_DATA_PROVIDER)) {
		void* elem = NULL;
		while (true) {
			elem = InputDataProvider::getData(this->inputPool, false);
			if (elem == NULL) {
				usleep(SPARAM(INPUT_DATA_PROVIDER_CALLER_RETRY_TIME_USEC));
			} else {
				break;
			}
		}
		annoDatum = (class AnnotatedDatum*)elem;
	} else {
		annoDatum = this->dataReader.getNextData();
	}
	if (SPARAM(INPUT_DATA_PROVIDER_MEASURE_PERFORMANCE)) {
		SPERF_END(DATAINPUT_ACCESS_TIME, startTime);
	}

	// Given a roidb, construct a minibatch sampled from it.
	const uint32_t numImages = 1;
	// Sample random scales to use for each image in this batch
	vector<uint32_t> randomScaleInds;
	npr_randint(0, TRAIN_SCALES.size(), numImages, randomScaleInds);

	uint32_t roisPerImage = TRAIN_BATCH_SIZE / numImages;
	uint32_t fgRoisPerImage = np_round(TRAIN_FG_FRACTION * roisPerImage);


	// compute resize dimension
	uint32_t targetSize = TRAIN_SCALES[randomScaleInds[0]];
	uint32_t width = annoDatum->width;
	uint32_t height = annoDatum->height;
	uint32_t channels = annoDatum->channels;

	const vector<uint32_t> imShape = {width, height, channels};
	uint32_t imSizeMin = np_min(imShape, 0, 2);
	uint32_t imSizeMax = np_max(imShape, 0, 2);
	float imScale = float(targetSize) / float(imSizeMin);
	// Prevent the biggest axis from being more than MAX_SIZE
	if (np_round(imScale * imSizeMax) > TRAIN_MAX_SIZE) {
		imScale = float(TRAIN_MAX_SIZE) / float(imSizeMax);
	}

	ResizeParam& resizeParam = this->dataTransformer.param.resizeParam;
	resizeParam.height = std::round(height * imScale);
	resizeParam.width = std::round(width * imScale);

	this->_inputShape[0] = {numImages, channels, resizeParam.height, resizeParam.width};
	this->_outputData[0]->reshape(this->_inputShape[0]);

	vector<AnnotationGroup> transformedAnnoVec;
	this->dataTransformer.transform(annoDatum, this->_outputData[0], 0, transformedAnnoVec);

	// Count the number of bboxes.
	int numBBoxes = 0;
	for (int g = 0; g < transformedAnnoVec.size(); g++) {
		//numBBoxes += transformedAnnoVec[g].annotations.size();
		for (int a = 0; a < transformedAnnoVec[g].annotations.size(); a++) {
			if (!transformedAnnoVec[g].annotations[a].bbox.difficult) {
				numBBoxes += 1;
			}
		}
	}


	// im_info
	Dtype* imInfoData = this->_inputData[1]->mutable_host_data();
	imInfoData[0] = (Dtype)this->_inputShape[0][2];
	imInfoData[1] = (Dtype)this->_inputShape[0][3];
	imInfoData[2] = (Dtype)imScale;




	// gt boxes: (x1, y1, x2, y2, cls)

	vector<uint32_t> gtBoxesShape(4);
	gtBoxesShape[0] = 1;
	gtBoxesShape[1] = 1;
	gtBoxesShape[3] = 5;
	//cout << "numBBoxes: " << numBBoxes << endl;
	if (numBBoxes == 0) {
		// Store all -1 in the label.
		gtBoxesShape[2] = 1;
		this->_outputData[2]->reshape(gtBoxesShape);
		soooa_set<Dtype>(8, -1, this->_outputData[2]->mutable_host_data());
	} else {
		// Reshape the label and store the annotation.
		gtBoxesShape[2] = numBBoxes;
		this->_outputData[2]->reshape(gtBoxesShape);
		Dtype* gtBBoxesData = this->_outputData[2]->mutable_host_data();
		int idx = 0;

		for (int g = 0; g < transformedAnnoVec.size(); g++) {
			const AnnotationGroup& annoGroup = transformedAnnoVec[g];
			for (int a = 0; a < annoGroup.annotations.size(); a++) {
				if (!annoGroup.annotations[a].bbox.difficult) {
					const Annotation_s& anno = annoGroup.annotations[a];
					const NormalizedBBox& bbox = anno.bbox;
					gtBBoxesData[idx++] = bbox.xmin * this->_outputData[0]->getShape(3);
					gtBBoxesData[idx++] = bbox.ymin * this->_outputData[0]->getShape(2);
					gtBBoxesData[idx++] = bbox.xmax * this->_outputData[0]->getShape(3);
					gtBBoxesData[idx++] = bbox.ymax * this->_outputData[0]->getShape(2);
					gtBBoxesData[idx++] = annoGroup.group_label;
				}
			}
		}
	}
}





template<typename Dtype>
int RoIDataLayer<Dtype>::getNumTrainData() {
    return this->dataReader.getNumData();
}

template<typename Dtype>
int RoIDataLayer<Dtype>::getNumTestData() {
    return 0;
}

template<typename Dtype>
void RoIDataLayer<Dtype>::shuffleTrainDataSet() {

}







/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* RoIDataLayer<Dtype>::initLayer() {
	RoIDataLayer* layer = NULL;
	SNEW(layer, RoIDataLayer<Dtype>);
	SASSUME0(layer != NULL);

	if ((WorkContext::curBootMode == BootMode::ServerClientMode) &&
		SPARAM(USE_INPUT_DATA_PROVIDER)) {
		const string& name = "RoIDataLayer";
		InputDataProvider::addPool(WorkContext::curNetworkID, WorkContext::curDOPID,
			name, DRType::DatumType, (void*)&layer->dataReader);
		layer->inputPool = InputDataProvider::getInputPool(WorkContext::curNetworkID,
				WorkContext::curDOPID, name);
	}
    return (void*)layer;
}

template<typename Dtype>
void RoIDataLayer<Dtype>::destroyLayer(void* instancePtr) {
    RoIDataLayer<Dtype>* layer = (RoIDataLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void RoIDataLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
    SASSERT0(index < 3);

    RoIDataLayer<Dtype>* layer = (RoIDataLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool RoIDataLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    RoIDataLayer<Dtype>* layer = (RoIDataLayer<Dtype>*)instancePtr;
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
void RoIDataLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	RoIDataLayer<Dtype>* layer = (RoIDataLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void RoIDataLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void RoIDataLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool RoIDataLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    // data
    TensorShape outputShape1;
    outputShape1.N = TRAIN_IMS_PER_BATCH;
    outputShape1.C = 3;
    outputShape1.H = vec_max(TRAIN_SCALES);
    outputShape1.W = TRAIN_MAX_SIZE;
    outputShape.push_back(outputShape1);

    // im_info
    TensorShape outputShape2;
    outputShape2.N = 1;
    outputShape2.C = 1;
    outputShape2.H = 1;
    outputShape2.W = 3;
    outputShape.push_back(outputShape2);

    // gt_boxes
    TensorShape outputShape3;
    outputShape3.N = 1;
    outputShape3.C = 1;
    outputShape3.H = 1;
    outputShape3.W = 5;
    outputShape.push_back(outputShape3);

    return true;
}

template<typename Dtype>
uint64_t RoIDataLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class RoIDataLayer<float>;
