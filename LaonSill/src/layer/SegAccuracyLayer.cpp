/*
 * SegAccuracyLayer.cpp
 *
 *  Created on: Aug 7, 2017
 *      Author: jkim
 */

#include "SegAccuracyLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "MathFunctions.h"
#include "MemoryMgmt.h"


using namespace std;

template <typename Dtype>
SegAccuracyLayer<Dtype>::SegAccuracyLayer()
: MeasureLayer<Dtype>() {
	this->type = Layer<Dtype>::SegAccuracy;

	this->confusionMatrix.clear();
	this->confusionMatrix.resize(this->_inputData[0]->channels());

	for (int c = 0; c < SLPROP(SegAccuracy, ignoreLabel).size(); c++) {
		this->ignoreLabel.insert(SLPROP(SegAccuracy, ignoreLabel)[c]);
	}
}

template <typename Dtype>
SegAccuracyLayer<Dtype>::~SegAccuracyLayer() {

}

template <typename Dtype>
Dtype SegAccuracyLayer<Dtype>::getAccuracy() {
	Dtype accuracy = this->_outputData[0]->host_data()[0];
	return accuracy;
}

template<typename Dtype>
Dtype SegAccuracyLayer<Dtype>::measure() {
    return getAccuracy();
}

template<typename Dtype>
Dtype SegAccuracyLayer<Dtype>::measureAll() {
    // TODO: implement!!!
    SASSERT0(false);
    return getAccuracy();
}


template <typename Dtype>
void SegAccuracyLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0) &&
			!Layer<Dtype>::_isInputShapeChanged(1))
		return;

	SASSERT(1 <= this->_inputData[0]->channels(),
			"top_k must be less than or equal to the number of channels (classes.)");
	SASSERT(this->_inputData[0]->batches() == this->_inputData[1]->batches(),
			"The data and label should have the same batches.");
	SASSERT(this->_inputData[1]->batches() == 1,
			"The label should have one channel.");
	SASSERT(this->_inputData[0]->height() == this->_inputData[1]->height(),
				"The data should have the same height as label.");
	SASSERT(this->_inputData[0]->width() == this->_inputData[1]->width(),
				"The data should have the same width as label.");

	this->_outputData[0]->reshape({1, 1, 1, 3});
	this->_inputShape[0] = this->_inputData[0]->getShape();
	this->_inputShape[1] = this->_inputData[1]->getShape();
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* inputData = this->_inputData[0]->host_data();
	const Dtype* inputLabel = this->_inputData[1]->host_data();
	int batches = this->_inputData[0]->batches();
	int channels = this->_inputData[0]->channels();
	int height = this->_inputData[0]->height();
	int width = this->_inputData[0]->width();

	int dataIndex;
	int labelIndex;
	int topK = 1;		// only support for topK = 1

	// remove old predictions if reset() flag is true
	if (SLPROP(SegAccuracy, reset)) {
		this->confusionMatrix.clear();
	}

	for (int i = 0; i < batches; i++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				// Top-k accuracy
				std::vector<std::pair<Dtype, int>> inputDataVector;

				for (int c = 0; c < channels; c++) {
					dataIndex = (c * height + h) * width + w;
					inputDataVector.push_back(std::make_pair(inputData[dataIndex], c));
				}

				std::partial_sort(
						inputDataVector.begin(), inputDataVector.begin() + topK,
						inputDataVector.end(), std::greater<std::pair<Dtype, int>>());

				// check if true label is in top k predictions
				labelIndex = h * width + w;
				const int gtLabel = static_cast<int>(inputLabel[labelIndex]);

				if (this->ignoreLabel.count(gtLabel) != 0) {
					// ignore the pixel with this gtLabel
					continue;
				} else if (gtLabel >= 0 && gtLabel < channels) {
					// current position is not "255" indicating ambiguous position
					this->confusionMatrix.accumulate(gtLabel, inputDataVector[0].second);
				} else {
					SASSERT(false, "Unexpected label %d. num: %d. row: %d. col %d.",
							gtLabel, i, h, w);
				}
			}
		}
		inputData += this->_inputData[0]->offset(1);
		inputLabel += this->_inputData[1]->offset(1);
	}

	this->_outputData[0]->mutable_host_data()[0] = (Dtype)this->confusionMatrix.accuracy();
	this->_outputData[0]->mutable_host_data()[1] = (Dtype)this->confusionMatrix.avgRecall(false);
	this->_outputData[0]->mutable_host_data()[2] = (Dtype)this->confusionMatrix.avgJaccard();

}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::backpropagation() {
	for (int i = 0; i < SLPROP_BASE(propDown).size(); i++) {
		if (SLPROP_BASE(propDown)[i]) {
			SASSERT(false, "Not Implemented Yet");
		}
	}
}

/*
template <typename Dtype>
void SegAccuracyLayer<Dtype>::() {

}
*/




/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* SegAccuracyLayer<Dtype>::initLayer() {
	SegAccuracyLayer* layer = NULL;
	SNEW(layer, SegAccuracyLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void SegAccuracyLayer<Dtype>::destroyLayer(void* instancePtr) {
    SegAccuracyLayer<Dtype>* layer = (SegAccuracyLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void SegAccuracyLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	if (isInput) {
		SASSERT0(index < 2);
	} else {
		SASSERT0(index < 1);
	}

    SegAccuracyLayer<Dtype>* layer = (SegAccuracyLayer<Dtype>*)instancePtr;
    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool SegAccuracyLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    SegAccuracyLayer<Dtype>* layer = (SegAccuracyLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void SegAccuracyLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	SegAccuracyLayer<Dtype>* layer = (SegAccuracyLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void SegAccuracyLayer<Dtype>::backwardTensor(void* instancePtr) {
	SegAccuracyLayer<Dtype>* layer = (SegAccuracyLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void SegAccuracyLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool SegAccuracyLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() < 2)
        return false;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 3;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t SegAccuracyLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class SegAccuracyLayer<float>;
