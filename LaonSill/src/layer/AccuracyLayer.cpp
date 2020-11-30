/*
 * AccuracyLayer.cpp
 *
 *  Created on: Apr 25, 2017
 *      Author: jkim
 */

#include "AccuracyLayer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "StdOutLog.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
AccuracyLayer<Dtype>::AccuracyLayer()
: MeasureLayer<Dtype>() {
	const vector<string>& inputs = SLPROP_BASE(input);
	const vector<string>& outputs = SLPROP_BASE(output);

	SASSERT0(inputs.size() == 2 && outputs.size() == 1);
	SASSERT((inputs[0] != outputs[0]) &&
			(inputs[1] != outputs[0]),
			"this layer does not allow in-place computation.");

	//this->topK = builder->_topK;
	//this->labelAxis = builder->_axis;

	this->hasIgnoreLabel = (SLPROP(Accuracy, ignoreLabel) >= 0);
	//if (this->hasIgnoreLabel) {
	//	this->ignoreLabel = builder->_ignoreLabel;
	//}

	this->numCorrect = 0;
	this->numIterations = 0;
}

template <typename Dtype>
AccuracyLayer<Dtype>::~AccuracyLayer() {

}

template <typename Dtype>
void AccuracyLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	// label shape는 변하지 않음.
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	// XXX: SO!!!! TEMPORAL
	//this->_inputData[0]->reshape({10, 1, 1000, 1});

	this->_inputShape[0] = this->_inputData[0]->getShape();
	this->_inputShape[1] = this->_inputData[1]->getShape();

	/*
	this->_printOn();
	this->_inputData[0]->print_shape();
	this->_inputData[1]->print_shape();
	//exit(1);
	this->_printOff();
	*/

	const uint32_t topK = SLPROP(Accuracy, topK);
	const int labelAxis = SLPROP(Accuracy, axis);
	SASSERT(topK <= this->_inputData[0]->getCount() / this->_inputData[1]->getCount(),
			"topK must be less than or equal to the number of classes.");
	this->outerNum = this->_inputData[0]->getCountByAxis(0, labelAxis);
	this->innerNum = this->_inputData[0]->getCountByAxis(labelAxis + 1);

	SASSERT(this->outerNum * this->innerNum == this->_inputData[1]->getCount(),
			"Number of labels must match number of predictions: outerNum->%d, innerNum->%d, inputData1_count->%d",
          this->outerNum, this->innerNum, this->_inputData[1]->getCount());

	vector<uint32_t> outputShape({1, 1, 1, 1});
	this->_outputData[0]->reshape(outputShape);
}

template <typename Dtype>
void AccuracyLayer<Dtype>::feedforward() {
	reshape();

	Dtype accuracy = 0;
	const Dtype* inputData = this->_inputData[0]->host_data();
	const Dtype* inputLabel = this->_inputData[1]->host_data();
	const int dim = this->_inputData[0]->getCount() / this->outerNum;
	const int numLabels = this->_inputData[0]->getShape(SLPROP(Accuracy, axis));
	vector<Dtype> maxVal(SLPROP(Accuracy, topK) + 1);
	vector<int> maxId(SLPROP(Accuracy, topK) + 1);

	int count = 0;
	for (int i = 0; i < this->outerNum; i++) {
		for (int j = 0; j < this->innerNum; j++) {
			const int labelValue = static_cast<int>(inputLabel[i * this->innerNum + j]);
			if (this->hasIgnoreLabel && labelValue == SLPROP(Accuracy, ignoreLabel)) {
				continue;
			}
			SASSERT0(labelValue >= 0);
			SASSERT0(labelValue < numLabels);
			// Tok-k accuracy
			vector<pair<Dtype, int>> inputDataVector;
			for (int k = 0; k < numLabels; k++) {
				inputDataVector.push_back(make_pair(
						inputData[i * dim + k * this->innerNum + j], k));
			}
			std::partial_sort(
					inputDataVector.begin(), inputDataVector.begin() + SLPROP(Accuracy, topK),
					inputDataVector.end(), std::greater<pair<Dtype, int>>());
			// check if true label is in top k predictions
			for (int k = 0; k < SLPROP(Accuracy, topK); k++) {
				if (inputDataVector[k].second == labelValue) {
					accuracy++;
					break;
				}
			}
			count++;
		}
	}

	this->_outputData[0]->mutable_host_data()[0] = accuracy / count;

	/*
	this->_printOn();
	this->_outputData[0]->print_data({}, false);
	this->_printOff();
	*/

	// Accuracy layer should not be used as a loss function.


	//cout << "accuracy: " << accuracy << endl;
    this->numCorrect += accuracy;
    this->numIterations++;

    /*
    this->_printOn();
    this->_outputData[0]->print_data({}, false);
    this->_printOff();
    */
    if (this->numIterations >= SNPROP(testInterval)) {
    	float acc = numCorrect / ((float)numIterations*SNPROP(batchSize));
    	STDOUT_LOG("average accuracy[%s] : %f", SLPROP_BASE(name).c_str(), acc);
        this->numIterations = 0;
        this->numCorrect = 0;
    }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::backpropagation() {
	//SASSERT(false, "Not implemented yet.");
}


template <typename Dtype>
Dtype AccuracyLayer<Dtype>::getAccuracy() {
	Dtype accuracy = this->_outputData[0]->host_data()[0];
	return accuracy;
}

template<typename Dtype>
Dtype AccuracyLayer<Dtype>::measure() {
    return getAccuracy();
}

template<typename Dtype>
Dtype AccuracyLayer<Dtype>::measureAll() {
    float acc = numCorrect / ((float)numIterations*SNPROP(batchSize));
    return (Dtype)acc;
}


/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* AccuracyLayer<Dtype>::initLayer() {
	AccuracyLayer* layer = NULL;
	SNEW(layer, AccuracyLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void AccuracyLayer<Dtype>::destroyLayer(void* instancePtr) {
    AccuracyLayer<Dtype>* layer = (AccuracyLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void AccuracyLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	if (isInput) {
		SASSERT0(index < 2);
	} else {
		SASSERT0(index < 1);
	}

    AccuracyLayer<Dtype>* layer = (AccuracyLayer<Dtype>*)instancePtr;
    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool AccuracyLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    AccuracyLayer<Dtype>* layer = (AccuracyLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void AccuracyLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	AccuracyLayer<Dtype>* layer = (AccuracyLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void AccuracyLayer<Dtype>::backwardTensor(void* instancePtr) {
	AccuracyLayer<Dtype>* layer = (AccuracyLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void AccuracyLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool AccuracyLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    if (inputShape.size() != 2) {
        return false;
    }

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 1;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t AccuracyLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class AccuracyLayer<float>;
