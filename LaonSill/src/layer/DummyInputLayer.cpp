/*
 * DummyInputLayer.cpp
 *
 *  Created on: Jan 21, 2017
 *      Author: jkim
 */

#include "DummyInputLayer.h"
#include "PropMgmt.h"
#include "Data.h"
#include "MemoryMgmt.h"

using namespace std;

template<typename Dtype>
DummyInputLayer<Dtype>::DummyInputLayer()
: InputLayer<Dtype>() {
	this->type = Layer<Dtype>::DummyInput;
}

template <typename Dtype>
DummyInputLayer<Dtype>::~DummyInputLayer() {

}


template <typename Dtype>
void DummyInputLayer<Dtype>::feedforward() {
	reshape();
}

template <typename Dtype>
void DummyInputLayer<Dtype>::reshape() {
	const vector<string>& outputs = SLPROP(Input, output);
	vector<string>& inputs = SLPROP(Input, input);
	if (inputs.size() < 1) {
		for (uint32_t i = 0; i < outputs.size(); i++) {
			inputs.push_back(outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	bool adjusted = Layer<Dtype>::_adjustInputShape();

	if (adjusted) {

		const uint32_t batches = SNPROP(batchSize);
		const vector<uint32_t>& shapes = SLPROP(DummyInput, shapes);
		const int inputDataSize = this->_inputData.size();
		SASSERT0(inputDataSize * Data<Dtype>::SHAPE_SIZE == shapes.size());

		for (int i = 0; i < inputDataSize; i++) {
			vector<uint32_t> shape;
			for (int j = 0; j < Data<Dtype>::SHAPE_SIZE; j++) {
				if (shapes[i * Data<Dtype>::SHAPE_SIZE + j] == 0) {
					SASSERT0(j == 0);
					shape.push_back(batches);
				} else {
					shape.push_back(shapes[i*Data<Dtype>::SHAPE_SIZE + j]);
				}
			}
			this->_inputData[i]->reshape(shape);
			this->_inputShape[i] = shape;
		}

		/*
		if (this->_inputData.size() == 2) {
			const uint32_t batches = SNPROP(batchSize);
			const uint32_t channels = SLPROP(DummyInput, channels);
			const uint32_t rows = SLPROP(DummyInput, rows);
			const uint32_t cols = SLPROP(DummyInput, cols);
			const uint32_t numClasses = SLPROP(DummyInput, numClasses);

			const vector<uint32_t> dataShape = {batches, channels, rows, cols};
			this->_inputData[0]->reshape(dataShape);
			this->_inputShape[0] = dataShape;

			const vector<uint32_t> labelShape = {batches, 1, 1, 1};
			this->_inputData[1]->reshape(labelShape);
			this->_inputShape[1] = labelShape;
		} else {
			const vector<uint32_t> shape = {1, 1, 1, 1};
			for (int i = 0; i < this->_inputData.size(); i++) {
				this->_inputData[i]->reshape(shape);
				this->_inputShape[i] = shape;
			}
		}
		*/
	}

}


template<typename Dtype>
int DummyInputLayer<Dtype>::getNumTrainData() {
    return SLPROP(DummyInput, numTrainData);
}

template<typename Dtype>
int DummyInputLayer<Dtype>::getNumTestData() {
    return SLPROP(DummyInput, numTestData);
}

template<typename Dtype>
void DummyInputLayer<Dtype>::shuffleTrainDataSet() {
}

template class DummyInputLayer<float>;




/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* DummyInputLayer<Dtype>::initLayer() {
	DummyInputLayer* layer = NULL;
	SNEW(layer, DummyInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void DummyInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    DummyInputLayer<Dtype>* layer = (DummyInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void DummyInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(false);
	}

    DummyInputLayer<Dtype>* layer = (DummyInputLayer<Dtype>*)instancePtr;

	SASSERT0(layer->_outputData.size() == index);
	layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool DummyInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DummyInputLayer<Dtype>* layer = (DummyInputLayer<Dtype>*)instancePtr;
    layer->reshape();

    /*
    if (SNPROP(miniBatch) == 0) {
		int trainDataNum = layer->getNumTrainData();
		if (trainDataNum % SNPROP(batchSize) == 0) {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize);
		} else {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize) + 1;
		}
		WorkContext::curPlanInfo->miniBatchCount = SNPROP(miniBatch);
	}
	*/

    return true;
}

template<typename Dtype>
void DummyInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	DummyInputLayer<Dtype>* layer = (DummyInputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void DummyInputLayer<Dtype>::backwardTensor(void* instancePtr) {
}

template<typename Dtype>
void DummyInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool DummyInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    const uint32_t batches = SNPROP(batchSize);
    const vector<uint32_t>& shapes = SLPROP(DummyInput, shapes);

    if (shapes.size() % 4 != 0) {
        return false;
    }

    for (int i = 0; i < shapes.size(); i += Data<Dtype>::SHAPE_SIZE) {
        TensorShape _outputShape;
        for (int j = 0; j < Data<Dtype>::SHAPE_SIZE; j++) {
            if (j == 0) {
                _outputShape.N = batches;
            } else {
                tensorRefByIndex(_outputShape, j) = shapes[i * Data<Dtype>::SHAPE_SIZE + j];
            }
        }
        outputShape.push_back(_outputShape);
    }

    return true;
}

template<typename Dtype>
uint64_t DummyInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template void* DummyInputLayer<float>::initLayer();
template void DummyInputLayer<float>::destroyLayer(void* instancePtr);
template void DummyInputLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool DummyInputLayer<float>::allocLayerTensors(void* instancePtr);
template void DummyInputLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void DummyInputLayer<float>::backwardTensor(void* instancePtr);
template void DummyInputLayer<float>::learnTensor(void* instancePtr);
template bool DummyInputLayer<float>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape);
template uint64_t DummyInputLayer<float>::calcGPUSize(vector<TensorShape> inputShape);
