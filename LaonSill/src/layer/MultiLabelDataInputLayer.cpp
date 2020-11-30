/*
 * MultiLabelDataInputLayer.cpp
 *
 *  Created on: Jul 12, 2017
 *      Author: jkim
 */

#include "MultiLabelDataInputLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "IO.h"
#include "WorkContext.h"
#include "Param.h"
#include "Perf.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
MultiLabelDataInputLayer<Dtype>::MultiLabelDataInputLayer()
: InputLayer<Dtype>(),
  dataReader(SLPROP(Input, source)) {
	this->type = Layer<Dtype>::MultiLabelDataInput;
	const string dataSetName = SLPROP(MultiLabelDataInput, dataSetName);
	if (dataSetName.empty()) {
		this->dataReader.selectDataSetByIndex(0);
	} else {
		this->dataReader.selectDataSetByName(dataSetName);
	}
}

template <typename Dtype>
MultiLabelDataInputLayer<Dtype>::~MultiLabelDataInputLayer() {

}


template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	Layer<Dtype>::_adjustInputShape();

	class Datum* datum;
	struct timespec startTime;

	if (SPARAM(INPUT_DATA_PROVIDER_MEASURE_PERFORMANCE)) {
		SPERF_START(DATAINPUT_ACCESS_TIME, &startTime);
	}

	if ((WorkContext::curBootMode == BootMode::ServerClientMode) &&
		SPARAM(USE_INPUT_DATA_PROVIDER)) {
		void* elem = NULL;
		while (true) {
			elem = InputDataProvider::getData(this->inputPool, true);

			if (elem == NULL) {
				usleep(SPARAM(INPUT_DATA_PROVIDER_CALLER_RETRY_TIME_USEC));
			} else {
				break;
			}
		}
		datum = (class Datum*)elem;
	} else {
		datum = this->dataReader.peekNextData();
	}
	if (SPARAM(INPUT_DATA_PROVIDER_MEASURE_PERFORMANCE)) {
		SPERF_END(DATAINPUT_ACCESS_TIME, startTime);
	}

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		//if (!Layer<Dtype>::_isInputShapeChanged(i))
		//	continue;

		// data
		if (i == 0) {
			vector<uint32_t> dataShape =
			{SNPROP(batchSize), uint32_t(datum->channels), uint32_t(datum->height),
					uint32_t(datum->width)};
			this->_inputData[0]->reshape(dataShape);
			this->_inputShape[0] = dataShape;
		}
		// label
		else if (i == 1) {
			vector<uint32_t> labelShape = {SNPROP(batchSize),
					(uint32_t)SLPROP(MultiLabelDataInput, labelCount), 1, 1};
			this->_inputData[1]->reshape(labelShape);
			this->_inputShape[1] = labelShape;
		}
	}
}



template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::feedforward() {
	reshape();
	load_batch();
}

template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
	load_batch();
}

template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::load_batch() {
	vector<float> mean = SLPROP(Input, mean);
	bool hasMean = false;
	if (mean.size() > 0) {
		hasMean = true;
	}
	
    this->_inputData[1]->reset_host_data();
	for (int item_id = 0; item_id < SNPROP(batchSize); item_id++) {
		int offset = this->_inputData[0]->offset(item_id);
		Dtype* output_data = this->_inputData[0]->mutable_host_data();
		output_data += offset;

		class Datum* datum;
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
			datum = (class Datum*)elem;
		} else {
			datum = this->dataReader.getNextData();
		}
		if (SPARAM(INPUT_DATA_PROVIDER_MEASURE_PERFORMANCE)) {
			SPERF_END(DATAINPUT_ACCESS_TIME, startTime);
		}

		const string& data = datum->data;
		const int datum_channels = datum->channels;
		const int datum_height = datum->height;
		const int datum_width = datum->width;

		int height = datum_height;
		int width = datum_width;

		int h_off = 0;
		int w_off = 0;

		// DATA TRANSFORM ////////////////////////////////////////////////////////////////////
		// expand mean to follow real number of image channels
		if (hasMean) {
			SASSERT(mean.size() == 1 || mean.size() == datum_channels,
					"Specify either 1 mean value or as many as channels: %d", datum_channels);
			if (datum_channels > 1 && mean.size() == 1) {
				// Replicate the mean for simplicity
				for (int c = 1; c < datum_channels; c++) {
					mean.push_back(mean[0]);
				}
			}
		}

		// apply mean and scale
		const float scale = SLPROP(Input, scale);
		Dtype datum_element;
		int top_index, data_index;
		for (int c = 0; c < datum_channels; c++) {
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
					top_index = (c * height + h) * width + w;
					datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));

					if (hasMean) {
						output_data[top_index] = (datum_element - mean[c]) * scale;
					} else {
						output_data[top_index] = datum_element * scale;
					}
				}
			}
		}
		//////////////////////////////////////////////////////////////////////////////////////

		Dtype* output_label = this->_inputData[1]->mutable_host_data();
		const int labelCount = SLPROP(MultiLabelDataInput, labelCount);
		int label = datum->label;
		if (label < labelCount) {
			output_label[item_id * labelCount + label] = Dtype(1);
		}
		for (int labelIdx = 0; labelIdx < datum->float_data.size(); labelIdx++) {
			label = (int)datum->float_data[labelIdx];
			if (label < labelCount) {
				output_label[item_id * labelCount + label] = Dtype(1);
			}
		}

		SDELETE(datum);
	}
}


template <typename Dtype>
int MultiLabelDataInputLayer<Dtype>::getNumTrainData() {
	return this->dataReader.getNumData();
}

template <typename Dtype>
int MultiLabelDataInputLayer<Dtype>::getNumTestData() {
	return 0;
}

template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::shuffleTrainDataSet() {

}








/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* MultiLabelDataInputLayer<Dtype>::initLayer() {
	MultiLabelDataInputLayer* layer = NULL;
	SNEW(layer, MultiLabelDataInputLayer<Dtype>);
	SASSUME0(layer != NULL);

    if ((WorkContext::curBootMode == BootMode::ServerClientMode) &&
		SPARAM(USE_INPUT_DATA_PROVIDER)) {
		InputDataProvider::addPool(WorkContext::curNetworkID, WorkContext::curDOPID,
			SLPROP_BASE(name), DRType::DatumType, (void*)&layer->dataReader);
		layer->inputPool = InputDataProvider::getInputPool(WorkContext::curNetworkID,
														  WorkContext::curDOPID,
														  SLPROP_BASE(name));
	}
    return (void*)layer;
}

template<typename Dtype>
void MultiLabelDataInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    MultiLabelDataInputLayer<Dtype>* layer = (MultiLabelDataInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void MultiLabelDataInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	if (isInput) {
		SASSERT0(false);
	} else {
		SASSERT0(index < 2);
	}

    MultiLabelDataInputLayer<Dtype>* layer = (MultiLabelDataInputLayer<Dtype>*)instancePtr;
    if (!isInput) {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool MultiLabelDataInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    MultiLabelDataInputLayer<Dtype>* layer = (MultiLabelDataInputLayer<Dtype>*)instancePtr;
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
void MultiLabelDataInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	MultiLabelDataInputLayer<Dtype>* layer = (MultiLabelDataInputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void MultiLabelDataInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void MultiLabelDataInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool MultiLabelDataInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    SDFHeader header = SDF::retrieveSDFHeader(SLPROP(Input, source));
    const int batches = SNPROP(batchSize);
    const int channels = header.channels;
    const int height = header.maxHeight;
    const int width = header.maxWidth;

    // data
    TensorShape outputShape1;
    outputShape1.N = batches;
    outputShape1.C = channels;
    outputShape1.H = height;
    outputShape1.W = width;
    outputShape.push_back(outputShape1);

    // label
    TensorShape outputShape2;
    outputShape2.N = batches;
    outputShape2.C = SLPROP(MultiLabelDataInput, labelCount);
    outputShape2.H = 1;
    outputShape2.W = 1;
    outputShape.push_back(outputShape2);

    return true;
}

template<typename Dtype>
uint64_t MultiLabelDataInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class MultiLabelDataInputLayer<float>;
