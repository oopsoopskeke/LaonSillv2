/*
 * DataInputLayer.cpp
 *
 *  Created on: Jun 30, 2017
 *      Author: jkim
 */

#include <unistd.h>

#include <vector>

#include "DataInputLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "IO.h"
#include "WorkContext.h"
#include "Param.h"
#include "Perf.h"
#include "MemoryMgmt.h"
#include "SDF.h"
#include "NetworkRecorder.h"

using namespace std;

template <typename Dtype>
DataInputLayer<Dtype>::DataInputLayer()
: InputLayer<Dtype>(),
  dataReader(SLPROP(Input, source)),
  dataTransformer(&SLPROP(DataInput, dataTransformParam)) {
	this->type = Layer<Dtype>::DataInput;
	const string dataSetName = SLPROP(DataInput, dataSetName);
	if (dataSetName.empty()) {
		this->dataReader.selectDataSetByIndex(0);
	} else {
		this->dataReader.selectDataSetByName(dataSetName);
	}

	DataTransformParam& dataTransformParam = this->dataTransformer.param;
	dataTransformParam.resizeParam = SLPROP(DataInput, resizeParam);
	dataTransformParam.resizeParam.updateInterpMode();
}

template <typename Dtype>
DataInputLayer<Dtype>::~DataInputLayer() {
	// TODO Auto-generated destructor stub
}

template <typename Dtype>
void DataInputLayer<Dtype>::reshape() {
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
	    if (!Layer<Dtype>::_isInputShapeChanged(i)) {
			continue;
        }

		// data
		if (i == 0) {
			vector<uint32_t> dataShape = this->dataTransformer.inferDataShape(datum);
			dataShape[0] = SNPROP(batchSize);
			//vector<uint32_t> dataShape = {SNPROP(batchSize), uint32_t(datum->channels),
			//		uint32_t(datum->height), uint32_t(datum->width)};
			this->_inputData[0]->reshape(dataShape);
			this->_inputShape[0] = dataShape;
		}
		// label
		else if (i == 1) {
			vector<uint32_t> labelShape = {SNPROP(batchSize), 1, 1, 1};
			this->_inputData[1]->reshape(labelShape);
			this->_inputShape[1] = labelShape;
		}
	}

    SDELETE(datum);
}



template <typename Dtype>
void DataInputLayer<Dtype>::feedforward() {
	reshape();
	load_batch();
}

template <typename Dtype>
void DataInputLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
	load_batch();
}

template <typename Dtype>
void DataInputLayer<Dtype>::load_batch() {
	/*
	vector<float> mean = SLPROP(Input, mean);
	bool hasMean = false;
	if (mean.size() > 0) {
		hasMean = true;
	}
	*/

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

        this->dataTransformer.transform(datum, this->_outputData[0], item_id);

        /*
        //datum->print();
		const string& data = datum->data;
		const int datum_channels = datum->channels;
		const int datum_height = datum->height;
		const int datum_width = datum->width;

		int height = datum_height;
		int width = datum_width;

		int h_off = 0;
		int w_off = 0;

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
		*/

		// if label tensor specified ...
		if (this->_outputData.size() > 1) {
			Dtype* output_label = this->_inputData[1]->mutable_host_data();
			output_label[item_id] = datum->label;
		}


		/*
		cout << "label: " << datum->label << endl;
		//this->_printOn();
		//this->_outputData[0]->print_data({}, false);
		//this->_printOff();
		Dtype* temp = (Dtype*)malloc(sizeof(Dtype) * datum->getImgSize());
		ConvertCHWToHWC(datum->channels, datum->height, datum->width, output_data, temp);
		//PrintImageData(datum->channels, datum->height, datum->width, temp, true);

		cv::Mat cv_img(datum->height, datum->width, CV_32FC3, temp);
		cv_img.convertTo(cv_img, CV_8UC3);
		cv::imshow("result", cv_img);
		cv::waitKey(0);
		cv::destroyWindow("result");
		free(temp);
		//exit(1);
		 */


		/*
		const int singleImageSize = this->_outputData[0]->getCountByAxis(1);
		const int imageHeight = this->_outputData[0]->getShape(2);		// network image size
		const int imageWidth = this->_outputData[0]->getShape(3);
		const int height = datum->height;				// final image size
		const int width = datum->width;

		const vector<Dtype> pixelMeans = {104.0, 117.0, 123.0};
		//const vector<Dtype> pixelMeans = {0, 0, 0};
		const Dtype* dataData = this->_inputData[0]->host_data();
		cv::Mat result;
		Data<Dtype> temp("temp");
		transformInv(1, singleImageSize, imageHeight, imageWidth, height, width,
				pixelMeans, dataData, temp, result);
				*/



		SDELETE(datum);

	}
}


template <typename Dtype>
int DataInputLayer<Dtype>::getNumTrainData() {
	return this->dataReader.getNumData();
}

template <typename Dtype>
int DataInputLayer<Dtype>::getNumTestData() {
	return 0;
}

template <typename Dtype>
void DataInputLayer<Dtype>::shuffleTrainDataSet() {

}








/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* DataInputLayer<Dtype>::initLayer() {
	DataInputLayer* layer = NULL;
	SNEW(layer, DataInputLayer<Dtype>);
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
void DataInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    DataInputLayer<Dtype>* layer = (DataInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void DataInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	if (isInput) {
		SASSERT0(false);
	} else {
		SASSERT0(index < 2);
	}

    DataInputLayer<Dtype>* layer = (DataInputLayer<Dtype>*)instancePtr;
    if (!isInput) {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DataInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DataInputLayer<Dtype>* layer = (DataInputLayer<Dtype>*)instancePtr;
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
void DataInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	DataInputLayer<Dtype>* layer = (DataInputLayer<Dtype>*)instancePtr;
	layer->feedforward();

}

template<typename Dtype>
void DataInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void DataInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool DataInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    // XXX: 일단 Lenet을 테스트하는 선에서만 동작할 수 있도록 제한적으로 구현하였다.
    //      추후에 보강이 필요하다.
    SDFHeader header = SDF::retrieveSDFHeader(SLPROP(DataInput, source));

    SASSERT0(header.uniform);
    SASSERT0(header.minHeight == header.maxHeight);
    SASSERT0(header.minWidth == header.maxWidth);

    if (SLPROP_BASE(output).size() < 1 || SLPROP_BASE(output).size() > 2) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "Data Input Layer should have 1 or 2 output tensors but it has %d tensors",
            (int)SLPROP_BASE(output).size());
        return false;
    }

    TensorShape outputShape1;
    outputShape1.N = SNPROP(batchSize);
    outputShape1.C = header.channels;
    outputShape1.H = header.minHeight;
    outputShape1.W = header.minWidth;
    outputShape.push_back(outputShape1);

    if (SLPROP_BASE(output).size() == 2) {
        TensorShape outputShape2;
        outputShape2.N = SNPROP(batchSize);
        outputShape2.C = 1;
        outputShape2.H = 1;
        outputShape2.W = 1;
        outputShape.push_back(outputShape2);
    }

    return true;
}

template<typename Dtype>
uint64_t DataInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class DataInputLayer<float>;
