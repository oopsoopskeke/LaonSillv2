/*
 * AnnotatedDataLayer.cpp
 *
 *  Created on: Sep 4, 2017
 *      Author: jkim
 */

#include "AnnotatedDataLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "IO.h"
#include "WorkContext.h"
#include "Param.h"
#include "Perf.h"
#include "EnumDef.h"
#include "MathFunctions.h"
#include "Sampler.h"
#include "ImageUtil.h"
#include "MemoryMgmt.h"

using namespace std;




template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer()
: AnnotatedDataLayer(NULL) {}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(_AnnotatedDataPropLayer* prop)
: InputLayer<Dtype>(),
  dataReader(GET_PROP(prop, AnnotatedData, source)),
  dataTransformer(&GET_PROP(prop, AnnotatedData, dataTransformParam)) {
	this->type = Layer<Dtype>::AnnotatedData;
	const string dataSetName = GET_PROP(prop, AnnotatedData, dataSetName);
	if (dataSetName.empty()) {
		this->dataReader.selectDataSetByIndex(0);
	} else {
		this->dataReader.selectDataSetByName(dataSetName);
	}


	if (prop) {
		this->prop = NULL;
		SNEW(this->prop, _AnnotatedDataPropLayer);
		SASSUME0(this->prop != NULL);
		*(this->prop) = *(prop);
	} else {
		this->prop = NULL;
	}

	DataTransformParam& dataTransformParam = this->dataTransformer.param;
	dataTransformParam.resizeParam = GET_PROP(prop, AnnotatedData, resizeParam);
	dataTransformParam.resizeParam.updateInterpMode();
	dataTransformParam.distortParam = GET_PROP(prop, AnnotatedData, distortParam);
	dataTransformParam.expandParam = GET_PROP(prop, AnnotatedData, expandParam);
	dataTransformParam.noiseParam = GET_PROP(prop, AnnotatedData, noiseParam);
	dataTransformParam.emitConstraint = GET_PROP(prop, AnnotatedData, emitConstraint);
	//this->dataTransformer.param.print();

	this->outputLabels = !(GET_PROP(prop, AnnotatedData, output).size() == 1);
	this->hasAnnoType = !(GET_PROP(prop, AnnotatedData, annoType) == AnnotationType::ANNO_NONE);

	if (GET_PROP(prop, AnnotatedData, batchSampler0).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler0));
	if (GET_PROP(prop, AnnotatedData, batchSampler1).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler1));
	if (GET_PROP(prop, AnnotatedData, batchSampler2).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler2));
	if (GET_PROP(prop, AnnotatedData, batchSampler3).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler3));
	if (GET_PROP(prop, AnnotatedData, batchSampler4).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler4));
	if (GET_PROP(prop, AnnotatedData, batchSampler5).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler5));
	if (GET_PROP(prop, AnnotatedData, batchSampler6).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler6));
	if (GET_PROP(prop, AnnotatedData, batchSampler7).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler7));
	if (GET_PROP(prop, AnnotatedData, batchSampler8).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler8));
	if (GET_PROP(prop, AnnotatedData, batchSampler9).hasMaxSample())
		this->batchSamplers.push_back(GET_PROP(prop, AnnotatedData, batchSampler9));

	//for (int i = 0; i < this->batchSamplers.size(); i++) {
	//	this->batchSamplers[i].print();
	//}

	this->labelMapFile = GET_PROP(prop, AnnotatedData, labelMapFile);
	// Make sure dimension is consistent within batch.
	if (this->dataTransformer.param.resizeParam.prob >= 0.f) {
		if (this->dataTransformer.param.resizeParam.resizeMode == ResizeMode::FIT_SMALL_SIZE) {
			SASSERT(SNPROP(batchSize) == 1, "Only support batch size of 1 for FIT_SMALL_SIZE.");
		}
	}
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < GET_PROP(prop, AnnotatedData, output).size(); i++) {
			GET_PROP(prop, AnnotatedData, input).push_back(
                GET_PROP(prop, AnnotatedData, output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	Layer<Dtype>::_adjustInputShape();

	// Read a data point, and use it to initialize the output data.
	class AnnotatedDatum* annoDatum;
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
        annoDatum = (class AnnotatedDatum*)elem;
    } else {
        annoDatum = this->dataReader.peekNextData();
    }
    if (SPARAM(INPUT_DATA_PROVIDER_MEASURE_PERFORMANCE)) {
        SPERF_END(DATAINPUT_ACCESS_TIME, startTime);
    }

    const int batchSize = SNPROP(batchSize);
    // Use data transformer to infer the expected data shape from annoDatum.
    vector<uint32_t> outputShape = this->dataTransformer.inferDataShape(annoDatum);
    outputShape[0] = batchSize;
    this->_outputData[0]->reshape(outputShape);
    //this->_outputData[0]->print_shape();

    // label
    if (this->outputLabels) {
    	vector<uint32_t> labelShape(4, 1);
    	if (this->hasAnnoType) {
    		AnnotationType annoType = GET_PROP(prop, AnnotatedData, annoType);
    		int numBBoxes = 0;
    		if (annoType == AnnotationType::BBOX) {
    			for (int g = 0; g < annoDatum->annotation_groups.size(); g++) {
    				numBBoxes += annoDatum->annotation_groups[g].annotations.size();
    			}
    			labelShape[0] = 1;
    			labelShape[1] = 1;
    			labelShape[2] = std::max(numBBoxes, 1);
    			labelShape[3] = 8;
    		} else {
    			SASSERT(false, "Unknown annotation type.");
    		}
    	} else {
    		labelShape[0] = batchSize;
    	}
    	this->_outputData[1]->reshape(labelShape);
    }

    SDELETE(annoDatum);
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::feedforward() {
	reshape();
	load_batch();
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
	load_batch();
}



template <typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch() {
	Dtype* outputData = this->_outputData[0]->mutable_host_data();
	Dtype* outputLabel = NULL;
	if (this->outputLabels && !this->hasAnnoType) {
		outputLabel = this->_outputData[1]->mutable_host_data();
	}

	// Store transformed annotation.
	map<int, vector<AnnotationGroup>> allAnno;
	int numBBoxes = 0;

	const int batchSize = SNPROP(batchSize);
	DataTransformParam& transformParam = this->dataTransformer.param;

	for (int itemId = 0; itemId < batchSize; itemId++) {
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

		//annoDatum->print();
#if 0
		ImageUtil<Dtype>::dispDatum(annoDatum, "annDatum");
#endif

		// distort & expand
		AnnotatedDatum distortDatum;
		AnnotatedDatum* expandDatum = NULL;
		if (transformParam.hasDistortParam()) {
			//cout << "hasDistortParam()" << endl;
			distortDatum = *annoDatum;
			this->dataTransformer.distortImage(annoDatum, &distortDatum);

#if 0
			if (itemId == 0) {
				ImageUtil<Dtype>::dispDatum(&distortDatum, "distortDatum");
			}
#endif

			if (transformParam.hasExpandParam()) {
				//cout << "hasExpandParam()" << endl;
				expandDatum = NULL;
				SNEW(expandDatum, AnnotatedDatum);
				SASSUME0(expandDatum != NULL);
				this->dataTransformer.expandImage(&distortDatum, expandDatum);

#if 0
				if (itemId == 0) {
					ImageUtil<Dtype>::dispDatum(expandDatum, "expandDatum");
				}
#endif


			} else {
				//cout << "not hasExpandParam()" << endl;
				expandDatum = &distortDatum;
			}
		} else {
			//cout << "not hasDistortParam()" << endl;
			if (transformParam.hasExpandParam()) {
				//cout << "hasExpandParam()" << endl;
				expandDatum = NULL;
				SNEW(expandDatum, AnnotatedDatum);
				SASSUME0(expandDatum != NULL);
				this->dataTransformer.expandImage(annoDatum, expandDatum);
			} else {
				//cout << "not hasExpandParam()" << endl;
				expandDatum = annoDatum;
			}
		}


		//if (itemId == 4) {
		//	cout << "break ... " << endl;
		//}

		// cropped by batch sampler
		AnnotatedDatum* sampledDatum = NULL;
		bool hasSampled = false;
		if (this->batchSamplers.size() > 0) {
			// Generate sampled bboxes from expandDatum.
			vector<NormalizedBBox> sampledBBoxes;
			GenerateBatchSamples(*expandDatum, this->batchSamplers, &sampledBBoxes);
			if (sampledBBoxes.size() > 0) {
				// Randomly pick a sampled bbox and crop the expandDatum.
				int randIdx = soooa_rng_rand() % sampledBBoxes.size();
				sampledDatum = NULL;
				SNEW(sampledDatum, AnnotatedDatum);
				SASSUME0(sampledDatum != NULL);
				this->dataTransformer.cropImage(expandDatum, sampledBBoxes[randIdx],
						sampledDatum);

#if 0
				if (itemId == 0) {
					ImageUtil<Dtype>::dispDatum(sampledDatum, "sampledDatum");
				}
#endif

				hasSampled = true;
			} else {
				sampledDatum = expandDatum;
			}
		} else {
			sampledDatum = expandDatum;
		}
		SASSERT0(sampledDatum != NULL);

		// Apply data transformations (mirror, scale, crop ... )
		//int offset = this->_outputData[0]->offset(itemId);
		//Dtype* dataPtr = this->_outputData[0]->mutable_host_data() + offset;
		vector<AnnotationGroup> transformedAnnoVec;
		if (this->outputLabels) {
			if (this->hasAnnoType) {
				// Make sure all data have same annotation type.
				SASSERT(sampledDatum->type != AnnotationType::ANNO_NONE,
						"Some datum misses AnnotationType.");
				SASSERT(GET_PROP(prop, AnnotatedData, annoType) == sampledDatum->type,
						"Different AnnotationType.");

				// Transform datum and annotation_group at the same time.
				transformedAnnoVec.clear();
				this->dataTransformer.transform(sampledDatum, this->_outputData[0], itemId,
						transformedAnnoVec);
				if (GET_PROP(prop, AnnotatedData, annoType) == AnnotationType::BBOX) {
					// Count the number of bboxes.
					for (int g = 0; g < transformedAnnoVec.size(); g++) {
						numBBoxes += transformedAnnoVec[g].annotations.size();
					}
				} else {
					SASSERT(false, "Unknown annotation type.");
				}
				allAnno[itemId] = transformedAnnoVec;
			} else {
				this->dataTransformer.transform(sampledDatum, this->_outputData[0], itemId);
				// Otherwise, store the label from datum.
				SASSERT(sampledDatum->hasLabel(), "Cannot find any label.");
				outputLabel[itemId] = sampledDatum->label;
			}
		} else {
			this->dataTransformer.transform(sampledDatum, this->_outputData[0], itemId);
		}
		// clear memory.
		if (hasSampled) {
			SDELETE(sampledDatum);
		}
		if (transformParam.hasExpandParam()) {
			SDELETE(expandDatum);
		}
        SDELETE(annoDatum);
		// reader_.free().push(...) 		???
	}

	// Store "rich" annotation if needed.
	if (this->outputLabels && hasAnnoType) {
		vector<uint32_t> labelShape(4);
		if (GET_PROP(prop, AnnotatedData, annoType) == AnnotationType::BBOX) {
			labelShape[0] = 1;
			labelShape[1] = 1;
			labelShape[3] = 8;
			//cout << "numBBoxes: " << numBBoxes << endl;
			if (numBBoxes == 0) {
				// Store all -1 in the label.
				labelShape[2] = 1;
				this->_outputData[1]->reshape(labelShape);
				soooa_set<Dtype>(8, -1, this->_outputData[1]->mutable_host_data());
			} else {
				// Reshape the label and store the annotation.
				labelShape[2] = numBBoxes;
				this->_outputData[1]->reshape(labelShape);
				outputLabel = this->_outputData[1]->mutable_host_data();
				int idx = 0;
				for (int itemId = 0; itemId < batchSize; itemId++) {
					const vector<AnnotationGroup>& annoVec = allAnno[itemId];
					for (int g = 0; g < annoVec.size(); g++) {
						const AnnotationGroup& annoGroup = annoVec[g];
						for (int a = 0; a < annoGroup.annotations.size(); a++) {
							const Annotation_s& anno = annoGroup.annotations[a];
							const NormalizedBBox& bbox = anno.bbox;
							outputLabel[idx++] = itemId;
							outputLabel[idx++] = annoGroup.group_label;
							outputLabel[idx++] = anno.instance_id;
							outputLabel[idx++] = bbox.xmin;
							outputLabel[idx++] = bbox.ymin;
							outputLabel[idx++] = bbox.xmax;
							outputLabel[idx++] = bbox.ymax;
							outputLabel[idx++] = bbox.difficult;
						}
					}
				}
			}
		} else {
			SASSERT(false, "Unknown annotation type.");
		}
	}



#if 0
	int xmin, ymin, xmax, ymax;
	char labelBuf[256];
	//int i = 0, j = 0;

	cout << "-----------------------------------------------------" << endl;
	this->_printOn();
	this->_outputData[1]->print_data({}, false, -1);
	this->_printOff();
	for (int i = 0; i < this->_outputData[0]->batches(); i++) {
		cv::Mat cv_img = ConvertCHWDataToHWCCV(this->_outputData[0], i);
		cout << "rows: " << cv_img.rows << ", cols: " << cv_img.cols << endl;
		for (int j = 0; j < this->_outputData[1]->height(); j++) {
			if ((int)(outputLabel[j * 8 + 0]) == i) {
				xmin = (int)(outputLabel[j * 8 + 3] * this->_outputData[0]->width());
				ymin = (int)(outputLabel[j * 8 + 4] * this->_outputData[0]->height());
				xmax = (int)(outputLabel[j * 8 + 5] * this->_outputData[0]->width());
				ymax = (int)(outputLabel[j * 8 + 6] * this->_outputData[0]->height());

				cv::rectangle(cv_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
				sprintf(labelBuf, "%d", (int)outputLabel[j * 8 + 1]);
				cv::putText(cv_img, string(labelBuf), cv::Point(xmin, ymin + 15.0f), 2, 0.5f, cv::Scalar(255, 0, 0));
			}
		}
		cv::imshow("result", cv_img);
		cv::waitKey(0);
		cv::destroyWindow("result");
	}
#endif
}














template <typename Dtype>
int AnnotatedDataLayer<Dtype>::getNumTrainData() {
	return this->dataReader.getNumData();
}

template <typename Dtype>
int AnnotatedDataLayer<Dtype>::getNumTestData() {
	return 0;
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::shuffleTrainDataSet() {

}








/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* AnnotatedDataLayer<Dtype>::initLayer() {
	AnnotatedDataLayer* layer = NULL;
	SNEW(layer, AnnotatedDataLayer<Dtype>);
	SASSUME0(layer != NULL);

    if ((WorkContext::curBootMode == BootMode::ServerClientMode) &&
        SPARAM(USE_INPUT_DATA_PROVIDER)) {
    	//const string& name = GET_PROP(prop, AnnotatedData, name);
    	const string& name = "AnnotatedDataLayer";
        InputDataProvider::addPool(WorkContext::curNetworkID, WorkContext::curDOPID,
            name, DRType::DatumType, (void*)&layer->dataReader);
        layer->inputPool = InputDataProvider::getInputPool(WorkContext::curNetworkID,
        		WorkContext::curDOPID, name);
    }
    return (void*)layer;
}

template<typename Dtype>
void AnnotatedDataLayer<Dtype>::destroyLayer(void* instancePtr) {
    AnnotatedDataLayer<Dtype>* layer = (AnnotatedDataLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void AnnotatedDataLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	SASSERT0(!isInput);
	SASSERT0(index < 2);

    AnnotatedDataLayer<Dtype>* layer = (AnnotatedDataLayer<Dtype>*)instancePtr;
	SASSERT0(layer->_outputData.size() == index);
	layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool AnnotatedDataLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    AnnotatedDataLayer<Dtype>* layer = (AnnotatedDataLayer<Dtype>*)instancePtr;
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
void AnnotatedDataLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	AnnotatedDataLayer<Dtype>* layer = (AnnotatedDataLayer<Dtype>*)instancePtr;
	layer->feedforward();

}

template<typename Dtype>
void AnnotatedDataLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void AnnotatedDataLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool AnnotatedDataLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    // 1. checkShape()를 호출하는 것부터 일단 inner layer는 아닌 상황임.
    // GET_PROP 대신 SLPROP을 사용해도 무방.
    // 2. 현재 infer shape에서 crop과 resize만 사용. 별도의 param 업데이트는 생략. 
    DataTransformer<Dtype> dataTransformer(&SLPROP(AnnotatedData, dataTransformParam));
	DataTransformParam& dataTransformParam = dataTransformer.param;
	dataTransformParam.resizeParam = SLPROP(AnnotatedData, resizeParam);
	dataTransformParam.resizeParam.updateInterpMode();

    SDFHeader header = SDF::retrieveSDFHeader(SLPROP(DataInput, source));
    const int channels = header.channels;
    const int height = header.maxHeight;
    const int width = header.maxWidth;

    vector<uint32_t> inferredShape = dataTransformer.inferDataShape(channels, height, width);

    TensorShape outputShape1;
    outputShape1.N = SNPROP(batchSize);
    outputShape1.C = inferredShape[1];
    outputShape1.H = inferredShape[2];
    outputShape1.W = inferredShape[3];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t AnnotatedDataLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class AnnotatedDataLayer<float>;
