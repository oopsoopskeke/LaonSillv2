/**
 * @file YOLOInputLayer.cpp
 * @date 2017-12-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "YOLOInputLayer.h"
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
YOLOInputLayer<Dtype>::YOLOInputLayer()
: YOLOInputLayer(NULL) {}

template <typename Dtype>
YOLOInputLayer<Dtype>::YOLOInputLayer(_YOLOInputPropLayer* prop)
: InputLayer<Dtype>(),
  dataReader(GET_PROP(prop, YOLOInput, source)),
  dataTransformer(&GET_PROP(prop, YOLOInput, dataTransformParam)){
	this->type = Layer<Dtype>::YOLOInput;
	const string dataSetName = GET_PROP(prop, YOLOInput, dataSetName);
	if (dataSetName.empty()) {
		this->dataReader.selectDataSetByIndex(0);
	} else {
		this->dataReader.selectDataSetByName(dataSetName);
	}


	if (prop) {
		this->prop = NULL;
		SNEW(this->prop, _YOLOInputPropLayer);
		SASSUME0(this->prop != NULL);
		*(this->prop) = *(prop);
	} else {
		this->prop = NULL;
	}

	DataTransformParam& dataTransformParam = this->dataTransformer.param;
	dataTransformParam.resizeParam = GET_PROP(prop, YOLOInput, resizeParam);
	dataTransformParam.resizeParam.updateInterpMode();

	this->outputLabels = !(GET_PROP(prop, YOLOInput, output).size() == 1);
	this->hasAnnoType = !(GET_PROP(prop, YOLOInput, annoType) == AnnotationType::ANNO_NONE);

	this->labelMapFile = GET_PROP(prop, YOLOInput, labelMapFile);

    // XXX: 나중에 확인해보자 아래코드는..
	// Make sure dimension is consistent within batch.
	if (this->dataTransformer.param.resizeParam.prob >= 0.f) {
		if (this->dataTransformer.param.resizeParam.resizeMode == ResizeMode::FIT_SMALL_SIZE) {
			SASSERT(SNPROP(batchSize) == 1, "Only support batch size of 1 for FIT_SMALL_SIZE.");
		}
	}
}

template <typename Dtype>
YOLOInputLayer<Dtype>::~YOLOInputLayer() {}

template <typename Dtype>
void YOLOInputLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < GET_PROP(prop, YOLOInput, output).size(); i++) {
			GET_PROP(prop, YOLOInput, input).push_back(
                GET_PROP(prop, YOLOInput, output)[i]);
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
    SASSUME0(this->outputLabels);
    vector<uint32_t> labelShape(4, 1);
    SASSUME0(this->hasAnnoType);

    AnnotationType annoType = GET_PROP(prop, YOLOInput, annoType);
    SASSUME0 (annoType == AnnotationType::BBOX);

    /**
     * output tensor는 다음과 같이 구성이 됩니다.
     *
     * {1, 1, obj개수, 8}
     *
     * 1. item_id
     * 2. group_label
     * 3. instance_id
     * 4. xmin
     * 5. ymin
     * 6. xmax
     * 7. ymax
     * 8. difficult
     *
     **/
    labelShape[0] = 1;
    labelShape[1] = 1;
    labelShape[2] = GET_PROP(prop, YOLOInput, gtBoxMax) * batchSize;
    labelShape[3] = 8;
    this->_outputData[1]->reshape(labelShape);

    SDELETE(annoDatum);
}

template <typename Dtype>
void YOLOInputLayer<Dtype>::feedforward() {
	reshape();
	load_batch();
}

template <typename Dtype>
void YOLOInputLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
	load_batch();
}

template <typename Dtype>
void YOLOInputLayer<Dtype>::load_batch() {
	Dtype* outputData = this->_outputData[0]->mutable_host_data();
	Dtype* outputLabel = NULL;
    outputLabel = this->_outputData[1]->mutable_host_data();

	// Store transformed annotation.

	const int batchSize = SNPROP(batchSize);
	DataTransformParam& transformParam = this->dataTransformer.param;

    int labelElemCountPerBatch = GET_PROP(prop, YOLOInput, gtBoxMax) * 8;

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

		// Apply data transformations (mirror, scale, crop ... )
		//int offset = this->_outputData[0]->offset(itemId);
		//Dtype* dataPtr = this->_outputData[0]->mutable_host_data() + offset;
		vector<AnnotationGroup> transformedAnnoVec;
		SASSUME0 (this->outputLabels);
        SASSUME0 (this->hasAnnoType);

        // Make sure all data have same annotation type.
        SASSERT(annoDatum->type != AnnotationType::ANNO_NONE,
                "Some datum misses AnnotationType.");
        SASSERT(GET_PROP(prop, YOLOInput, annoType) == annoDatum->type,
                "Different AnnotationType.");

        // Transform datum and annotation_group at the same time.
        this->dataTransformer.transform(annoDatum, this->_outputData[0], itemId,
                transformedAnnoVec);

        SASSUME0 (GET_PROP(prop, YOLOInput, annoType) == AnnotationType::BBOX);

        int labelBaseIndex = itemId * labelElemCountPerBatch;
        for (int i = 0; i < GET_PROP(prop, YOLOInput, gtBoxMax); i++) {
            outputLabel[labelBaseIndex + i * 8 + 0] = -1.0;
            outputLabel[labelBaseIndex + i * 8 + 1] = 0.0;
            outputLabel[labelBaseIndex + i * 8 + 2] = 0.0;
            outputLabel[labelBaseIndex + i * 8 + 3] = 0.0;
            outputLabel[labelBaseIndex + i * 8 + 4] = 0.0;
            outputLabel[labelBaseIndex + i * 8 + 5] = 0.0;
            outputLabel[labelBaseIndex + i * 8 + 6] = 0.0;
            outputLabel[labelBaseIndex + i * 8 + 7] = 0.0;
        }
        int bboxIdx = 0;

        for (int g = 0; g < transformedAnnoVec.size(); g++) {
            int numBBoxes = transformedAnnoVec[g].annotations.size();

            for (int a = 0; a < numBBoxes; a++) {
                const Annotation_s& anno = transformedAnnoVec[g].annotations[a];
                const NormalizedBBox& bbox = anno.bbox;

                int classLabel = transformedAnnoVec[g].group_label;

                //cout << "class label : " << classLabel << endl;
                //cout << "box x : " << bbox.xmin << " ~ " << bbox.xmax <<
                    //", y : " << bbox.ymin << " ~ " << bbox.ymax << endl;

                int outputLabelBaseIndex = labelBaseIndex + bboxIdx * 8;
                outputLabel[outputLabelBaseIndex + 0] = (float)itemId;
                outputLabel[outputLabelBaseIndex + 1] = (float)classLabel;
                outputLabel[outputLabelBaseIndex + 2] = (float)bboxIdx;
                outputLabel[outputLabelBaseIndex + 3] = (float)bbox.xmin;
                outputLabel[outputLabelBaseIndex + 4] = (float)bbox.ymin;
                outputLabel[outputLabelBaseIndex + 5] = (float)bbox.xmax;
                outputLabel[outputLabelBaseIndex + 6] = (float)bbox.ymax;
                outputLabel[outputLabelBaseIndex + 7] = 0.0; // difficult. current meaningless

                bboxIdx++;
                if (bboxIdx >= GET_PROP(prop, YOLOInput, gtBoxMax))
                    continue;

            }
        }
        // clear memory.
        SDELETE(annoDatum);
	}
}

template <typename Dtype>
int YOLOInputLayer<Dtype>::getNumTrainData() {
	return this->dataReader.getNumData();
}

template <typename Dtype>
int YOLOInputLayer<Dtype>::getNumTestData() {
	return 0;
}

template <typename Dtype>
void YOLOInputLayer<Dtype>::shuffleTrainDataSet() {

}

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* YOLOInputLayer<Dtype>::initLayer() {
	YOLOInputLayer* layer = NULL;
	SNEW(layer, YOLOInputLayer<Dtype>);
	SASSUME0(layer != NULL);

    if ((WorkContext::curBootMode == BootMode::ServerClientMode) &&
        SPARAM(USE_INPUT_DATA_PROVIDER)) {
    	//const string& name = GET_PROP(prop, YOLOInput, name);
    	const string& name = "YOLOInputLayer";
        InputDataProvider::addPool(WorkContext::curNetworkID, WorkContext::curDOPID,
            name, DRType::DatumType, (void*)&layer->dataReader);
        layer->inputPool = InputDataProvider::getInputPool(WorkContext::curNetworkID,
        		WorkContext::curDOPID, name);
    }
    return (void*)layer;
}

template<typename Dtype>
void YOLOInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLOInputLayer<Dtype>* layer = (YOLOInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLOInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	SASSERT0(!isInput);
	SASSERT0(index < 2);

    YOLOInputLayer<Dtype>* layer = (YOLOInputLayer<Dtype>*)instancePtr;
	SASSERT0(layer->_outputData.size() == index);
	layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool YOLOInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLOInputLayer<Dtype>* layer = (YOLOInputLayer<Dtype>*)instancePtr;
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
void YOLOInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	YOLOInputLayer<Dtype>* layer = (YOLOInputLayer<Dtype>*)instancePtr;
	layer->feedforward();

}

template<typename Dtype>
void YOLOInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void YOLOInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool YOLOInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    int imgHeight = 416;
    int imgWidth = 416;

    if (SLPROP(YOLOInput, resizeParam).prob == 1.0) {
        imgHeight = SLPROP(YOLOInput, resizeParam).height;
        imgWidth = SLPROP(YOLOInput, resizeParam).width;
    }

    if (SLPROP_BASE(output).size() < 1 || SLPROP_BASE(output).size() > 2) {
        SEVENT_PUSH(NETWORK_EVENT_TYPE_eVALIDATION,
            "YOLO Input Layer should have 1 or 2 output tensors but it has %d tensors",
            (int)SLPROP_BASE(output).size());
        return false;
    }

    TensorShape outputShape1;
    outputShape1.N = SNPROP(batchSize);
    outputShape1.C = 3;
    outputShape1.H = imgHeight;
    outputShape1.W = imgWidth;
    outputShape.push_back(outputShape1);

    if (SLPROP_BASE(output).size() == 2) {
        TensorShape outputShape2;
        outputShape2.N = 1; 
        outputShape2.C = 1;
        outputShape2.H = SNPROP(batchSize) * SLPROP(YOLOInput, gtBoxMax);
        outputShape2.W = 8;
        outputShape.push_back(outputShape2);
    }

    return true;
}

template<typename Dtype>
uint64_t YOLOInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class YOLOInputLayer<float>;
