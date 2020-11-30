/**
 * @file NoiseInputLayer.cpp
 * @date 2017-02-16
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <time.h>

#include <boost/random.hpp>
#include <boost/math/special_functions/next.hpp>

#include "common.h"
#include "NoiseInputLayer.h"
#include "InputLayer.h"
#include "Network.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

typedef boost::mt19937 RNGType;

template<typename Dtype>
NoiseInputLayer<Dtype>::NoiseInputLayer() : InputLayer<Dtype>() {
    this->type = Layer<Dtype>::NoiseInput;
    this->batchSize = 0;
    this->uniformArray = NULL;
    this->linearTransMatrix = NULL;
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::setRegenerateNoise(bool regenerate) {
    SLPROP(NoiseInput, regenerateNoise) = regenerate;
}

template<typename Dtype>
NoiseInputLayer<Dtype>::~NoiseInputLayer() {
    if (this->uniformArray != NULL) {
        SFREE(this->uniformArray);
    }
}

template <typename Dtype>
bool NoiseInputLayer<Dtype>::prepareUniformArray() {
    uint32_t batchSize = SNPROP(batchSize);
	RNGType rng;
    unsigned int seedValue = static_cast<unsigned int>(time(NULL)+getpid());
    rng.seed(seedValue);

    bool firstGenerate = false;

    if (this->uniformArray == NULL) {
        int allocSize = sizeof(Dtype) * SLPROP(NoiseInput, noiseDepth) * batchSize;
        this->uniformArray = NULL;
        SMALLOC(this->uniformArray, Dtype, allocSize);
        SASSERT0(this->uniformArray != NULL);
        firstGenerate = true;
    }

    if (firstGenerate || SLPROP(NoiseInput, regenerateNoise)) {
        boost::random::uniform_real_distribution<float> random_distribution(
            SLPROP(NoiseInput, noiseRangeLow), SLPROP(NoiseInput, noiseRangeHigh));
        boost::variate_generator<RNGType, boost::random::uniform_real_distribution<float>>
            variate_generator(rng, random_distribution);

        for (int i = 0; i < SLPROP(NoiseInput, noiseDepth) * batchSize; ++i) {
            this->uniformArray[i] = (Dtype)variate_generator();
        }

        return true;
    }

    return false;
}

template <typename Dtype>
void NoiseInputLayer<Dtype>::prepareLinearTranMatrix() {
    // FIXME: deprecated function. should be deleted!!!
}

template <typename Dtype>
void NoiseInputLayer<Dtype>::reshape() {
    uint32_t batchSize = SNPROP(batchSize);

    bool isNoiseGenerated = prepareUniformArray();

    if ((this->uniformArray == NULL) && (SLPROP(NoiseInput, useLinearTrans))) {
        prepareLinearTranMatrix();
    }

	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
		    SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}

    Layer<Dtype>::_adjustInputShape();

    this->batchSize = batchSize;
    if (!SLPROP(NoiseInput, useLinearTrans)) {
        this->_inputShape[0][0] = batchSize;
        this->_inputShape[0][1] = 1;
        this->_inputShape[0][2] = (unsigned int)SLPROP(NoiseInput, noiseDepth);
        this->_inputShape[0][3] = 1;

        this->_inputData[0]->reshape(this->_inputShape[0]);
    } else {
        this->_inputShape[0][0] = batchSize;
        this->_inputShape[0][1] = (unsigned int)SLPROP(NoiseInput, tranChannels);
        this->_inputShape[0][2] = (unsigned int)SLPROP(NoiseInput, tranRows);
        this->_inputShape[0][3] = (unsigned int)SLPROP(NoiseInput, tranCols);

        this->_inputData[0]->reshape(this->_inputShape[0]);
    }

    if (isNoiseGenerated) {
        int copyElemCount;
        if (SLPROP(NoiseInput, useLinearTrans)) {
            copyElemCount = SLPROP(NoiseInput, tranChannels) * SLPROP(NoiseInput, tranRows) *
                SLPROP(NoiseInput, tranCols) * batchSize;
            this->_inputData[0]->set_device_with_host_data(this->linearTransMatrix,
                0, copyElemCount); 
        } else {
            copyElemCount = SLPROP(NoiseInput, noiseDepth) * batchSize;
            this->_inputData[0]->set_device_with_host_data(this->uniformArray,
                0, copyElemCount); 
        }
    }
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    reshape();

}

template<typename Dtype>
int NoiseInputLayer<Dtype>::getNumTrainData() {
    if (this->_dataSet != NULL) {
        return this->_dataSet->getNumTrainData();
    } else {    
        uint32_t batches = SNPROP(batchSize);
        return batches;
    }
}

template<typename Dtype>
int NoiseInputLayer<Dtype>::getNumTestData() {
    if (this->_dataSet != NULL) {
        return this->_dataSet->getNumTestData();
    } else {
        return 1;
    }
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->_dataSet != NULL) {
        return this->_dataSet->shuffleTrainDataSet();
    }
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* NoiseInputLayer<Dtype>::initLayer() {
	NoiseInputLayer* layer = NULL;
	SNEW(layer, NoiseInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    NoiseInputLayer<Dtype>* layer = (NoiseInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    NoiseInputLayer<Dtype>* layer = (NoiseInputLayer<Dtype>*)instancePtr;

    SASSERT0(!isInput);
    SASSERT0(index == 0);
    SASSERT0(layer->_outputData.size() == 0);
    layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool NoiseInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    NoiseInputLayer<Dtype>* layer = (NoiseInputLayer<Dtype>*)instancePtr;
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
void NoiseInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    NoiseInputLayer<Dtype>* layer = (NoiseInputLayer<Dtype>*)instancePtr;
    layer->feedforward(miniBatchIdx);
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing..
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool NoiseInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {
  
    // XXX: 현재는 NoiseInputLayer를 사용하는 모듈은 오직 KistiKeyword 예제이다. 
    //      해당 레이어는 추후에 삭제가 될 예정이기 때문에 KistiKeryword 모듈에만 적합하게
    //      동작하는 수준으로만 checkShape 함수를 맞추도록 하겠다.

    TensorShape outputShape1;
    if (!SLPROP(NoiseInput, useLinearTrans)) {
        outputShape1.N = SNPROP(batchSize);
        outputShape1.C = 1;
        outputShape1.H = SLPROP(NoiseInput, noiseDepth);
        outputShape1.W = 1;
    } else {
        outputShape1.N = SNPROP(batchSize);
        outputShape1.C = SLPROP(NoiseInput, tranChannels);
        outputShape1.H = SLPROP(NoiseInput, tranRows);
        outputShape1.W = SLPROP(NoiseInput, tranCols);
    }
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t NoiseInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class NoiseInputLayer<float>;
