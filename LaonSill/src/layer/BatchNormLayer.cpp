/**
 * @file BatchNormLayer.cpp
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "BatchNormLayer.h"
#include "Util.h"
#include "SysLog.h"
#include "ColdLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer() : LearnableLayer<Dtype>() {
	this->type                  = Layer<Dtype>::BatchNorm;
    this->depth                 = 0;

	this->_paramsInitialized.resize(5);
	this->_paramsInitialized[ParamType::Gamma] = false;
	this->_paramsInitialized[ParamType::Beta] = false;
	this->_paramsInitialized[ParamType::GlobalMean] = false;
	this->_paramsInitialized[ParamType::GlobalVar] = false;
	this->_paramsInitialized[ParamType::GlobalCount] = false;

	this->_params.resize(5);
	this->_params[ParamType::Gamma] = NULL;
	SNEW(this->_params[ParamType::Gamma], Data<Dtype>, SLPROP_BASE(name) + "_gamma");
	SASSUME0(this->_params[ParamType::Gamma] != NULL);

	this->_params[ParamType::Beta] = NULL;
	SNEW(this->_params[ParamType::Beta], Data<Dtype>, SLPROP_BASE(name) + "_beta");
	SASSUME0(this->_params[ParamType::Beta] != NULL);

	this->_params[ParamType::GlobalMean] = NULL;
	SNEW(this->_params[ParamType::GlobalMean], Data<Dtype>, SLPROP_BASE(name) + "_global_mean");
	SASSUME0(this->_params[ParamType::GlobalMean] != NULL);

	this->_params[ParamType::GlobalVar] = NULL;
	SNEW(this->_params[ParamType::GlobalVar], Data<Dtype>, SLPROP_BASE(name) + "_global_var");
	SASSUME0(this->_params[ParamType::GlobalVar] != NULL);

	this->_params[ParamType::GlobalCount] = NULL;
	SNEW(this->_params[ParamType::GlobalCount], Data<Dtype>, SLPROP_BASE(name) + "_global_count");
	SASSUME0(this->_params[ParamType::GlobalCount] != NULL);

    Optimizer opt = (Optimizer)SNPROP(optimizer);
    int paramHistoryDataCount = Update<Dtype>::getParamHistoryDataCount(opt);

	this->_paramsHistory.resize(5);
	this->_paramsHistory[ParamType::Gamma] = NULL;
	this->_paramsHistory[ParamType::Beta] = NULL;
	this->_paramsHistory[ParamType::GlobalMean] = NULL;
	this->_paramsHistory[ParamType::GlobalVar] = NULL;
	this->_paramsHistory[ParamType::GlobalCount] = NULL;

    if (paramHistoryDataCount >= 1) {
        SNEW(this->_paramsHistory[ParamType::Gamma], Data<Dtype>, 
                SLPROP_BASE(name) + "_gamma_history");
        SASSUME0(this->_paramsHistory[ParamType::Gamma] != NULL);

        SNEW(this->_paramsHistory[ParamType::Beta], Data<Dtype>, 
                SLPROP_BASE(name) + "_beta_history");
        SASSUME0(this->_paramsHistory[ParamType::Beta] != NULL);

        SNEW(this->_paramsHistory[ParamType::GlobalMean], Data<Dtype>, 
                SLPROP_BASE(name) + "_global_mean_history");
        SASSUME0(this->_paramsHistory[ParamType::GlobalMean] != NULL);

        SNEW(this->_paramsHistory[ParamType::GlobalVar], Data<Dtype>, 
                SLPROP_BASE(name) + "_global_var_history");
        SASSUME0(this->_paramsHistory[ParamType::GlobalVar] != NULL);

        SNEW(this->_paramsHistory[ParamType::GlobalCount], Data<Dtype>, 
                SLPROP_BASE(name) + "_global_count_history");
        SASSUME0(this->_paramsHistory[ParamType::GlobalCount] != NULL);
    }


	this->_paramsHistory2.resize(5);
	this->_paramsHistory2[ParamType::Gamma] = NULL;
	this->_paramsHistory2[ParamType::Beta] = NULL;
	this->_paramsHistory2[ParamType::GlobalMean] = NULL;
	this->_paramsHistory2[ParamType::GlobalVar] = NULL;
	this->_paramsHistory2[ParamType::GlobalCount] = NULL;


    if (paramHistoryDataCount >= 2) {
        SNEW(this->_paramsHistory2[ParamType::Gamma], Data<Dtype>,
                SLPROP_BASE(name) + "_gamma_history2");
        SASSUME0(this->_paramsHistory2[ParamType::Gamma] != NULL);

        SNEW(this->_paramsHistory2[ParamType::Beta], Data<Dtype>, 
                SLPROP_BASE(name) + "_beta_history2");
        SASSUME0(this->_paramsHistory2[ParamType::Beta] != NULL);

        SNEW(this->_paramsHistory2[ParamType::GlobalMean], Data<Dtype>, 
                SLPROP_BASE(name) + "_global_mean_history2");
        SASSUME0(this->_paramsHistory2[ParamType::GlobalMean] != NULL);

        SNEW(this->_paramsHistory2[ParamType::GlobalVar], Data<Dtype>, 
                SLPROP_BASE(name) + "_global_var_history2");
        SASSUME0(this->_paramsHistory2[ParamType::GlobalVar] != NULL);

        SNEW(this->_paramsHistory2[ParamType::GlobalCount], Data<Dtype>, 
                SLPROP_BASE(name) + "_global_count_history2");
        SASSUME0(this->_paramsHistory2[ParamType::GlobalCount] != NULL);
    }

	this->meanSet = NULL;
	SNEW(this->meanSet, Data<Dtype>, SLPROP_BASE(name) + "_mean");
	SASSUME0(this->meanSet != NULL);

	this->varSet = NULL;
	SNEW(this->varSet, Data<Dtype>, SLPROP_BASE(name) + "_variance");
	SASSUME0(this->varSet != NULL);

	this->normInputSet = NULL;
	SNEW(this->normInputSet, Data<Dtype>, SLPROP_BASE(name) + "_normalizedInput");
	SASSUME0(this->normInputSet != NULL);
}

template<typename Dtype>
void BatchNormLayer<Dtype>::setTrain(bool train) {
    SLPROP(BatchNorm, train) = train;
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* BatchNormLayer<Dtype>::initLayer() {
	BatchNormLayer* layer = NULL;
	SNEW(layer, BatchNormLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void BatchNormLayer<Dtype>::destroyLayer(void* instancePtr) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void BatchNormLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool BatchNormLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void BatchNormLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void BatchNormLayer<Dtype>::backwardTensor(void* instancePtr) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void BatchNormLayer<Dtype>::learnTensor(void* instancePtr) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    layer->update();
}

template<typename Dtype>
bool BatchNormLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    if (inputShape.size() != 1) {
        return false;
    }

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t BatchNormLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    Optimizer opt = (Optimizer)SNPROP(optimizer);
    int paramHistoryDataCount = Update<Dtype>::getParamHistoryDataCount(opt);

	uint32_t batches = inputShape[0].N;
	uint32_t channels = inputShape[0].C;
	uint32_t rows = inputShape[0].H;
	uint32_t cols = inputShape[0].W;

    uint64_t eachParamSize = 
        4 * ALIGNUP(1 * channels * rows * cols * sizeof(Dtype), SPARAM(CUDA_MEMPAGE_SIZE)) +
        ALIGNUP(1 * 1 * 1 * 1 * sizeof(Dtype), SPARAM(CUDA_MEMPAGE_SIZE));
    
    uint64_t totalSize = (paramHistoryDataCount + 1 ) * eachParamSize;

    return totalSize;
}

template class BatchNormLayer<float>;
