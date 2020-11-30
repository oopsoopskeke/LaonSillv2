/*
 * TestUtil.h
 *
 *  Created on: Feb 16, 2017
 *      Author: jkim
 */

#ifndef TESTUTIL_H_
#define TESTUTIL_H_

#include <map>
#include <vector>
#include <string>

#include "cnpy.h"
#include "Data.h"
#include "BaseLayer.h"
#include "InputLayer.h"
#include "LossLayer.h"
#include "LearnableLayer.h"



const std::string DELIM 		= "*";
const std::string TYPE_DATA 	= "data";
const std::string TYPE_DIFF 	= "diff";
const std::string SIG_BOTTOM 	= DELIM + "bottom" + DELIM;
const std::string SIG_TOP 		= DELIM + "top" + DELIM;
const std::string SIG_PARAMS 	= DELIM + "params" + DELIM;
const std::string BLOBS_PREFIX	= "anonymous" + DELIM + "blobs" + DELIM;
const std::string NPZ_PATH 		= "/home/jkim/Dev/data/numpy_array/";

const std::string PARAMS		= "_params_";
const std::string BLOBS			= "_blobs_";

const float COMPARE_ERROR 		= 1.0e-5;


enum DataType {
	DATA,
	GRAD,
	DATA_GRAD
};

enum DataEndType {
	INPUT,
	OUTPUT
};






// cuda device 설정
// cublas, cudnn handle 생성
void setUpCuda(int gpuid);
// cublas, cudnn handle 파괴
void cleanUpCuda();

// npz_path의 npz file로부터 layer_name에 해당하는 레이어 데이터를 조회, nameDataMap을 채움
void buildNameDataMapFromNpzFile(const std::string& npz_path, const std::string& layer_name,
		std::map<std::string, Data<float>*>& nameDataMap);

// dataNameVec에 해당하는 만큼 data 객체를 생성, dataVec에 추가
void fillLayerDataVec(const std::vector<std::string>& dataNameVec,
		std::vector<Data<float>*>& dataVec);





void printNpzFiles(cnpy::npz_t& cnpy_npz);
void printData(std::vector<Data<float>*>& dataVec, DataType dataType = DataType::DATA_GRAD);

void printConfigOn();
void printConfigOff();


template <typename T, typename S>
void cleanUpMap(std::map<T, S*>& dict);

template <typename T>
void cleanUpObject(T* obj);

template <typename T, typename S>
S* retrieveValueFromMap(std::map<T, S*>& dict, const T& key);















template <typename Dtype>
std::string PrepareContext(const std::string& networkFilePath, const int numSteps = 10);

template <typename Dtype>
void RetrieveLayers(const std::string networkID,
		std::vector<std::pair<int, Layer<Dtype>*>>* outerLayers = NULL,
		std::vector<std::pair<int, Layer<Dtype>*>>* layers = NULL,
		std::pair<int, InputLayer<Dtype>*>* inputLayer = NULL,
		std::vector<std::pair<int, LossLayer<Dtype>*>>* lossLayers = NULL,
		std::vector<std::pair<int, LearnableLayer<Dtype>*>>* learnableLayers = NULL);

template <typename Dtype>
void PrintLayerList(const std::string networkID,
		const std::vector<std::pair<int, Layer<Dtype>*>>* layers = NULL,
		const std::vector<std::pair<int, LearnableLayer<Dtype>*>>* learnableLayers = NULL);

template <typename Dtype>
void PrintLayerDataConfig(const std::string networkID,
		const std::vector<std::pair<int, Layer<Dtype>*>>& layers);

template <typename Dtype>
void LoadParams(const std::string& networkName, const int numSteps, const NetworkStatus status,
		std::vector<std::map<std::string, Data<Dtype>*>>& nameParamsMapList);

template <typename Dtype>
void LoadBlobs(const std::string& networkName, const int numSteps,
		std::vector<std::map<std::string, Data<Dtype>*>>& nameBlobsMapList);



// paramVec에 param_prefix에 해당하는 Data를 nameDataMap으로부터 조회하여 복사
template <typename Dtype>
void FillParam(const std::string networkID,
		std::map<std::string, Data<Dtype>*>& nameDataMap,
		std::pair<int, LearnableLayer<Dtype>*>& learnableLayerPair);

template <typename Dtype>
void FillParams(const std::string networkID,
		std::vector<std::pair<int, LearnableLayer<Dtype>*>>& learnableLayers,
		std::map<std::string, Data<Dtype>*>& nameParamsMap);


// dataVec에 data_prefix에 해당하는 Data를 nameDataMap으로부터 조회하여 복사
template <typename Dtype>
void FillDatum(const std::string networkID,
		std::map<std::string, Data<Dtype>*>& nameDataMap,
		std::pair<int, Layer<Dtype>*>& layerPair,
		DataEndType dataEndType);

template <typename Dtype>
void FillData(const std::string networkID,
		std::vector<std::pair<int, Layer<Dtype>*>>& layers,
		std::map<std::string, Data<Dtype>*>& nameBlobsMap,
		DataEndType dataEndType);


template <typename Dtype>
void BuildNameLayerMap(const std::string networkID,
		std::vector<std::pair<int, Layer<Dtype>*>>& layers,
		std::map<std::string, std::pair<int, Layer<Dtype>*>>& nameLayerMap);




template <typename Dtype>
void PrintNameDataMapList(const std::string& name,
		std::vector<std::map<std::string, Data<Dtype>*>>& nameDataMapList);

template <typename Dtype>
void PrintNameDataMap(const std::string& name,
		std::map<std::string, Data<Dtype>*>& nameDataMap, bool printData = false);



/**
 * dataEndType이 INPUT인 경우, input data의 grad를 비교,
 * OUTPUT인 경우, output data의 data를 비교
 */
template <typename Dtype>
bool CompareData(const std::string networkID,
		std::map<std::string, Data<Dtype>*>& nameDataMap,
		std::pair<int, Layer<Dtype>*>& layerPair, DataEndType dataEndType);

template <typename Dtype>
bool CompareData(std::map<std::string, Data<Dtype>*>& nameDataMap, Data<Dtype>* targetData,
		DataEndType dataEndType);

template <typename Dtype>
bool CompareParam(std::map<std::string, Data<Dtype>*>& nameDataMap,
		const std::string& param_prefix, std::vector<Data<Dtype>*>& paramVec,
		DataType dataType);



#endif /* TESTUTIL_H_ */
