#include "TestUtil.h"
#include "Cuda.h"
#include "BaseLayer.h"
#include "LearnableLayer.h"
#include "SysLog.h"
#include "PlanParser.h"
#include "Network.h"
#include "SplitLayer.h"
#include "MemoryMgmt.h"
#include "PlanOptimizer.h"
#include "MeasureManager.h"

using namespace std;
using namespace cnpy;



const vector<uint32_t> getShape(const string& data_key, NpyArray& npyArray);
const string getDataKeyFromDataName(const string& data_name);
const string getDataTypeFromDataName(const string& data_name);
void Tokenize(const string& str, vector<string>& tokens, const string& delimiters = " ", bool print = false);

template <typename T, typename S>
bool hasKey(map<T, S*>& dict, const T& key);






void setUpCuda(int gpuid) {
	SASSERT0(gpuid >= 0);

	checkCudaErrors(cudaSetDevice(gpuid));
	checkCUBLAS(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));
}

void cleanUpCuda() {
	checkCUBLAS(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));
}


/**
 * layer name 기반의 npz 파일을 load하던 코드에서 generic하게 npz 파일 (blobs, params...)을
 * load하게 되었는데 변수명은 그대로 layer_name로 남아 있는 것으로 추정.
 */
void buildNameDataMapFromNpzFile(const string& npz_path, const string& layer_name,
		map<string, Data<float>*>& nameDataMap) {

	char from = '/';
	char to = '*';
	string safe_layer_name = layer_name;
	std::replace(safe_layer_name.begin(), safe_layer_name.end(), from, to);
	//cout << "layer_name has changed from " << layer_name << " to " << safe_layer_name << endl;

	const string npz_file = npz_path + safe_layer_name + ".npz";
	npz_t cnpy_npz;
	try {
		cnpy_npz = npz_load(npz_file);
	} catch (std::runtime_error e) {
		//std::cout << e << std::endl;
		cout << "failed to load " << npz_file << endl;
		return;
	}
	//printNpzFiles(cnpy_npz);

	for (npz_t::iterator itr = cnpy_npz.begin(); itr != cnpy_npz.end(); itr++) {
		string data_name = itr->first;
		NpyArray npyArray = itr->second;

		const string data_key = getDataKeyFromDataName(data_name);
		const string data_type = getDataTypeFromDataName(data_name);

		//cout << data_name << ": <" << data_key << "/" << data_type << ">" << endl;

		Data<float>* data = retrieveValueFromMap(nameDataMap, data_key);
		if (!data) {
			data = NULL;
			SNEW(data, Data<float>, data_key, false);
			SASSUME0(data != NULL);
			const vector<uint32_t> shape = getShape(data_key, npyArray);
			data->reshape(shape);
			nameDataMap[data_key] = data;
		}

		if (data_type == TYPE_DATA) {
			data->set_host_data((float*)npyArray.data);
		} else if (data_type == TYPE_DIFF) {
			data->set_host_grad((float*)npyArray.data);
		}
	}
}




const vector<uint32_t> getShape(const string& data_key, NpyArray& npyArray) {
	vector<uint32_t> shape(4);
	const uint32_t shapeSize = npyArray.shape.size();

	/*
	if (shapeSize == 4)
		shape = npyArray.shape;
	else if (shapeSize == 2) {

	}
	*/


	vector<string> tokens;
	Tokenize(data_key, tokens, "*", false);
	SASSERT0(tokens.size() == 3);

	if (tokens[1] == "params") {
		for (uint32_t i = 0; i < 4 - shapeSize; i++) {
			shape[i] = 1;
		}
		for (uint32_t i = 4 - shapeSize; i < 4; i++) {
			shape[i] = npyArray.shape[i - (4 - shapeSize)];
			if (shape[i] == 0)
				shape[i] = 1;
		}
	} else if (tokens[1] == "bottom" || tokens[1] == "top" || tokens[1] == "blobs") {
		SASSERT(shapeSize >= 1 && shapeSize <= 4,
				"shapeSize is %d", shapeSize);
		if (shapeSize == 1) {
			shape[0] = npyArray.shape[0];
			shape[1] = 1;
			shape[2] = 1;
			shape[3] = 1;
		} else if (shapeSize == 2) {
			cout << "***********CAUTION: Consult with JHKIM!!!************" << endl;
			/*
			shape[0] = npyArray.shape[0];
			shape[1] = 1;
			shape[2] = npyArray.shape[1];
			shape[3] = 1;
			*/
			/*
			shape[0] = 1;
			shape[1] = 1;
			shape[2] = npyArray.shape[0];
			shape[3] = npyArray.shape[1];
			*/
			shape[0] = npyArray.shape[0];
			shape[1] = npyArray.shape[1];
			shape[2] = 1;
			shape[3] = 1;
		} else if (shapeSize == 3) {
			shape[0] = 1;
			shape[1] = npyArray.shape[0];
			shape[2] = npyArray.shape[1];
			shape[3] = npyArray.shape[2];
		} else if (shapeSize == 4) {
			shape = npyArray.shape;
		}
	}

	return shape;
}

const string getDataKeyFromDataName(const string& data_name) {
	SASSERT0(data_name.length() > 5);
	return data_name.substr(0, data_name.length() - 5);
}

const string getDataTypeFromDataName(const string& data_name) {
	SASSERT0(data_name.length() > 5);
	const string data_type = data_name.substr(data_name.length() - 4);
	SASSERT0(data_type == TYPE_DATA || data_type == TYPE_DIFF);
	return data_type;
}



template <typename T, typename S>
bool hasKey(map<T, S*>& dict, const T& key) {
	typename map<T, S*>::iterator itr = dict.find(key);
	return (itr != dict.end());
}
template bool hasKey(map<string, Data<float>*>& dict, const string& key);

template <typename T, typename S>
S* retrieveValueFromMap(map<T, S*>& dict, const T& key) {
	typename map<T, S*>::iterator itr = dict.find(key);
	if (itr == dict.end()) {
		return 0;
	} else
		return itr->second;
}
template Data<float>* retrieveValueFromMap(map<string, Data<float>*>& dict,
		const string& key);


template <typename T, typename S>
void cleanUpMap(map<T, S*>& dict) {
	typename map<T, S*>::iterator itr;
	for (itr = dict.begin(); itr != dict.end(); itr++) {
		if (itr->second)
			SDELETE(itr->second);
	}
}
template void cleanUpMap(map<string, Data<float>*>& dict);

template <typename T>
void cleanUpObject(T* obj) {
	if (obj)
		SDELETE(obj);
}
template void cleanUpObject(Layer<float>* obj);
//template void cleanUpObject(Layer<float>::Builder* obj);
template void cleanUpObject(LearnableLayer<float>* obj);
//template void cleanUpObject(LearnableLayer<float>::Builder* obj);
//template void cleanUpObject(LayersConfig<float>* obj);




void printNpzFiles(npz_t& cnpy_npz) {
	cout << "<npz_t array list>----------------" << endl;
	for (npz_t::iterator itr = cnpy_npz.begin(); itr != cnpy_npz.end(); itr++) {
		std::cout << itr->first << std::endl;
	}
	cout << "----------------------------------" << endl;
}


void printData(vector<Data<float>*>& dataVec, DataType dataType) {
	printConfigOn();
	for (uint32_t i = 0; i < dataVec.size(); i++) {
		switch (dataType) {
		case DATA:
			dataVec[i]->print_data({}, false);
			break;
		case GRAD:
			dataVec[i]->print_grad({}, false);
			break;
		case DATA_GRAD:
			dataVec[i]->print_data({}, false);
			dataVec[i]->print_grad({}, false);
			break;
		}
	}
	printConfigOff();
}

void printConfigOn() {
	Data<float>::printConfig = true;
	SyncMem<float>::printConfig = true;
}

void printConfigOff() {
	Data<float>::printConfig = false;
	SyncMem<float>::printConfig = false;
}



void fillLayerDataVec(const vector<string>& dataNameVec, vector<Data<float>*>& dataVec) {
	dataVec.clear();
	for (uint32_t i = 0; i < dataNameVec.size(); i++) {
		Data<float>* data = NULL;
		SNEW(data, Data<float>, dataNameVec[i]);
		SASSUME0(data != NULL);
		dataVec.push_back(data);
	}
}







bool isSplitDataName(string dataName) {
	const string delimiter = "_";

	size_t pos = 0;
	vector<string> tokens;
	while ((pos = dataName.find(delimiter)) != std::string::npos) {
		string token = dataName.substr(0, pos);
		if (token.length() > 0) {
			tokens.push_back(token);
		}
		dataName.erase(0, pos + delimiter.length());
	}
	if (dataName.length() > 0) {
		tokens.push_back(dataName);
	}

	const int tokenSize = tokens.size();
	if (tokenSize < 5) {
		return false;
	}

	// 결정적이지는 않음.
	// 일반 사용자의 레이어에 이런 케이스가 없다고 가정함
	return (tokens[tokenSize - 2] == "split");
}


void Tokenize(const string& str, vector<string>& tokens, const string& delimiters, bool print) {
	// 맨 첫 글자가 구분자인 경우 무시
	string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// 구분자가 아닌 첫 글자를 찾는다
	string::size_type pos = str.find_first_of(delimiters, lastPos);

	while (string::npos != pos || string::npos != lastPos) {
		// token을 찾았으니 vector에 추가한다
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// 구분자를 뛰어넘는다.  "not_of"에 주의하라
		lastPos = str.find_first_not_of(delimiters, pos);
		// 다음 구분자가 아닌 글자를 찾는다
		pos = str.find_first_of(delimiters, lastPos);
	}

	if (print) {
		cout << "Tokenize Result------------------------" << endl;
		for (uint32_t i = 0; i < tokens.size(); i++) {
			cout << "'" << tokens[i] << "'" << endl;
		}
		cout << "---------------------------------" << endl;
	}
}













template <typename Dtype>
string PrepareContext(const std::string& networkFilePath, const int numSteps) {
	// network definition file로부터 network를 생성하고 network id를 반환
	string networkID = PlanParser::loadNetwork(networkFilePath);
	// 현재의 context를 생성한 network로 설정
	WorkContext::updateNetwork(networkID);
	PlanOptimizer::buildPlans(networkID);



	WorkContext::updateNetwork(networkID);
	WorkContext::updatePlan(0, true);


	//WorkContext::updatePlan(0, true);

	// network id로 생성된 network를 조회
	//Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
	// network에 대해 최대 iteration? epoch?을 지정하여 build
	//network->build(numSteps);



	return networkID;
}

template string PrepareContext<float>(const std::string& networkFilePath, const int numSteps);


template <typename Dtype>
void RetrieveLayers(const string networkID,
		std::vector<std::pair<int, Layer<Dtype>*>>* outerLayers,
		std::vector<std::pair<int, Layer<Dtype>*>>* layers,
		std::pair<int, InputLayer<Dtype>*>* inputLayer,
		std::vector<std::pair<int, LossLayer<Dtype>*>>* lossLayers,
		std::vector<std::pair<int, LearnableLayer<Dtype>*>>* learnableLayers) {
	// outerLayers: (!splitLayer and innerLayer) 경우 제외 모든 레이어
	// layers: outerLayers 중, split layer가 아닌 레이어들
	// inputLayer: layers 중, input layer, 하나라고 가정
	// lossLayers: layers 중, loss layer
	// learnableLayers: layers 중, learnable layer

	Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
	PhysicalPlan* pp = WorkContext::curPhysicalPlan;
	// PhysicalPlan의 instanceMap: <layerID, Layer*>의 map
	for (map<int, void*>::iterator iter = pp->instanceMap.begin();
		iter != pp->instanceMap.end(); iter++) {

		int layerID = iter->first;
		void* instancePtr = iter->second;
		Layer<Dtype>* layer = (Layer<Dtype>*)instancePtr;

		// split layer가 아니고 inner layer인 경우? skip
		// (무슨 케이스지?)
		if (!dynamic_cast<SplitLayer<Dtype>*>(layer) &&
			// skip inner layer
			network->isInnerLayer(layerID)) {
				continue;
		}

		if (outerLayers) {
			// 그 외, 모든 layer를 outerLayers에 추가
			outerLayers->push_back(std::make_pair(layerID, layer));
		}

		if (dynamic_cast<SplitLayer<Dtype>*>(layer)) {
			continue;
		}

		// 해당 network에 대해 layerID의 layer context 설정
		WorkContext::updateLayer(networkID, layerID);

		if (layers) {
			// split layer를 제외, 모든 layer를 layers에 추가
			layers->push_back(std::make_pair(layerID, layer));
		}

		if (inputLayer && dynamic_cast<InputLayer<Dtype>*>(layer)) {
			*inputLayer = std::make_pair(layerID, (InputLayer<Dtype>*)instancePtr);
		}

		if (lossLayers && dynamic_cast<LossLayer<Dtype>*>(layer)) {
			lossLayers->push_back(
					std::make_pair(layerID, (LossLayer<Dtype>*)instancePtr));
		}

		if (learnableLayers && SLPROP_BASE(learnable)) {
			learnableLayers->push_back(
					std::make_pair(layerID, (LearnableLayer<Dtype>*)instancePtr));
		}
	}
	if (inputLayer) {
		SASSERT0(inputLayer->second);
	}
}

template void RetrieveLayers<float>(const string networkID,
		std::vector<std::pair<int, Layer<float>*>>* outerLayers,
		std::vector<std::pair<int, Layer<float>*>>* layers,
		std::pair<int, InputLayer<float>*>* inputLayer,
		std::vector<std::pair<int, LossLayer<float>*>>* lossLayers,
		std::vector<std::pair<int, LearnableLayer<float>*>>* learnableLayers);


template <typename Dtype>
void PrintLayerList(const string networkID,
		const std::vector<std::pair<int, Layer<Dtype>*>>* layers,
		const std::vector<std::pair<int, LearnableLayer<Dtype>*>>* learnableLayers) {

	if (layers) {
		std::cout << "Layers: " << std::endl;
		for (int i = 0; i < layers->size(); i++) {
			int layerID = layers->at(i).first;
			Layer<Dtype>* layer = layers->at(i).second;

			WorkContext::updateLayer(networkID, layerID);
			std::cout << layer->getName() << std::endl;
		}
	}

	if (learnableLayers) {
		std::cout << std::endl << "LearnableLayers: " << std::endl;
		for (int i = 0; i < learnableLayers->size(); i++) {
			int layerID = learnableLayers->at(i).first;
			LearnableLayer<Dtype>* learnableLayer = learnableLayers->at(i).second;

			WorkContext::updateLayer(networkID, layerID);
			std::cout << learnableLayer->getName() << std::endl;
		}
	}
}

template void PrintLayerList<float>(const string networkID,
		const std::vector<std::pair<int, Layer<float>*>>* layers,
		const std::vector<std::pair<int, LearnableLayer<float>*>>* learnableLayers);


template <typename Dtype>
void PrintLayerDataConfig(const string networkID,
		const std::vector<std::pair<int, Layer<Dtype>*>>& layers) {

	std::cout << "::: LAYER DATA CONFIGURATION :::" << std::endl;
	for (int i = 0; i < layers.size(); i++) {
		//layers[i]->reshape();
		WorkContext::updateLayer(networkID, layers[i].first);
		layers[i].second->printDataConfig();
	}
}

template void PrintLayerDataConfig<float>(const string networkID,
		const std::vector<std::pair<int, Layer<float>*>>& layers);








template <typename Dtype>
void LoadParams(const std::string& networkName, const int numSteps, const NetworkStatus status,
		vector<map<string, Data<Dtype>*>>& nameParamsMapList) {
	// step별 param들을 모두 load to nameParamsMap

	// XXX: inference test를 위해 = 제거,
	// 일반 테스트시 '<' --> '<='로 복구해야 함!!!
	const int _numSteps = status == NetworkStatus::Train ? numSteps + 1 : numSteps;
	for (int i = 0; i < _numSteps; i++) {
		const string strIdx = to_string(i);
		map<string, Data<Dtype>*> nameParamsMap;
		buildNameDataMapFromNpzFile(NPZ_PATH + networkName + "/",
				networkName + PARAMS + strIdx, nameParamsMap);
		nameParamsMapList.push_back(nameParamsMap);
		//PrintNameDataMap("nameParamsMap" + strIdx, nameParamsMap);
	}
}

template void LoadParams<float>(const string& networkName, const int numSteps,
		const NetworkStatus status, vector<map<string, Data<float>*>>& nameParamsMapList);


template <typename Dtype>
void LoadBlobs(const string& networkName, const int numSteps,
		vector<map<string, Data<Dtype>*>>& nameBlobsMapList) {
	// step별 blob들을 모두 load to nameBlobsMap

	for (int i = 0; i < numSteps; i++) {
		const string strIdx = to_string(i);
		map<string, Data<Dtype>*> nameBlobsMap;
		buildNameDataMapFromNpzFile(NPZ_PATH + networkName + "/",
				networkName + BLOBS + strIdx, nameBlobsMap);
		nameBlobsMapList.push_back(nameBlobsMap);
		//PrintNameDataMap("nameBlobsMap" + strIdx, nameBlobsMap);
	}
}

template void LoadBlobs(const string& networkName, const int numSteps,
		vector<map<string, Data<float>*>>& nameBlobsMapList);




template <typename Dtype>
void FillParam(const string networkID,
		map<string, Data<Dtype>*>& nameDataMap,
		std::pair<int, LearnableLayer<Dtype>*>& learnableLayerPair) {

	WorkContext::updateLayer(networkID, learnableLayerPair.first);
	LearnableLayer<Dtype>* learnableLayer = learnableLayerPair.second;
	auto layerType = learnableLayer->getType();
	const string param_prefix = learnableLayer->getName() + SIG_PARAMS;

	for (uint32_t i = 0; i < learnableLayer->_params.size(); i++) {
		string key = param_prefix + to_string(i);
		cout << "FillParam: setting param for key->" << key << endl;

		// Caffe 방식의 BatchNormLayer의 경우
		Data<Dtype>* param = retrieveValueFromMap(nameDataMap, key);
		/*
		if (param == NULL && layerType == Layer<Dtype>::LayerType::BatchNorm2 && i >= 3) {

			// BatchNorm 레이어의 이름이 _bn으로 끝나고
			// Scale 레이어의 이름이 _scale로 끝난다고 전제, 강제 변환
			string layerName = learnableLayer->getName();
			//cout << "key converted from " << key << endl;
			key = layerName.replace(layerName.end() - 3, layerName.end(), "_scale") +
					SIG_PARAMS + to_string(i - 3);
			//cout << "to " << key << endl;

			param = retrieveValueFromMap(nameDataMap, key);
		}
		*/
		SASSERT0(param != 0);

		Data<Dtype>* targetParam = learnableLayer->_params[i];
		targetParam->set(param, false, 1);
	}

	for (int j = 0; j < learnableLayer->_params.size(); j++) {
		learnableLayer->_paramsInitialized[j] = true;
	}
}

template void FillParam(const string networkID,
		map<string, Data<float>*>& nameDataMap,
		std::pair<int, LearnableLayer<float>*>& learnableLayerPair);


template <typename Dtype>
void FillParams(const string networkID,
		std::vector<std::pair<int, LearnableLayer<Dtype>*>>& learnableLayers,
		map<string, Data<Dtype>*>& nameParamsMap) {

	for (int i = 0; i < learnableLayers.size(); i++) {
		FillParam(networkID, nameParamsMap, learnableLayers[i]);

		/*
		WorkContext::updateLayer(networkID, learnableLayers[i].first);
		LearnableLayer<Dtype>* learnableLayer = learnableLayers[i].second;
		fillParam(nameParamsMapList[0], learnableLayer->getName() + SIG_PARAMS,
				learnableLayer->_params);
		// 반드시 외부에서 params init되었음을 설정해야 한다.
		for (int j = 0; j < learnableLayer->_params.size(); j++) {
			learnableLayer->_paramsInitialized[j] = true;
		}
		*/
	}
	/*
	std::vector<LearnableLayer<Dtype>*>& learnableLayers =
			this->layersConfig->_learnableLayers;

	cout << "fill params ... ---------------------------------------------------" << endl;
	for (int i = 0; i < learnableLayers.size(); i++) {
		fillParam(this->nameParamsMapList[0], learnableLayers[i]->name + SIG_PARAMS,
				learnableLayers[i]->_params);
		// 반드시 외부에서 params init되었음을 설정해야 한다.
		for (int j = 0; j < learnableLayers[i]->_params.size(); j++) {
			learnableLayers[i]->_paramsInitialized[j] = true;
		}
	}
	cout << "-------------------------------------------------------------------" << endl;
	*/
}

template void FillParams(const string networkID,
		std::vector<std::pair<int, LearnableLayer<float>*>>& learnableLayers,
		map<string, Data<float>*>& nameParamsMapList);





/**
 * 지정된 dataEndType에 따라 해당 end의 data, grad에 모두 값을 채움
 */
template <typename Dtype>
void FillDatum(const string networkID,
		map<string, Data<Dtype>*>& nameDataMap,
		std::pair<int, Layer<Dtype>*>& layerPair,
		DataEndType dataEndType) {

	WorkContext::updateLayer(networkID, layerPair.first);
	Layer<Dtype>* layer = layerPair.second;
	//const string data_prefix = layer->getName() + SIG_BOTTOM;

	vector<Data<Dtype>*>* dataVec = NULL;
	switch(dataEndType) {
	case INPUT: dataVec = &layer->_inputData; break;
	case OUTPUT: dataVec = &layer->_outputData; break;
	}

	for (uint32_t i = 0; i < dataVec->size(); i++) {
		//const string dataName = layer->_inputData[i]->_name;
		//const string key = data_prefix + dataName;
		const string key = BLOBS_PREFIX + dataVec->at(i)->_name;

		Data<float>* data = retrieveValueFromMap(nameDataMap, key);
		SASSERT(data != 0, "couldnt find data %s ... \n", key.c_str());

		Data<float>* targetData = dataVec->at(i);

		switch(dataEndType){
		case INPUT: targetData->set(data, false, 1); break;
		case OUTPUT: targetData->set(data, false, 2); break;
		}

	}
}

template void FillDatum(const string networkID,
		map<string, Data<float>*>& nameDataMap,
		std::pair<int, Layer<float>*>& layerPair, DataEndType dataEndType);


template <typename Dtype>
void FillData(const string networkID,
		std::vector<std::pair<int, Layer<Dtype>*>>& layers,
		map<string, Data<Dtype>*>& nameBlobsMap, DataEndType dataEndType) {

	for (int i = 0; i < layers.size(); i++) {
		//WorkContext::updateLayer(networkID, layers[i].first);
		//Layer<Dtype>* layer = layers[i].second;
		//fillData(nameBlobsMapList[0], layer->getName() + SIG_BOTTOM, layer->_inputData);
		FillDatum(networkID, nameBlobsMap, layers[i], dataEndType);
	}
}

template void FillData(const string networkID,
		std::vector<std::pair<int, Layer<float>*>>& layers,
		map<string, Data<float>*>& nameBlobsMap, DataEndType dataEndType);


template <typename Dtype>
void BuildNameLayerMap(const string networkID,
		std::vector<std::pair<int, Layer<Dtype>*>>& layers,
		std::map<std::string, std::pair<int, Layer<Dtype>*>>& nameLayerMap) {

	for (int i = 0; i < layers.size(); i++) {
		WorkContext::updateLayer(networkID, layers[i].first);
		Layer<Dtype>* layer = layers[i].second;
		nameLayerMap[layer->getName()] = layers[i];
	}
}

template void BuildNameLayerMap(const string networkID,
		std::vector<std::pair<int, Layer<float>*>>& layers,
		std::map<std::string, std::pair<int, Layer<float>*>>& nameLayerMap);




template <typename Dtype>
void PrintNameDataMapList(const string& name,
		vector<map<string, Data<Dtype>*>>& nameDataMapList) {

	for (int i = 0; i < nameDataMapList.size(); i++) {
		PrintNameDataMap(name + "_" + to_string(i), nameDataMapList[i]);
	}
}

template void PrintNameDataMapList(const string& name,
		vector<map<string, Data<float>*>>& nameDataMapList);



template <typename Dtype>
void PrintNameDataMap(const string& name, map<string, Data<Dtype>*>& nameDataMap,
		bool printData) {
	printConfigOn();
	cout << "PrintNameDataMap: " << name << endl;
	for (typename map<string, Data<float>*>::iterator itr = nameDataMap.begin();
			itr != nameDataMap.end(); itr++) {

		if (printData) {
			itr->second->print_data({}, false);
			itr->second->print_grad({}, false);
		} else
			itr->second->print_shape();
	}
	printConfigOff();
}


template void PrintNameDataMap(const string& name, map<string, Data<float>*>& nameDataMap,
		bool printData);











//bool CompareData(const map<string, Data<Dtype>*>& nameDataMap, const string& data_prefix,
//		vector<Data<Dtype>*>& dataVec, uint32_t compareType) {

/**
 * CompareData는 layer의 outputData에 대해서 체크하도록 고정,
 * 나중에 필요할 경우 타입 별로 arg 받아서 처리할 것.
 */
template <typename Dtype>
bool CompareData(const string networkID,
		map<string, Data<Dtype>*>& nameDataMap,
		std::pair<int, Layer<Dtype>*>& layerPair, DataEndType dataEndType) {

	WorkContext::updateLayer(networkID, layerPair.first);
	Layer<Dtype>* layer = layerPair.second;

	bool final_result = true;
	vector<Data<Dtype>*>* dataVec = NULL;
	switch(dataEndType) {
	case INPUT: dataVec = &layer->_inputData; break;
	case OUTPUT: dataVec = &layer->_outputData; break;
	}

	for (uint32_t i = 0; i < dataVec->size(); i++) {
		Data<Dtype>* targetData = dataVec->at(i);
		bool partial_result = CompareData(nameDataMap, targetData, dataEndType);

		final_result = final_result && partial_result;
	}
	return final_result;
}

template bool CompareData(const string networkID,
		map<string, Data<float>*>& nameDataMap,
		std::pair<int, Layer<float>*>& layerPair, DataEndType dataEndType);


template <typename Dtype>
bool CompareData(map<string, Data<Dtype>*>& nameDataMap, Data<Dtype>* targetData,
		DataEndType dataEndType) {
	const string dataName = targetData->_name;


	bool result = false;
	Data<float>* data = NULL;
	//bool isSplit = false;
	// dataName이 splitLayer의 output에 해당하는 경우 일단 무시 ...
	if (isSplitDataName(dataName)) {
		cout << "SPLIT DATA: " << dataName << endl;
		//return false;
		//isSplit = true;


		vector<string> tokens;
		Tokenize(dataName, tokens, "_", false);

		int tokenIdx = 0;
		while (true) {
			string key = BLOBS_PREFIX;
			for (int i = 0; i < tokens.size()-1; i++) {
				key += tokens[i] + "_";
			}
			key += to_string(tokenIdx);
			cout << "split key: " << key << endl;
			data = retrieveValueFromMap(nameDataMap, key);
			if (data == NULL) {
				return result;
			}

			switch(dataEndType) {
			case INPUT: result = targetData->compareGrad(data, COMPARE_ERROR, false); break;
			case OUTPUT: result = targetData->compareData(data, COMPARE_ERROR, false); break;
			default: SASSERT0(false); break;
			}

			if (result) {
				cout << "[split data matched at " << tokenIdx << "]" << endl;
				return result;
			} else {
				cout << "[split data is not matched at " << tokenIdx << "]"<< endl;
			}
			tokenIdx++;
		}


	} else {
		const string key = BLOBS_PREFIX + dataName;
		data = retrieveValueFromMap(nameDataMap, key);
		SASSERT(data != 0, "Could not find Data named %s", key.c_str());

		switch(dataEndType) {
		case INPUT: result = targetData->compareGrad(data, COMPARE_ERROR); break;
		case OUTPUT: result = targetData->compareData(data, COMPARE_ERROR); break;
		default: SASSERT0(false); break;
		}
	}

	/**
	 * 레이어 테스트의 경우 data의 이름을 직접 입력,
	 * Split Data여도 이름이 정확하기 때문에 그냥 테스트할 수 있음.
	 * 반면 네트워크 테스트의 경우 현재 SplitLayer의 데이터 이름 규약이
	 * Caffe와 다르기 때문에 이름에 해당하는 데이터가 없는 경우 일단 그냥 무시한다.
	 */


	if (!result) {
		//printConfigOn();
		switch(dataEndType) {
		case INPUT:
			data->print_grad({}, false);
			targetData->print_grad({}, false);
			break;
		case OUTPUT:
			data->print_data({}, false);
			targetData->print_data({}, false);
			break;
		}
		//printConfigOff();
	}
	return result;
}

template bool CompareData(map<string, Data<float>*>& nameDataMap, Data<float>* targetData,
		DataEndType dataEndType);


template <typename Dtype>
bool CompareParam(map<string, Data<Dtype>*>& nameDataMap, const string& param_prefix,
		vector<Data<Dtype>*>& paramVec, DataType dataType) {

	bool finalResult = true;
	for (uint32_t i = 0; i < paramVec.size(); i++) {
		const string key = param_prefix + to_string(i);
		Data<Dtype>* data = retrieveValueFromMap(nameDataMap, key);
		SASSERT0(data != 0);

		Data<Dtype>* targetData = paramVec[i];

		targetData->print_shape();
		data->print_shape();

		bool result = false;
		switch(dataType) {
		case DATA:
			result = targetData->compareData(data, COMPARE_ERROR);
			break;
		case GRAD:
			result = targetData->compareGrad(data, COMPARE_ERROR);
			break;
		}

		if (!result) {
			switch(dataType) {
			case DATA:
				data->print_data({}, false);
				targetData->print_data({}, false);
				break;
			case GRAD:
				data->print_grad({}, false);
				targetData->print_grad({}, false);
			}
		}
		finalResult = finalResult && result;
	}
	return finalResult;
}

template bool CompareParam(map<string, Data<float>*>& nameDataMap, const string& param_prefix,
		vector<Data<float>*>& paramVec, DataType dataType);












