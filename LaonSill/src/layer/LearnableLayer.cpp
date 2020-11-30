/*
 * LearnableLayer.cpp
 *
 *  Created on: 2017. 2. 21.
 *      Author: jhkim
 */

#include "LearnableLayer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"
#include "Update.h"

using namespace std;

template<typename Dtype>
LearnableLayer<Dtype>::LearnableLayer() : Layer<Dtype>() {
    Optimizer opt = (Optimizer)SNPROP(optimizer);
    this->numHistories = Update<Dtype>::getParamHistoryDataCount(opt);
}

template <typename Dtype>
double LearnableLayer<Dtype>::sumSquareParamsData() {
	double result = 0.0;
	for(uint32_t i = 0; i < this->_params.size(); i++) {
		result += this->_params[i]->sumsq_device_data();
	}
	return result;
}

template <typename Dtype>
double LearnableLayer<Dtype>::sumSquareParamsGrad() {
	double result = 0.0;
	for(uint32_t i = 0; i < this->_params.size(); i++) {
		result += this->_params[i]->sumsq_device_grad();
	}
	return result;
}

template <typename Dtype>
void LearnableLayer<Dtype>::scaleParamsGrad(float scale) {
	for(uint32_t i = 0; i < this->_params.size(); i++) {
		this->_params[i]->scale_device_grad(scale);
	}
}

template <typename Dtype>
uint32_t LearnableLayer<Dtype>::boundParams() {
	uint32_t updateCount = 0;
	for (uint32_t i = 0; i < this->_params.size(); i++) {
		updateCount += this->_params[i]->bound_grad();
	}
	return updateCount;
}

template <typename Dtype>
uint32_t LearnableLayer<Dtype>::numParams() {
	return this->_params.size();
}

template <typename Dtype>
void LearnableLayer<Dtype>::saveParams(ofstream& ofs) {
	uint32_t numParams = _params.size();
	//ofs.write((char*)&numParams, sizeof(uint32_t));
	for(uint32_t i = 0; i < numParams; i++) {
		_params[i]->save(ofs);
	}
}

template <typename Dtype>
void LearnableLayer<Dtype>::loadParams(ifstream& ifs) {
	uint32_t numParams;
	ifs.read((char*)&numParams, sizeof(uint32_t));
	for(uint32_t i = 0; i < numParams; i++) {
		_params[i]->load(ifs);
	}
}

template <typename Dtype>
void LearnableLayer<Dtype>::loadParams(map<string, Data<Dtype>*>& dataMap) {
	typename map<string, Data<Dtype>*>::iterator it;

	//char tempName[80];
	for (uint32_t i = 0; i < this->_params.size(); i++) {

		// XXX: so temporal ~~~
		//Util::refineParamName(this->_params[i]->_name.c_str(), tempName);
		//string refinedName(tempName);
		//cout << "refineName: " << refinedName << ", ";

		cout << "looking for " << this->_params[i]->_name;
		it = dataMap.find(this->_params[i]->_name.c_str());
		if (it == dataMap.end()) {
			cout << " ... could not find ... " << endl;
			continue;
		}
		cout << " ... found ... " << endl;

		this->_params[i]->reshapeLike(it->second);
		this->_params[i]->set_device_with_host_data(it->second->host_data());
		this->_paramsInitialized[i] = true;
	}
}

template<typename Dtype>
void LearnableLayer<Dtype>::donateParam(LearnableLayer<Dtype>* receiver) {
    receiver->_params.clear();
    receiver->_paramsHistory.clear();
    receiver->_paramsHistory2.clear();
    receiver->_paramsInitialized.clear();
    receiver->updateParams.clear();

    for (int i = 0; i < this->_params.size(); i++) {
        receiver->_params.push_back(this->_params[i]);
    }

    SASSERT0(this->_paramsHistory.size() == this->_paramsHistory2.size());

    for (int i = 0; i < this->_paramsHistory.size(); i++) {
        receiver->_paramsHistory.push_back(this->_paramsHistory[i]);
        receiver->_paramsHistory2.push_back(this->_paramsHistory2[i]);
    }

    for (int i = 0; i < this->_paramsInitialized.size(); i++) {
        receiver->_paramsInitialized.push_back(this->_paramsInitialized[i]);
    }

    for (int i = 0; i < this->updateParams.size(); i++) {
        receiver->updateParams.push_back(this->updateParams[i]);
    }
}

template <typename Dtype>
void LearnableLayer<Dtype>::resizeParam(const int numParams) {
    this->_params.resize(numParams);
    this->_paramsHistory.resize(numParams);
    this->_paramsHistory2.resize(numParams);
    this->_paramsInitialized.resize(numParams);
}

template <typename Dtype>
void LearnableLayer<Dtype>::initParam(const int paramIdx, const std::string& paramName) {
	SASSERT0(paramIdx < this->_params.size());
	SASSERT0(paramIdx < this->_paramsHistory.size());
	SASSERT0(paramIdx < this->_paramsHistory2.size());

	const string& layerName = SLPROP_BASE(name);
	SNEW(this->_params[paramIdx], Data<Dtype>, layerName + "_" + paramName);
	SASSUME0(this->_params[paramIdx] != NULL);

    if (this->numHistories >= 1) {
        SNEW(this->_paramsHistory[paramIdx], Data<Dtype>, layerName + "_" + paramName + "_history");
        SASSUME0(this->_paramsHistory[paramIdx] != NULL);
    } else {
        this->_paramsHistory[paramIdx] = NULL;
    }

    if (this->numHistories >= 2) {
	    SNEW(this->_paramsHistory2[paramIdx], Data<Dtype>, layerName + "_" + paramName + "_history2");
	    SASSUME0(this->_paramsHistory2[paramIdx] != NULL);
    } else {
        this->_paramsHistory2[paramIdx] = NULL;
    }
	this->_paramsInitialized[paramIdx] = false;

	UpdateParam updateParam;
	updateParam.paramType = paramIdx;
	updateParam.paramDataPtr = (void*)this->_params[paramIdx];
	updateParam.paramHis1Ptr = (void*)this->_paramsHistory[paramIdx];
	updateParam.paramHis2Ptr = (void*)this->_paramsHistory2[paramIdx];
	this->updateParams.push_back(updateParam);
}

template <typename Dtype>
void LearnableLayer<Dtype>::reshapeParam(const int paramIdx, const vector<uint32_t>& shape) {
	SASSERT0(paramIdx < this->_params.size());
	SASSERT0(paramIdx < this->_paramsHistory.size());
	SASSERT0(paramIdx < this->_paramsHistory2.size());

	this->_params[paramIdx]->reshape(shape);

    if (this->numHistories >= 1) {
	    this->_paramsHistory[paramIdx]->reshape(shape);
    }
    if (this->numHistories >= 2) {
	    this->_paramsHistory2[paramIdx]->reshape(shape);
    }
}

template <typename Dtype>
void LearnableLayer<Dtype>::releaseParam(const int paramIdx) {
	SASSERT0(paramIdx < this->_params.size());
	SASSERT0(paramIdx < this->_paramsHistory.size());
	SASSERT0(paramIdx < this->_paramsHistory2.size());

	if (this->_params[paramIdx] != NULL) {
		SDELETE(this->_params[paramIdx]);
		this->_params[paramIdx] = NULL;
	}
	if (this->_paramsHistory[paramIdx] != NULL) {
		SDELETE(this->_paramsHistory[paramIdx]);
		this->_paramsHistory[paramIdx] = NULL;
	}
	if (this->_paramsHistory2[paramIdx] != NULL) {
		SDELETE(this->_paramsHistory2[paramIdx]);
		this->_paramsHistory2[paramIdx] = NULL;
	}
}


template class LearnableLayer<float>;













