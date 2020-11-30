/*
 * ParamManipulator.cpp
 *
 *  Created on: Jun 27, 2017
 *      Author: jkim
 */

#include "ParamManipulator.h"
#include "Data.h"
#include "SysLog.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
ParamManipulator<Dtype>::ParamManipulator(const string& oldParamPath,
		const string& newParamPath) {
	this->oldParamPath = oldParamPath;
	this->newParamPath = newParamPath;

	loadParam();
}

template <typename Dtype>
ParamManipulator<Dtype>::~ParamManipulator() {

	for (int i = 0; i < this->dataList.size(); i++) {
		if (this->dataList[i]) {
			SDELETE(this->dataList[i]);
		}
	}
}

template <typename Dtype>
void ParamManipulator<Dtype>::loadParam() {
    ifstream ifs(this->oldParamPath, std::ios::in | std::ios::binary);

    SASSERT0(ifs.is_open());

    uint32_t numData;
    ifs.read((char*)&numData, sizeof(uint32_t));

    for (uint32_t j = 0; j < numData; j++) {
    	Data<float>* data = NULL;
    	SNEW(data, Data<float>, "", true);
    	SASSUME0(data != NULL);
        data->load(ifs);

        string dataName = data->_name;
        map<string, Data<float>*>::iterator it = this->dataMap.find(dataName);
        SASSERT0(it == this->dataMap.end());

        this->dataMap[dataName] = data;
        this->dataList.push_back(data);
    }
    ifs.close();
}

template <typename Dtype>
void ParamManipulator<Dtype>::printParamList() {
	for (int i = 0; i < this->dataList.size(); i++) {
		cout << i << ":\t";
		this->dataList[i]->print_shape();
	}
}

template <typename Dtype>
void ParamManipulator<Dtype>::changeParamNames(
		const vector<pair<string, string>>& namePairList) {

	for (int i = 0; i < namePairList.size(); i++) {
		changeParamName(namePairList[i].first, namePairList[i].second);
	}
}


template <typename Dtype>
void ParamManipulator<Dtype>::changeParamName(const string& oldParamName,
		const string& newParamName) {
	cout << "Changing Param Name from " << oldParamName << " to " << newParamName << endl;

	Data<Dtype>* param = findParam(oldParamName);
	if (!param) {
		return;
	}

	this->dataMap.erase(oldParamName);
	param->_name = newParamName;
	this->dataMap[newParamName] = param;
}

template <typename Dtype>
void ParamManipulator<Dtype>::denormalizeParams(const vector<string>& paramNames,
		const vector<float>& means, const vector<float>& stds) {
	SASSERT0(paramNames.size() == 2);

	cout << "Denormalizing Param ";
	for (int i = 0; i < paramNames.size(); i++) {
		cout << paramNames[i] << ", ";
	}
	cout << endl;

	Data<Dtype>* param0 = findParam(paramNames[0]);
	Data<Dtype>* param1 = findParam(paramNames[1]);
	if (!param0 || !param1) {
		return;
	}

	Dtype* srcPtr0 = param0->mutable_host_data();
	const int numRows0 = param0->getShape(2);
	const int numCols0 = param0->getShape(3);
	int index;
	int id1;
	for (int row = 0; row < numRows0; row++) {
		id1 = row % 4;
		for (int col = 0; col < numCols0; col++) {
			index = row * numCols0 + col;
			srcPtr0[index] *= stds[id1];
		}
	}

	Dtype* srcPtr1 = param1->mutable_host_data();
	const int numRows1 = param1->getShape(1);
	for (int row = 0; row < numRows1; row++) {
		id1 = row % 4;
		srcPtr1[row] = srcPtr1[row] * stds[id1] + means[id1];
	}
}


template <typename Dtype>
void ParamManipulator<Dtype>::save() {
	ofstream paramOfs(this->newParamPath.c_str(), ios::out | ios::binary);

	uint32_t numParams = this->dataList.size();
	paramOfs.write((char*)&numParams, sizeof(uint32_t));

	for (int i = 0; i < this->dataList.size(); i++) {
		this->dataList[i]->save(paramOfs);
	}
	paramOfs.close();
}

template <typename Dtype>
Data<Dtype>* ParamManipulator<Dtype>::findParam(const string& paramName) {
	auto it = this->dataMap.find(paramName);
	if (it == this->dataMap.end()) {
		cout << "No such param name: " << paramName << endl;
		return 0;
	} else {
		return it->second;
	}
}



template class ParamManipulator<float>;










































