/*
 * Layer.cpp
 *
 *  Created on: 2016. 6. 10.
 *      Author: jhkim
 */

#include "BaseLayer.h"

#include <stddef.h>
#include <utility>

#include "Network.h"
#include "SysLog.h"
#include "PropMgmt.h"

using namespace std;

template<typename Dtype>
const string Layer<Dtype>::getName() {
    return SLPROP_BASE(name);
}

template<typename Dtype>
vector<string>& Layer<Dtype>::getInputs() { 
    return SLPROP_BASE(input);
}

template<typename Dtype>
vector<string>& Layer<Dtype>::getOutputs() {
    return SLPROP_BASE(output);
}

template<typename Dtype>
uint32_t Layer<Dtype>::getInputsSize() {
    return SLPROP_BASE(input).size();
}

template<typename Dtype>
uint32_t Layer<Dtype>::getOutputsSize() {
    return SLPROP_BASE(output).size();
}

template<typename Dtype>
int Layer<Dtype>::getId() {
   return SLPROP_BASE(id); 
}

template <typename Dtype>
Layer<Dtype>::Layer() {
}

template <typename Dtype>
Layer<Dtype>::~Layer() {
	// 다음 레이어들에 대해 소멸을 요청
	// 현재의 레이어가 요청하는 다음 레이어에 대해 마지막 이전 레이어인 경우,
	// 다음 레이어에 대해 소멸을 요청하게 된다.
    // (multi-branch인 경우 복수의 소멸 요청을 피하기 위해)
}

template <typename Dtype>
void Layer<Dtype>::reshape() {}

template <typename Dtype>
void Layer<Dtype>::feedforward() {
	this->_outputData[0]->set_device_data(this->_inputData[0]);
}

template <typename Dtype>
void Layer<Dtype>::backpropagation() {
	this->_inputData[0]->set_device_grad(this->_outputData[0]);
}

template <typename Dtype>
void Layer<Dtype>::printDataConfig() {
	cout << "for layer " << SLPROP_BASE(name) << endl;
	cout << "\tinput:" << endl;
	for (int i = 0; i < this->_inputData.size(); i++) {
		cout << "\t\t";
		this->_inputData[i]->print_shape();
	}
	cout << "\toutput:" << endl;
	for (int i = 0; i < this->_outputData.size(); i++) {
		cout << "\t\t";
		this->_outputData[i]->print_shape();
	}
}

template <typename Dtype>
bool Layer<Dtype>::_adjustInputShape() {
	const uint32_t inputSize = _inputData.size();

	// 입력 shape가 입력 데이터만큼 할당되지 않은 경우 해당 사이즈만큼 재할당
	if (_inputShape.size() != inputSize) {
		_inputShape.resize(inputSize);
		for (uint32_t i = 0; i < inputSize; i++) {
			_inputShape[i].resize(4);
		}
		return true;
	} else
		return false;
}


template <typename Dtype>
bool Layer<Dtype>::_isInputShapeChanged(uint32_t index) {
	assert(index < this->_inputData.size());
	assert(this->_inputData.size() == this->_inputShape.size());

	return (this->_inputData[index]->getCount() == 0 ||
			this->_inputData[index]->getShape() != this->_inputShape[index]);
}



template <typename Dtype>
void Layer<Dtype>::_printOn() {
	Data<Dtype>::printConfig = 1;
	SyncMem<Dtype>::printConfig = 1;
}

template <typename Dtype>
void Layer<Dtype>::_printOff() {
	Data<Dtype>::printConfig = 0;
	SyncMem<Dtype>::printConfig = 0;
}

template class Layer<float>;
