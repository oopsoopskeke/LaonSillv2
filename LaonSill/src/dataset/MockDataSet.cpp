/*
 * MockDataSet.cpp
 *
 *  Created on: 2016. 4. 23.
 *      Author: jhkim
 */

#include <random>

#include "MockDataSet.h"
#include "Util.h"

using namespace std;

template <typename Dtype>
MockDataSet<Dtype>::MockDataSet(uint32_t rows, uint32_t cols, uint32_t channels,
    uint32_t numTrainData, uint32_t numTestData, uint32_t numLabels, uint32_t mode)
	: DataSet<Dtype>(rows, cols, channels, numTrainData, numTestData), 
      numLabels(numLabels), mode(mode) {}

template <typename Dtype>
MockDataSet<Dtype>::~MockDataSet() {}

template <typename Dtype>
void MockDataSet<Dtype>::load() {
	switch(mode) {
	case FULL_RANDOM:
		_fillFullRandom();
		break;
	case NOTABLE_IMAGE:
		_fillNotableImage();
		break;
	}
}


template <typename Dtype>
void MockDataSet<Dtype>::_fillFullRandom() {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> ud(-0.1, 0.1);

	uint32_t i;
	// load train data
	for(i = 0; i < this->dataSize*this->numTrainData; i++) {
		(*this->trainDataSet)[i] = static_cast<Dtype>(ud(gen));
	}
	for(i = 0; i < this->numTrainData; i++) {
		uint32_t label = static_cast<uint32_t>((ud(gen)+0.1)*numLabels*5);
		(*this->trainLabelSet)[i] = static_cast<uint32_t>(label);
	}

	// load test data
	for(i = 0; i < this->dataSize*this->numTestData; i++) {
		(*this->testDataSet)[i] = static_cast<Dtype>(ud(gen));
	}
	for(i = 0; i < this->numTestData; i++) {
		uint32_t label = static_cast<uint32_t>((ud(gen)+0.1)*numLabels*5);
		(*this->testLabelSet)[i] = static_cast<uint32_t>(label);
	}
}

template <typename Dtype>
void MockDataSet<Dtype>::_fillNotableImage() {
	for(uint32_t i = 0; i < this->numTrainData; i++) {
		for(uint32_t j = 0; j < this->dataSize; j++) {
			(*this->trainDataSet)[this->dataSize*i+j] = i;
		}
	}
	for(uint32_t i = 0; i < this->numTrainData; i++) {
		(*this->trainLabelSet)[i] = i%numLabels;
	}

	for(uint32_t i = 0; i < this->numTestData; i++) {
		for(uint32_t j = 0; j < this->dataSize; j++) {
			(*this->testDataSet)[this->dataSize*i+j] = i;
		}
	}
	for(uint32_t i = 0; i < this->numTestData; i++) {
		(*this->testLabelSet)[i] = i%numLabels;
	}
}

template class MockDataSet<float>;
