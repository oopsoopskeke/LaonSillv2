/*
 * DataSet.cpp
 *
 *  Created on: 2016. 8. 16.
 *      Author: jhkim
 */

#include "DataSet.h"
#include "MemoryMgmt.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
DataSet<Dtype>::DataSet() {
	//mean[0] = 0;
	//mean[1] = 0;
	//mean[2] = 0;

	trainDataSet = NULL;
	trainLabelSet = NULL;
	trainSetIndices = NULL;

	validationDataSet = NULL;
	validationLabelSet = NULL;
	validationSetIndices = NULL;

	testDataSet = NULL;
	testLabelSet = NULL;
	testSetIndices = NULL;
}

template <typename Dtype>
DataSet<Dtype>::DataSet(uint32_t rows, uint32_t cols, uint32_t channels,
        uint32_t numTrainData, uint32_t numTestData) {
	this->rows = rows;
	this->cols = cols;
	this->channels = channels;
	this->dataSize = rows*cols*channels;
	this->numTrainData = numTrainData;
	this->numTestData = numTestData;

	this->trainDataSet = NULL;
	SNEW(this->trainDataSet, vector<Dtype>, this->dataSize * numTrainData);
	SASSUME0(this->trainDataSet != NULL);

	this->trainLabelSet = NULL;
	SNEW(this->trainLabelSet, vector<Dtype>, numTrainData);
	SASSUME0(this->trainLabelSet != NULL);

	this->trainSetIndices = NULL;
	SNEW(this->trainSetIndices, vector<uint32_t>, numTrainData);
	SASSUME0(this->trainSetIndices != NULL);
	iota(trainSetIndices->begin(), trainSetIndices->end(), 0);

	this->testDataSet = NULL;
	SNEW(this->testDataSet, vector<Dtype>, this->dataSize * numTestData);
	SASSUME0(this->testDataSet != NULL);

	this->testLabelSet = NULL;
	SNEW(this->testLabelSet, vector<Dtype>, numTestData);
	SASSUME0(this->testLabelSet != NULL);

	this->testSetIndices = NULL;
	SNEW(this->testSetIndices, vector<uint32_t>, numTestData);
	SASSUME0(this->testSetIndices != NULL);
	iota(testSetIndices->begin(), testSetIndices->end(), 0);

	validationDataSet = NULL;
	validationLabelSet = NULL;
	validationSetIndices = NULL;
}

template <typename Dtype>
DataSet<Dtype>::~DataSet() {
	if(trainDataSet) SDELETE(trainDataSet);
	if(trainLabelSet) SDELETE(trainLabelSet);
	if(trainSetIndices) SDELETE(trainSetIndices);

	if(validationDataSet) SDELETE(validationDataSet);
	if(validationLabelSet) SDELETE(validationLabelSet);
	if(validationSetIndices) SDELETE(validationSetIndices);

	if(testDataSet) SDELETE(testDataSet);
	if(testLabelSet) SDELETE(testLabelSet);
	if(testSetIndices) SDELETE(testSetIndices);
}

/*
template <typename Dtype>
void DataSet<Dtype>::setMean(const vector<Dtype>& means) {
	assert(means.size() == 1 || means.size() == 3);

	for(uint32_t i = 0; i < means.size(); i++) {
		this->mean[i] = means[i];
	}
}
*/

template <typename Dtype>
const Dtype* DataSet<Dtype>::getTrainDataAt(int index) {
	if(index >= numTrainData) {
		cout << "train data index over numTrainData ... " << endl;
		exit(1);
	}
	return &(*trainDataSet)[dataSize*(*trainSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getTrainLabelAt(int index) {
	if(index >= numTrainData) {
		cout << "train label index over numTrainData ... " << endl;
		exit(1);
	}
	return &(*trainLabelSet)[(*trainSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getValidationDataAt(int index) {
	if(index >= numValidationData) {
		cout << "validation data index over numValidationData ... " << endl;
		exit(1);
	}
	return &(*validationDataSet)[dataSize*(*validationSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getValidationLabelAt(int index) {
	if(index >= numValidationData) {
		cout << "validation label index over numValidationData ... " << endl;
		exit(1);
	}
	return &(*validationLabelSet)[(*validationSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getTestDataAt(int index) {
	if(index >= numTestData) {
		cout << "test data index over numTestData ... " << endl;
		exit(1);
	}
	return &(*testDataSet)[dataSize*(*testSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getTestLabelAt(int index) {
	if(index >= numTestData) {
		cout << "test label index over numTestData ... " << endl;
		exit(1);
	}
	return &(*testLabelSet)[(*testSetIndices)[index]];
}

/*
template <typename Dtype>
void DataSet<Dtype>::zeroMean(bool hasMean) {
	uint32_t di, ci, hi, wi;
	double sum[3] = {0.0, 0.0, 0.0};

	if(!hasMean) {
		for(di = 0; di < numTrainData; di++) {
			for(ci = 0; ci < channels; ci++) {
				for(hi = 0; hi < rows; hi++) {
					for(wi = 0; wi < cols; wi++) {
						sum[ci] += 
                            (*trainDataSet)[wi+hi*cols+ci*cols*rows+di*cols*rows*channels];
					}
				}
			}
		}

		for(ci = 0; ci < channels; ci++) {
			mean[ci] = (Dtype)(sum[ci] / (rows*cols*numTrainData));
		}
	}

	for(di = 0; di < numTrainData; di++) {
		for(ci = 0; ci < channels; ci++) {
			for(hi = 0; hi < rows; hi++) {
				for(wi = 0; wi < cols; wi++) {
					(*trainDataSet)[wi+hi*cols+ci*cols*rows+di*cols*rows*channels] -=
                        mean[ci];
				}
			}
		}
	}

	for(di = 0; di < numTestData; di++) {
		for(ci = 0; ci < channels; ci++) {
			for(hi = 0; hi < rows; hi++) {
				for(wi = 0; wi < cols; wi++) {
					(*testDataSet)[wi+hi*cols+ci*cols*rows+di*cols*rows*channels] -=
                        mean[ci];
				}
			}
		}
	}
}
*/

template <typename Dtype>
void DataSet<Dtype>::shuffleTrainDataSet() {
	random_shuffle(&(*trainSetIndices)[0], &(*trainSetIndices)[numTrainData]);
}

template <typename Dtype>
void DataSet<Dtype>::shuffleValidationDataSet() {
	random_shuffle(&(*validationSetIndices)[0], &(*validationSetIndices)[numValidationData]);
}

template <typename Dtype>
void DataSet<Dtype>::shuffleTestDataSet() {
	random_shuffle(&(*testSetIndices)[0], &(*testSetIndices)[numTestData]);
}

template class DataSet<float>;
