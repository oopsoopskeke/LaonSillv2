/*
 * ClsOutfileLayer.cpp
 *
 *  Created on: Jan 11, 2018
 *      Author: jkim
 */

#include <boost/filesystem.hpp>

#include "ClsOutfileLayer.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "MemoryMgmt.h"
#include "PropMgmt.h"


using namespace std;

template <typename Dtype>
ClsOutfileLayer<Dtype>::ClsOutfileLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::ClsOutfile;

	SASSERT(SNPROP(status) == NetworkStatus::Test,
			"ClsOutfileLayer can be run only in Test Status");

	this->dataSetFile = SLPROP(ClsOutfile, dataSetFile);

	if (this->dataSetFile.length() == 0) {
		this->hasDataSetFile = false;
	} else {
		if (!boost::filesystem::exists(this->dataSetFile)) {
			STDOUT_LOG("[ERROR] File not exists: %s", this->dataSetFile.c_str());
			SASSERT(false, "File not exists: %s", this->dataSetFile.c_str());
		}
		this->hasDataSetFile = true;
	}

	this->outFilePath = SLPROP(ClsOutfile, outFilePath);
	this->scores.clear();
	//this->labels.clear();
}

template <typename Dtype>
ClsOutfileLayer<Dtype>::~ClsOutfileLayer() {
}

template <typename Dtype>
void ClsOutfileLayer<Dtype>::reshape() {

}

template <typename Dtype>
void ClsOutfileLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* softmax = this->_inputData[0]->host_data();
	//const Dtype* label = this->_inputData[1]->host_data();

	int batches = this->_inputData[0]->batches();
	for (int i = 0; i < batches; i++) {
		this->scores.push_back(softmax[this->_inputData[0]->offset(i) + 1]);
		//this->labels.push_back(label[i]);
	}

	if (SNPROP(iterations) >= SNPROP(miniBatch)) {
		// write file
		cout.precision(8);
		cout.setf(ios::fixed);

		vector<string> dataSets;
		if (this->hasDataSetFile) {
			ifstream infile(this->dataSetFile, ios_base::in);
			if (!infile.is_open())
				STDOUT_LOG("[ERROR] failed to open dataSetFile: %s", this->dataSetFile.c_str());
			SASSERT(infile.is_open(), "failed to open dataSetFile: %s",
					this->dataSetFile.c_str());
			string filename;
			string labelname;

			while (infile >> filename >> labelname) {
				dataSets.push_back(filename);
			}
		}

		ofstream outfile(this->outFilePath, ios_base::out);
		outfile.precision(8);
		outfile.setf(ios::fixed);

		//int correct = 0;
		if (!outfile.is_open())
			STDOUT_LOG("[ERROR] failed to open outFilePath: %s", this->outFilePath.c_str());
		SASSERT(outfile.is_open(), "failed to open outFilePath: %s",
				this->dataSetFile.c_str());

		for (int i = 0; i < this->scores.size(); i++) {
			if (this->hasDataSetFile) {
				outfile << dataSets[i % dataSets.size()] << " ";
			}
			outfile << scores[i] << endl;
			//cout << dataSets[i] << " " << this->scores[i] << " " << (int)this->labels[i] << endl;
			//if (this->scores[i] < 0.5f && this->labels[i] == Dtype(0) ||
			//		this->scores[i] >= 0.5f && this->labels[i] == Dtype(1)) {
			//	correct++;
			//}
		}
		//cout << "correct: " << correct << endl;
	}
}

template <typename Dtype>
void ClsOutfileLayer<Dtype>::backpropagation() {
	SASSERT(false, "backpropagation is not supported.");
}





/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* ClsOutfileLayer<Dtype>::initLayer() {
	ClsOutfileLayer* layer = NULL;
	SNEW(layer, ClsOutfileLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ClsOutfileLayer<Dtype>::destroyLayer(void* instancePtr) {
    ClsOutfileLayer<Dtype>* layer = (ClsOutfileLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ClsOutfileLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	SASSERT0(isInput);
	SASSERT0(index < 1);

    ClsOutfileLayer<Dtype>* layer = (ClsOutfileLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ClsOutfileLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ClsOutfileLayer<Dtype>* layer = (ClsOutfileLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ClsOutfileLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	ClsOutfileLayer<Dtype>* layer = (ClsOutfileLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void ClsOutfileLayer<Dtype>::backwardTensor(void* instancePtr) {
	ClsOutfileLayer<Dtype>* layer = (ClsOutfileLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void ClsOutfileLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool ClsOutfileLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    // 실제로 output shape를 필요로 하지 않음. 
    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t ClsOutfileLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class ClsOutfileLayer<float>;






