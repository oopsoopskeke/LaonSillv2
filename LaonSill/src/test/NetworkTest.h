/*
 * NetworkTest.h
 *
 *  Created on: Feb 21, 2017
 *      Author: jkim
 */

#ifndef NETWORKTEST_H_
#define NETWORKTEST_H_

#include <set>

#include "NetworkTestInterface.h"
#include "TestUtil.h"
#include "Util.h"
#include "Network.h"
#include "Data.h"
#include "SysLog.h"
#include "PlanParser.h"
#include "WorkContext.h"
#include "PropMgmt.h"
#include "InputLayer.h"
#include "PlanOptimizer.h"
#include "SplitLayer.h"
#include "DummyInputLayer.h"
#include "PropMgmt.h"


#define COPY_INPUT		1

template <typename Dtype>
class NetworkTest : public NetworkTestInterface<Dtype> {
public:
	NetworkTest(const std::string& networkFilePath, const std::string& networkName,
			const int numSteps, const NetworkStatus status,
			const std::vector<std::string>* forwardGTLayers = NULL,
			const std::vector<std::string>* backwardPeekLayers = NULL)

	: networkFilePath(networkFilePath), networkName(networkName), numSteps(numSteps),
	  status(status), forwardGTLayers(forwardGTLayers) {
		SASSERT0(this->networkFilePath.empty() != true);
		SASSERT0(this->networkName.empty() != true);
		SASSERT0(this->numSteps > 0);	// save network시 주석 처리해야 함.
	}

	virtual ~NetworkTest() {}

	virtual void setUp() {
		this->networkID = PrepareContext<Dtype>(this->networkFilePath, this->numSteps);
		this->network = Network<Dtype>::getNetworkFromID(this->networkID);

		RetrieveLayers(this->networkID, &this->outerLayers, &this->layers, &this->inputLayer,
				&this->lossLayers, &this->learnableLayers);

		this->hasNormalInputLayer =
				(dynamic_cast<DummyInputLayer<Dtype>*>(this->inputLayer.second) != 0);
		this->hasWeight = !SNPROP(loadPath).empty();

        if ((SNPROP(status) == NetworkStatus::Test) && !SNPROP(loadPathForTest).empty()) {
            this->hasWeight = true;
        }

		cout << "hasNormalInputLayer=" << this->hasNormalInputLayer <<
				", hasWeight=" << this->hasWeight << endl;

		BuildNameLayerMap(this->networkID, this->layers, this->nameLayerMap);

		PrintLayerList(this->networkID, &this->layers, &this->learnableLayers);

		PrintLayerDataConfig(this->networkID, this->layers);

		LoadParams(this->networkName, this->numSteps, status, this->nameParamsMapList);

		LoadBlobs(this->networkName, this->numSteps, this->nameBlobsMapList);

		if (!this->hasWeight) {
			FillParams(this->networkID, this->learnableLayers, this->nameParamsMapList[0]);
		}
	}

	virtual void cleanUp() {
		const int _numSteps = status == NetworkStatus::Train ? numSteps + 1 : numSteps;

		cout << "_numSteps: " << _numSteps << endl;
		cout << "nameParamsMapList size: " << this->nameParamsMapList.size() << endl;
		cout << "nameBlobsMapList size: " << this->nameBlobsMapList.size() << endl;
		for (int i = 0; i < _numSteps; i++) {
			cleanUpMap(this->nameParamsMapList[i]);
		}
		for (int i = 0; i < _numSteps - 1; i++) {
			cleanUpMap(this->nameBlobsMapList[i]);
		}
	}

	virtual void updateTest() {
		/*
		Data<Dtype>::printConfig = 1;
		SyncMem<Dtype>::printConfig = 1;
		vector<map<string, Data<Dtype>*>> maps = {this->nameParamsMapList[0], this->nameBlobsMapList[0]};
		for (int i = 0; i < maps.size(); i++) {
			map<string, Data<Dtype>*> temp_map = maps[i];
			for (auto itr = temp_map.begin(); itr != temp_map.end(); itr++) {
				Data<Dtype>* value = itr->second;
				value->print_data({}, false);
			}
		}
		Data<Dtype>::printConfig = 0;
		SyncMem<Dtype>::printConfig = 0;
		exit(1);
		*/

		for (int i = 0; i < this->numSteps; i++) {
			cout << "::::::::::STEP " << i << "::::::::::" << endl;

			// feedforward
			logStartTest("FEED FORWARD");
			forward(i);
			dataTest(i);
			logEndTest("FEED FORWARD");

			// forwardGTLayers 주석 참고
			if (this->forwardGTLayers) {
				for (int j = 0; j < this->forwardGTLayers->size(); j++) {
					forwardSingleLayerWithGT(i, this->forwardGTLayers->at(j));
				}
			}

			// backpropagation
			replaceDataWithGroundTruth(i);
			logStartTest("BACK PROPAGATION");
			backward();
			gradTest(i);
			logEndTest("BACK PROPAGATION");

			// update & compare result
			// 오직 param grad에 대해서만 테스트하므로,
			// in/out의 grad를 업데이트하는 것은 의미가 없음
			//replaceGradWithGroundTruth(i);
			logStartTest("UPDATE");
			update();
			paramTest(i);
			logEndTest("UPDATE");
			//replaceParamWithGroundTruth(i+1, 0);
		}
	}


private:
	void forward(const int nthStep) {
#if COPY_INPUT
		feedInputLayerData(nthStep);
#endif
		this->network->runPlanType(PlanType::PLANTYPE_FORWARD, false);
	}

	void feedInputLayerData(const int nthStep) {
		SASSERT0(nthStep < this->numSteps);

		InputLayer<Dtype>* inputLayer = this->inputLayer.second;
		for (int i = 0; i < inputLayer->_outputData.size(); i++) {
			const string inputDataName = BLOBS_PREFIX + inputLayer->_outputData[i]->_name;
			Data<Dtype>* data = retrieveValueFromMap(this->nameBlobsMapList[nthStep], inputDataName);
			inputLayer->_outputData[i]->set_host_data(data, 0, true);

			/*
			// XXX: for frcnn only ...
			inputLayer->_outputData[i]->set_host_data(data, 0, true);
			if (i == 1) {
				const vector<uint32_t> shape = inputLayer->_outputData[1]->getShape();
				inputLayer->_outputData[1]->reshape({1, 1, 1, shape[1]});
				inputLayer->_outputData[1]->print_shape();
			} else if (i == 2) {
				const vector<uint32_t> shape = inputLayer->_outputData[2]->getShape();
				inputLayer->_outputData[2]->reshape({1, 1, shape[0], shape[1]});
				inputLayer->_outputData[2]->print_shape();
			}
			*/

		}
		//printDataList(inputLayer->_outputData, 0);
	}

	void dataTest(const int nthStep) {
		// split layer issue와 관련하여 ...
		// data test에서 split layer의 data들은 input과 output이 동일한 data를 share하므로
		// 굳이 따로 비교를 하지 않아도 될 것으로 보임.
		SASSERT0(nthStep < this->numSteps);
		for (int i = 0; i < this->layers.size(); i++) {
			int layerID = this->layers[i].first;
			Layer<Dtype>* layer = this->layers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			cout << "-----------------------------data test at layer " << layer->getName() << endl;
			if (!CompareData(this->networkID, this->nameBlobsMapList[nthStep],
					this->layers[i], DataEndType::OUTPUT)) {
				std::cout << "[ERROR] data feedforward failed at layer " << layer->getName() <<
						std::endl;
			} else {
				std::cout << "data feedforward succeed at layer " << layer->getName() << std::endl;
			}
		}
	}

	void backward() {
		this->network->runPlanType(PlanType::PLANTYPE_BACKWARD, false);
	}

	void gradTest(const int nthStep) {
		SASSERT0(nthStep < this->numSteps);
		// 1. caffe의 backward 과정에서 input layer와
		// input layer의 다음 레이어 input data에 대해 backward 진행하지 않기 때문에
		// 적용된 diff가 없으므로 해당 data에 대해서는 체크하지 않는다.
		// 2. npz에 param grad의 gt가 없어서 param에 대해서는 테스트 진행하지 않음
		for (int i = this->layers.size() - 1; i > 1; i--) {
			int layerID = this->layers[i].first;
			Layer<Dtype>* layer = this->layers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			// test blobs except input layer and second layer
			if (!CompareData(this->networkID, this->nameBlobsMapList[nthStep],
					this->layers[i], DataEndType::INPUT)) {
				std::cout << "[ERROR] data backpropagation failed at layer " << layer->getName()
						<< std::endl;
			} else {
				std::cout << "data backpropagation succeed at layer " << layer->getName() <<
						std::endl;
			}
		}
	}

	void update() {
		std::cout.precision(15);
		this->network->runPlanType(PlanType::PLANTYPE_UPDATE, false);
	}

	void paramTest(int nthStep) {
		SASSERT0(nthStep < this->numSteps);

		for (int i = 0; i < this->learnableLayers.size(); i++) {
			int layerID = this->learnableLayers[i].first;
			LearnableLayer<Dtype>* learnableLayer = this->learnableLayers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			// test final delta
			// params grad는 update 과정에서 오염됨.
			/*
			if (!compareParam(this->nameParamsNewMap,
					learnableLayer->name + SIG_PARAMS, learnableLayer->_params, 1)) {
				std::cout << "[ERROR] param backpropagation failed at layer " <<
						learnableLayer->name << std::endl;
				//exit(1);
			} else {
				std::cout << "param backpropagation succeed at layer " <<
						learnableLayer->name << std::endl;
			}
			*/

			// test final params
			bool result = false;
			result = CompareParam(this->nameParamsMapList[nthStep + 1],
					learnableLayer->getName() + SIG_PARAMS, learnableLayer->_params,
					DataType::DATA);
			//result = CompareParam(this->nameParamsMapList[nthStep + 1],
			//		learnableLayer->getName() + SIG_PARAMS, learnableLayer->_params,
			//		DataType::GRAD);

			if (!result) {
				std::cout << "[ERROR] update failed at layer " <<
						learnableLayer->getName() << std::endl;
			} else {
				std::cout << "update succeed at layer " << learnableLayer->getName() <<
						std::endl;
			}
		}
	}


	void replaceDataWithGroundTruth(int stepIdx) {

		/*
		for (int i = 0; i < this->layers.size(); i++) {
			int layerID = this->layers[i].first;
			Layer<Dtype>* layer = this->layers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			for (int j = 0; j < layer->_inputData.size(); j++) {
				const string dataName = BLOBS_PREFIX + layer->_inputData[j]->_name;
				Data<Dtype>* data =
						retrieveValueFromMap(this->nameBlobsMapList[stepIdx], dataName);
				layer->_inputData[j]->set_host_data(data, 0, false);
			}
		}

		// for loss layer output data ...
		for (int i = 0; i < this->lossLayers.size(); i++) {
			int layerID = this->lossLayers[i].first;
			LossLayer<Dtype>* lossLayer = this->lossLayers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			const std::string lossDataName = lossLayer->_outputData[0]->_name;
			const string dataName = BLOBS_PREFIX + lossDataName;
			Data<Dtype>* data =
					retrieveValueFromMap(this->nameBlobsMapList[stepIdx], dataName);

			lossLayer->_outputData[0]->set_host_data(data, 0, false);
			lossLayer->_outputData[0]->set_host_grad(data, 0, false);
		}
		*/

		// split data의 경우 index가 맞지 않아 ... grad 케이스에는 문제가 될 수 있네.
		FillData(this->networkID, this->layers, this->nameBlobsMapList[stepIdx],
				DataEndType::INPUT);

		std::pair<int, Layer<Dtype>*> layerPair =
				std::make_pair(this->lossLayers[0].first, this->lossLayers[0].second);
		FillDatum(this->networkID, this->nameBlobsMapList[stepIdx],
				layerPair, DataEndType::OUTPUT);
	}

	void logStartTest(const std::string& testName) {
		cout << "<<< " + testName + " TEST ... -------------------------------" << endl;
	}

	void logEndTest(const std::string& testName) {
		cout << ">>> " + testName + " TEST DONE ... --------------------------" << endl;
	}

	void printDataList(const std::vector<Data<Dtype>*>& dataList, int type = 0, int summary = 6) {
		Data<Dtype>::printConfig = 1;
		SyncMem<Dtype>::printConfig = 1;

		for (int j = 0; j < dataList.size(); j++) {
			printData(dataList[j], type, summary);
		}
		/*
		if (type == 0) {
			for (int j = 0; j < dataList.size(); j++) {
				dataList[j]->print_data({}, false, summary);
			}
		} else if (type == 1) {
			for (int j = 0; j < dataList.size(); j++) {
				dataList[j]->print_grad({}, false, summary);
			}
		}
		*/

		Data<Dtype>::printConfig = 0;
		SyncMem<Dtype>::printConfig = 0;
	}

	void printData(Data<Dtype>* data, int type = 0, int summary = 6) {
		Data<Dtype>::printConfig = 1;
		SyncMem<Dtype>::printConfig = 1;
		if (type == 0) {
			data->print_data({}, false, summary);

		} else if (type == 1) {
			data->print_grad({}, false, summary);
		}
		Data<Dtype>::printConfig = 0;
		SyncMem<Dtype>::printConfig = 0;
	}

	void printDataByName(const int stepIdx, const vector<string> targetDataNames, DataType dataType) {
		std::cout << "printDataByName()" << std::endl;
		for (int i = 0; i < targetDataNames.size(); i++) {
			const string& targetDataName = BLOBS_PREFIX + targetDataNames[i];
			std::cout << "targetDataName: " << targetDataName << std::endl;

			auto itr = this->nameBlobsMapList[stepIdx].find(targetDataName);
			if (itr != this->nameBlobsMapList[stepIdx].end()) {
				Data<Dtype>* data = itr->second;

				Data<Dtype>::printConfig = 1;
				SyncMem<Dtype>::printConfig = 1;
				switch(dataType) {
				case DATA: data->print_data({}, false); break;
				case GRAD: data->print_grad({}, false); break;
				}
				Data<Dtype>::printConfig = 0;
				SyncMem<Dtype>::printConfig = 0;
			}
		}
	}


	void forwardSingleLayerWithGT(const int stepIdx, const string& targetLayerName) {
		auto itr = this->nameLayerMap.find(targetLayerName);
		SASSERT(itr != this->nameLayerMap.end(), "[ERROR] INVALID LAYER: %s",
				targetLayerName.c_str());

		int layerID = itr->second.first;
		Layer<Dtype>* layer = itr->second.second;

		FillDatum(this->networkID, this->nameBlobsMapList[stepIdx], itr->second, DataEndType::INPUT);

		LearnableLayer<Dtype>* learnableLayer = dynamic_cast<LearnableLayer<Dtype>*>(layer);
		if (learnableLayer) {
			std::pair<int, LearnableLayer<Dtype>*> learnableLayerPair =
					std::make_pair(layerID, learnableLayer);
			FillParam(this->networkID, this->nameParamsMapList[stepIdx], learnableLayerPair);
		}
		layer->feedforward();
	}

	void backwardSingleLayerWithGT(const string& targetLayerName) {

	}


public:
	Network<Dtype>* network;


	const std::string networkFilePath;
	const std::string networkName;
	const int numSteps;
	const NetworkStatus status;

	// forward때 업데이트되는 내부 상태가 있는 레이어의 경우
	// forwardGTLayers에 등록하여 input/output data뿐 아니라
	// 내부 상태까지 정답 input을 통해 업데이트 하여야 한다.
	const std::vector<std::string>* forwardGTLayers;

	const std::vector<std::string>* backwardPeekLayers;

    std::string networkID;
	bool hasNormalInputLayer;
	bool hasWeight;


	vector<map<string, Data<Dtype>*>> nameParamsMapList;
	vector<map<string, Data<Dtype>*>> nameBlobsMapList;

	//const std::string params = "_params_";
	//const std::string blobs = "_blobs_";

	std::pair<int, InputLayer<Dtype>*> inputLayer;
	std::vector<std::pair<int, Layer<Dtype>*>> layers;
	std::vector<std::pair<int, LearnableLayer<Dtype>*>> learnableLayers;
	std::vector<std::pair<int, Layer<Dtype>*>> outerLayers;
	std::vector<std::pair<int, LossLayer<Dtype>*>> lossLayers;

	std::map<std::string, std::pair<int, Layer<Dtype>*>> nameLayerMap;
};



#endif /* NETWORKTEST_H_ */
