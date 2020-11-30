/*
 * LayerTest.h
 *
 *  Created on: Feb 20, 2017
 *      Author: jkim
 */

#ifndef LAYERTEST_H_
#define LAYERTEST_H_

#include "LayerTestInterface.h"
#include "TestUtil.h"
#include "BaseLayer.h"
#include "SysLog.h"
#include "Network.h"
#include "StdOutLog.h"

using namespace std;



template <typename Dtype>
class LayerTest : public LayerTestInterface<Dtype> {
public:
	LayerTest(const std::string& networkFilePath, const std::string& networkName,
			const std::string& targetLayerName, const int numSteps = 1,
			const int numAfterSteps = 10, const NetworkStatus status = NetworkStatus::Train)
	: networkFilePath(networkFilePath), networkName(networkName),
	  targetLayerName(targetLayerName), numSteps(numSteps), numAfterSteps(numAfterSteps),
	  status(status) {
		SASSERT0(!this->networkFilePath.empty());
		this->withNetwork = true;
		if (this->networkName.empty()) {
			STDOUT_LOG("proceed without network name ...");
			this->withNetwork = false;
		}
		SASSERT0(!this->targetLayerName.empty());
		SASSERT0(this->numAfterSteps > 0);	// save network시 주석 처리해야 함.
		SASSERT0(this->numSteps > 0);
	}
	virtual void setUp() {
		this->networkID = PrepareContext<Dtype>(this->networkFilePath, 1);
		this->network = Network<Dtype>::getNetworkFromID(this->networkID);

		RetrieveLayers<float>(this->networkID, NULL, &this->layers, NULL, NULL,
				&this->learnableLayers);

		BuildNameLayerMap(this->networkID, this->layers, this->nameLayerMap);

		PrintLayerList(this->networkID, &this->layers, &this->learnableLayers);

		PrintLayerDataConfig(this->networkID, this->layers);

		LoadParams(this->networkName, this->numSteps, this->status,
				this->nameParamsMapList);
		PrintNameDataMapList("nameParamsMap", this->nameParamsMapList);

		LoadBlobs(this->networkName, this->numSteps, this->nameBlobsMapList);
		PrintNameDataMapList("nameBlobsMap", this->nameBlobsMapList);
	}

	virtual void cleanUp() {
		for (int i = 0; i <= this->numSteps; i++) {
			cleanUpMap(this->nameParamsMapList[i]);
		}
		for (int i = 0; i < this->numSteps; i++) {
			cleanUpMap(this->nameBlobsMapList[i]);
		}
	}

	virtual void forwardTest() {
		auto itr = this->nameLayerMap.find(this->targetLayerName);
		SASSERT(itr != this->nameLayerMap.end(), "[ERROR] INVALID LAYER: %s",
				this->targetLayerName.c_str());

		int layerID = itr->second.first;
		Layer<Dtype>* layer = itr->second.second;



		for (int i = 0; i < this->numSteps; i++) {
			if (!dynamic_cast<InputLayer<Dtype>*>(layer)) {
				FillDatum(this->networkID, this->nameBlobsMapList[i],
						itr->second, DataEndType::INPUT);
			}
			LearnableLayer<Dtype>* learnableLayer =
					dynamic_cast<LearnableLayer<Dtype>*>(layer);
			if (learnableLayer) {
				SASSERT(this->withNetwork,
						"there should be network, when target network is learnable");

				std::pair<int, LearnableLayer<Dtype>*> learnableLayerPair =
						std::make_pair(layerID, learnableLayer);
				FillParam(this->networkID, this->nameParamsMapList[i], learnableLayerPair);
			}


			//cout << "-----------------before feed forward ... " << endl;
			//printData(layer->_inputData, DataType::DATA);
			//printData(layer->_outputData, DataType::DATA);

			WorkContext::updateLayer(this->networkID, layerID);
			layer->feedforward();

			//cout << "-----------------after feed forward ... " << endl;
			//printData(layer->_outputData, DataType::DATA);

			CompareData(this->networkID, this->nameBlobsMapList[i], itr->second,
					DataEndType::OUTPUT);
		}
	}

	virtual void backwardTest() {
		auto itr = this->nameLayerMap.find(this->targetLayerName);
		SASSERT(itr != this->nameLayerMap.end(), "[ERROR] INVALID LAYER: %s",
				this->targetLayerName.c_str());

		int layerID = itr->second.first;
		Layer<Dtype>* layer = itr->second.second;

		FillDatum(this->networkID, this->nameBlobsMapList[0], itr->second, DataEndType::OUTPUT);
		LearnableLayer<Dtype>* learnableLayer = dynamic_cast<LearnableLayer<Dtype>*>(layer);
		if (learnableLayer) {
			std::pair<int, LearnableLayer<Dtype>*> learnableLayerPair =
					std::make_pair(layerID, learnableLayer);
			FillParam(this->networkID, this->nameParamsMapList[0], learnableLayerPair);
		}

		//cout << "-----------------before back propagation ... " << endl;
		//printData(layer->_inputData, DataType::GRAD);
		//printData(layer->_outputData, DataType::GRAD);

		layer->backpropagation();

		//cout << "-----------------after back propagation ... " << endl;
		//printData(layer->_inputData, DataType::GRAD);

		CompareData(this->networkID, this->nameBlobsMapList[0], itr->second, DataEndType::INPUT);
	}

public:
	Network<Dtype>* network;


private:
	const std::string networkFilePath;
	const std::string networkName;
	const std::string targetLayerName;
	const int numAfterSteps;
	const int numSteps;
	const NetworkStatus status;

    std::string networkID;
	bool withNetwork;


	std::vector<std::pair<int, Layer<Dtype>*>> layers;
	std::vector<std::pair<int, LearnableLayer<Dtype>*>> learnableLayers;

	std::vector<std::map<std::string, Data<Dtype>*>> nameParamsMapList;
	std::vector<std::map<std::string, Data<Dtype>*>> nameBlobsMapList;

	std::map<std::string, std::pair<int, Layer<Dtype>*>> nameLayerMap;
};



#endif /* LAYERTEST_H_ */
