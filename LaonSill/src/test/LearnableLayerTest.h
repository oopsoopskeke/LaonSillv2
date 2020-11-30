/*
 * LearnableLayerTest.h
 *
 *  Created on: Feb 21, 2017
 *      Author: jkim
 */

#ifndef LEARNABLELAYERTEST_H_
#define LEARNABLELAYERTEST_H_

#include "LayerTestInterface.h"
#include "TestUtil.h"
#include "LearnableLayer.h"

using namespace std;


template <typename Dtype>
class LearnableLayerTest : public LayerTestInterface<Dtype> {
public:
#if 0
	LearnableLayerTest(typename LearnableLayer<Dtype>::Builder* builder,
			NetworkConfig<Dtype>* networkConfig = 0)
	: builder(builder), layer(0), networkConfig(networkConfig) {}

	virtual ~LearnableLayerTest() {
		cleanUpObject(this->layer);
		cleanUpObject(this->builder);
		cleanUpMap(this->nameDataMap);
	}

	virtual void setUp() {
		buildNameDataMapFromNpzFile(NPZ_PATH, this->builder->_name, this->nameDataMap);
		printNameDataMap("nameDataMap", this->nameDataMap, false);

		// 최소 설정만 전달받고 나머지는 npz로부터 추론하는 것이 좋겠다.
		this->layer = dynamic_cast<LearnableLayer<Dtype>*>(this->builder->build());
		assert(this->layer != 0);
		if (this->networkConfig != 0) {
			this->layer->setNetworkConfig(this->networkConfig);
		}

		fillLayerDataVec(this->layer->_inputs, this->layer->_inputData);
		fillLayerDataVec(this->layer->_outputs, this->layer->_outputData);
	}

	virtual void cleanUp() {}

	virtual void forwardTest() {
		fillData(this->nameDataMap, this->layer->name + SIG_BOTTOM, this->layer->_inputData);
		fillParam(this->nameDataMap, this->layer->name + SIG_PARAMS, this->layer->_params);
		for (uint32_t i = 0; i < this->layer->_params.size(); i++)
			this->layer->_paramsInitialized[i] = true;

		//printDataList(this->layer->_inputData, 0);
		//printDataList(this->layer->_params, 0);

		this->layer->feedforward();

		//printDataList(this->layer->_outputData, 0);

		compareData(this->nameDataMap, this->layer->name + SIG_TOP, this->layer->_outputData,
				0);
	}

	virtual void backwardTest() {
		fillData(this->nameDataMap, this->layer->name + SIG_BOTTOM, this->layer->_inputData);
		fillData(this->nameDataMap, this->layer->name + SIG_TOP, this->layer->_outputData);
		fillParam(this->nameDataMap, this->layer->name + SIG_PARAMS, this->layer->_params);

		this->layer->backpropagation();

		//printDataList(this->layer->_inputData, 1);

		compareData(this->nameDataMap, this->layer->name + SIG_BOTTOM,
				this->layer->_inputData, 1);
	}

	void printDataList(const std::vector<Data<Dtype>*>& dataList, int type = 0) {
		Data<Dtype>::printConfig = 1;
		SyncMem<Dtype>::printConfig = 1;

		if (type == 0) {
			for (int j = 0; j < dataList.size(); j++) {
				dataList[j]->print_data({}, false);
			}
		} else if (type == 1) {
			for (int j = 0; j < dataList.size(); j++) {
				dataList[j]->print_grad({}, false);
			}
		}
		Data<Dtype>::printConfig = 0;
		SyncMem<Dtype>::printConfig = 0;
	}


private:
	NetworkConfig<Dtype>* networkConfig;
	typename LearnableLayer<Dtype>::Builder* builder;
	LearnableLayer<Dtype>* layer;

	map<string, Data<Dtype>*> nameDataMap;
#endif
};


#endif /* LEARNABLELAYERTEST_H_ */
