/**
 * @file FRCNN.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>
#include <string>

#include "common.h"
#include "Debug.h"
#include "FRCNN.h"
#include "StdOutLog.h"
#include "Network.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

#define INFERENCE 1

template<typename Dtype>
void FRCNN<Dtype>::setLayerTrain(Network<Dtype>* network, bool train) {
    vector<Layer<Dtype>*> layers = network->findLayersByType((int)Layer<Dtype>::BatchNorm);

    for (int i = 0; i < layers.size(); i++) {
        BatchNormLayer<Dtype>* bnLayer = dynamic_cast<BatchNormLayer<Dtype>*>(layers[i]);
        SASSUME0(bnLayer != NULL);

        bnLayer->setTrain(train);
    }
}

#if !INFERENCE
#define EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH	SPATH("examples/frcnn/frcnn_train.json")
#else
#define EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH	SPATH("examples/frcnn/frcnn_test.json")
#endif

template<typename Dtype>
void FRCNN<Dtype>::run() {
    string networkID = PlanParser::loadNetwork(string(EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);

#if !INFERENCE
    network->build(0);
    network->run(false);
#else
    network->build(1);
    network->run(true);
#endif

}


template class FRCNN<float>;
