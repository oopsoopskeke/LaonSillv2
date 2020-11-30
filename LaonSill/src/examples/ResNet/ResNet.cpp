/**
 * @file ResNet.cpp
 * @date 2018-02-26
 * @author jongheon kim 
 * @brief 
 * @details
 */

#include <vector>
#include <string>

#include "common.h"
#include "ResNet.h"
#include "StdOutLog.h"
#include "Network.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

//#define EXAMPLE_RESNET_TRAIN_NETWORK_FILEPATH	SPATH("examples/ResNet/resnet50_train.json")
#define EXAMPLE_RESNET_TRAIN_NETWORK_FILEPATH	SPATH("examples/ResNet/resnet50_union.json")

template<typename Dtype>
void ResNet<Dtype>::run() {
    string networkID = PlanParser::loadNetwork(string(EXAMPLE_RESNET_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(0);

	network->run(false);
}


template class ResNet<float>;
