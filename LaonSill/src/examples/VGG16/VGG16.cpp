/**
 * @file VGG16.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>
#include <string>

#include "common.h"
#include "VGG16.h"
#include "StdOutLog.h"
#include "Network.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

#define EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH	SPATH("examples/VGG16/vgg16_train.json")

template<typename Dtype>
void VGG16<Dtype>::run() {
    string networkID = PlanParser::loadNetwork(string(EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(0);

	network->run(false);
}


template class VGG16<float>;
