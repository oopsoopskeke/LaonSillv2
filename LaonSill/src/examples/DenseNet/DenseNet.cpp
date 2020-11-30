/**
 * @file DenseNet.cpp
 * @date 2018-01-24
 * @author soung park
 * @brief
 * @details
 */

#include <vector>
#include <string>

#include "common.h"
#include "DenseNet.h"
#include "StdOutLog.h"
#include "Network.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

#define EXAMPLE_DENSENET_TRAIN_NETWORK_FILEPATH 	SPATH("examples/DenseNet/densenet_test.json")

template<typename Dtype>
void DenseNet<Dtype>::run() {
    string networkID = PlanParser::loadNetwork(string(EXAMPLE_DENSENET_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(0);

	network->run(false);
}


template class DenseNet<float>;
