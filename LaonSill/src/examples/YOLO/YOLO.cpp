/**
 * @file YOLO.cpp
 * @date 2017-12-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>
#include <string>

#include "common.h"
#include "Debug.h"
#include "YOLO.h"
#include "StdOutLog.h"
#include "Network.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"
#include "DebugUtil.h"

using namespace std;

//#define YOLO_PRETRAIN     1
//#define YOLO_PRETRAIN2    1
//#define YOLO_TRAIN        1
//#define YOLO_INFERENCE    1

//#if YOLO_PRETRAIN
//#define EXAMPLE_YOLO_NETWORK_FILEPATH	    SPATH("examples/YOLO/yolo_pretrain.json")
//#elif YOLO_PRETRAIN2
//#define EXAMPLE_YOLO_NETWORK_FILEPATH	    SPATH("examples/YOLO/yolo_pretrain2.json")
//#elif YOLO_TRAIN
//#define EXAMPLE_YOLO_NETWORK_FILEPATH	    SPATH("examples/YOLO/yolo_train.json")
//#else
//#define EXAMPLE_YOLO_NETWORK_FILEPATH	    SPATH("examples/YOLO/yolo_test_live.json")
//#endif
#define EXAMPLE_YOLO_NETWORK_FILEPATH	    SPATH("examples/YOLO/yolo_union.json")

template<typename Dtype>
void YOLO<Dtype>::run() {
    string networkID = PlanParser::loadNetwork(string(EXAMPLE_YOLO_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);

    network->build(0);
    //network->run(false);
    network->run(true);

    LogicalPlan::cleanup(networkID);
    PhysicalPlan::removePlan(networkID);
    PropMgmt::removeNetworkProp(networkID);
    PropMgmt::removeLayerProp(networkID);
    SDELETE(network);
}

template class YOLO<float>;
