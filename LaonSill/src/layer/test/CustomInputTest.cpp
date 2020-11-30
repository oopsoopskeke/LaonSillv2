/**
 * @file CustomInputTest.cpp
 * @date 2017-06-29
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>
#include <string>

#include "CustomInputTest.h"
#include "Network.h"
#include "PlanParser.h"
#include "PlanOptimizer.h"
#include "common.h"
#include "StdOutLog.h"
#include "SysLog.h"
#include "WorkContext.h"
#include "CustomInputLayer.h"
#include "DebugUtil.h"

using namespace std;

#define NETWORK_FILEPATH       SPATH("layer/test/network_custom.conf.test")

#define FIRST_DATA_ELEM_COUNT_PER_BATCH             5
#define SECOND_DATA_ELEM_COUNT_PER_BATCH            1
#define SECOND_DATA_ELEM_MAX_NUMBER                 10

static void CustomInputCBFunc(int miniBatchIdx, int batchSize, void *args,
    std::vector<float*> &data) {
    SASSUME0(data.size() == 2);
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < FIRST_DATA_ELEM_COUNT_PER_BATCH; j++) {
            int index = i * FIRST_DATA_ELEM_COUNT_PER_BATCH + j;
            data[0][index] = 1.0 * (float)i;
        }

        for (int j = 0; j < SECOND_DATA_ELEM_COUNT_PER_BATCH; j++) {
            data[1][i] = (float)(i % SECOND_DATA_ELEM_MAX_NUMBER);
        }
    }
}

bool CustomInputTest::runSimpleTest() {
    string networkID = PlanParser::loadNetwork(string(NETWORK_FILEPATH));
    WorkContext::updateNetwork(networkID);
    PlanOptimizer::buildPlans(networkID);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);

    CustomInputLayer<float>* customInputLayer =
        (CustomInputLayer<float>*)network->findLayer("input");
    customInputLayer->registerCBFunc(CustomInputCBFunc, NULL);

    PlanOptimizer::runPlan(networkID, false);

    DebugUtil<float>::printNetworkEdges(stdout, "custom input layer test", networkID, 0);  

    return true;
}

bool CustomInputTest::runTest() {
    bool result = runSimpleTest();
    if (result) {
        STDOUT_LOG("*  - simple custom input test is success");
    } else {
        STDOUT_LOG("*  - simple custom input test is failed");
        return false;
    }

    return true;
}
