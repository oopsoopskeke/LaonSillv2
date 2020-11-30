/**
 * @file NetworkRunByPlanTest.cpp
 * @date 2017-06-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string>

#include "NetworkRunByPlanTest.h"
#include "common.h"
#include "StdOutLog.h"
#include "LogicalPlan.h"
#include "Network.h"
#include "PlanParser.h"

using namespace std;

#define PLAN_PARSER_TEST_NETWORK_FILEPATH       SPATH("plan/test/network.conf.test")

bool NetworkRunByPlanTest::runPlanOnceTest() {
    string networkID = PlanParser::loadNetwork(string(PLAN_PARSER_TEST_NETWORK_FILEPATH));
    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->build(10);

    STDOUT_LOG(" [1st minibatch] run forward");
    network->runPlanType(PLANTYPE_FORWARD, false);

    STDOUT_LOG(" [1st minibatch] run backward");
    network->runPlanType(PLANTYPE_BACKWARD, false);

    STDOUT_LOG(" [1st minibatch] run update");
    network->runPlanType(PLANTYPE_UPDATE, false);

    STDOUT_LOG(" [2nd minibatch] run forward");
    network->runPlanType(PLANTYPE_FORWARD, false);

    STDOUT_LOG(" [2nd minibatch] run backward");
    network->runPlanType(PLANTYPE_BACKWARD, false);

    STDOUT_LOG(" [2nd minibatch] run update");
    network->runPlanType(PLANTYPE_UPDATE, false);

    return true;
}

bool NetworkRunByPlanTest::runTest() {
    bool result = runPlanOnceTest();
    if (result) {
        STDOUT_LOG("*  - run plan once test is success");
    } else {
        STDOUT_LOG("*  - run plan once test is failed");
        return false;
    }

    return true;
}
