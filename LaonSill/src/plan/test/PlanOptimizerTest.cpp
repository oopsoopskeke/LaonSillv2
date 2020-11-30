/**
 * @file PlanOptimizerTest.cpp
 * @date 2017-05-18
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string>

#include "PlanOptimizerTest.h"
#include "PlanOptimizer.h"
#include "PlanParser.h"
#include "PropMgmt.h"
#include "common.h"
#include "StdOutLog.h"
#include "WorkContext.h"

using namespace std;

#define PLAN_PARSER_TEST_NETWORK_FILEPATH       SPATH("plan/test/network.conf.test")

bool PlanOptimizerTest::runSimpleTest() {
    string networkID = PlanParser::loadNetwork(string(PLAN_PARSER_TEST_NETWORK_FILEPATH));
    WorkContext::updateNetwork(networkID);
    PlanOptimizer::buildPlans(networkID);
    PlanOptimizer::runPlan(networkID, false);

    return true;
}

bool PlanOptimizerTest::runTest() {
    bool result = runSimpleTest();
    if (result) {
        STDOUT_LOG("*  - simple plan optimizer test is success");
    } else {
        STDOUT_LOG("*  - simple plan optimizer test is failed");
        return false;
    }

    return true;
}
