/**
 * @file PlanOptimizer.cpp
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <time.h>

#include <algorithm>

#include "PhysicalPlan.h"
#include "PlanOptimizer.h"
#include "ResourceManager.h"
#include "common.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "PropMgmt.h"
#include "WorkContext.h"
#include "Worker.h"
#include "Network.h"
#include "ThreadMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

vector<int> PlanOptimizer::options;

void PlanOptimizer::init() {
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_SINGLE_GPU))
        options.push_back(PLAN_OPT_SINGLE_GPU);
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_MULTI_GPU))
        options.push_back(PLAN_OPT_MULTI_GPU);
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_MULTI_NODE))
        options.push_back(PLAN_OPT_MULTI_NODE);
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_VERTICAL_SPLIT))
        options.push_back(PLAN_OPT_VERTICAL_SPLIT);
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_HORIZONTAL_SPLIT))
        options.push_back(PLAN_OPT_HORIZONTAL_SPLIT);
}

bool PlanOptimizer::buildPlans(string networkID, int option, PlanOptPolicy policy) {
    vector<int> availableOptions;
    for (int i = 0; i < PlanOptimizer::options.size(); i++) {
        if (option & PlanOptimizer::options[i])
            availableOptions.push_back(PlanOptimizer::options[i]);
    }

    if (availableOptions.size() == 0)
        return false;

    if (availableOptions.size() == 1) {
        setPlanContext(networkID, availableOptions[0], false);
        return true;
    }
   
    if (policy == PLAN_OPT_POLICY_USE_FIRST_AVAILABLE_OPTION) {
        setPlanContext(networkID, availableOptions[0], false);
        return true;
    }

    if (policy == PLAN_OPT_POLICY_USE_LAST_AVAILABLE_OPTION) {
        setPlanContext(networkID, availableOptions[availableOptions.size() - 1], false);
        return true;
    }

    double bestElapsedTime;
    bool isFirst = true;
    int bestOption;

    for (int i = 0; i < availableOptions.size(); i++) {
        setPlanContext(networkID, availableOptions[i], true);
        double curElapsedTime = runPlan(networkID, true);

        if (isFirst) {
            bestOption = availableOptions[i];
            isFirst = false;
            bestElapsedTime = curElapsedTime;
        } else if (curElapsedTime < bestElapsedTime) {
            bestElapsedTime = curElapsedTime; 
            bestOption = availableOptions[i];
        }
        unsetPlanContext(networkID);
    }

    return true;
}

bool PlanOptimizer::buildPlans(string networkID) {
    return buildPlans(networkID, PLAN_OPT_DEFAULT, PLAN_OPT_POLICY_DEFAULT);
}

double PlanOptimizer::runPlanByType(string networkID, PlanType planType, bool inference) {
    struct timespec startTime, endTime;
    clock_gettime(CLOCK_REALTIME, &startTime);
    double elapsed = 0.0;
   
    WorkContext::updateNetwork(networkID);
    if (inference) {
        SNPROP(status) = NetworkStatus::Test;
        SNPROP(epochs) = 1;
    } else {
        SNPROP(status) = NetworkStatus::Train;
    }

    if ((WorkContext::curBootMode == DeveloperMode) ||
        (WorkContext::curBootMode == TestMode) ||
        (WorkContext::curBootMode == ResumeJobMode) ||
        (WorkContext::curBootMode == SingleJobMode)) {

        WorkContext::updatePlan(0, true);

        PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
        bool jobRemain = true;
        while (jobRemain) {
            bool canRunPlan = true;
            while (canRunPlan) {
                canRunPlan = pp->runPlan(planType, inference);
            }

            unique_lock<mutex> planLock(pp->planMutex);
            bool exitLoop = false;
            if (pp->planTypeRCMap[planType] == 0)
                exitLoop = true;
            planLock.unlock();

            if (exitLoop)
                break;
        }
        jobRemain = pp->generatePlan(true, false);

        clock_gettime(CLOCK_REALTIME, &endTime);
        elapsed = (endTime.tv_sec - startTime.tv_sec) +
            + (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;
    } else {
        SASSERT(false, "not implemented yet");
    }

    return elapsed;
}

void PlanOptimizer::runAdhocPlan(string networkID, string inputLayerName, int channel, 
        int height, int width, float* imageData) {
    WorkContext::updateNetwork(networkID);

    for (int i = 0; i < WorkContext::curPlanInfo->dopCount; i++) {
        int consumerIdx = i;        // XXX: 멀티 노드 환경에서는 더 고려해야 한다.
        WorkContext::updatePlan(i, true);
        Worker::addRunPlanTask(i, networkID, i, true, WorkContext::curThreadID,
                true, inputLayerName, channel, height, width, imageData);

        ThreadMgmt::signal(ThreadMgmt::getThreadID(ThreadType::TaskConsumer, i),
                ThreadEvent::Wakeup);
    }
}

double PlanOptimizer::runPlan(string networkID, bool inference) {
    struct timespec startTime, endTime;
    double elapsed = 0.0;

    WorkContext::updateNetwork(networkID);
    if (inference) {
        SNPROP(status) = NetworkStatus::Test;
        SNPROP(epochs) = 1;
    } else {
        SNPROP(status) = NetworkStatus::Train;
    }


    if ((WorkContext::curBootMode == DeveloperMode) ||
        (WorkContext::curBootMode == TestMode) ||
        (WorkContext::curBootMode == ResumeJobMode) ||
        (WorkContext::curBootMode == SingleJobMode)) {

        WorkContext::updatePlan(0, true);

        PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
        clock_gettime(CLOCK_REALTIME, &startTime);
        bool jobRemain = true;
        while (jobRemain) {
            bool canRunPlan = true;
            while (canRunPlan) {
                canRunPlan = pp->runPlan(inference);
            }
            jobRemain = pp->generatePlan(true, false);
        }

        clock_gettime(CLOCK_REALTIME, &endTime);
        elapsed = (endTime.tv_sec - startTime.tv_sec) +
            + (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;
    } else {
        for (int i = 0; i < WorkContext::curPlanInfo->dopCount; i++) {
            int consumerIdx = i;        // XXX: 멀티 노드 환경에서는 더 고려해야 한다.
            WorkContext::updatePlan(i, true);
            Worker::addRunPlanTask(i, networkID, i, inference, WorkContext::curThreadID,
                    false, "", -1, -1, -1, NULL);

            ThreadMgmt::signal(ThreadMgmt::getThreadID(ThreadType::TaskConsumer, i),
                    ThreadEvent::Wakeup);
        }
    }

    return elapsed;
}

void PlanOptimizer::setSingleGPUPlanContext(string networkID, bool isTest) {
    // (1) make physical plan list
    vector<PhysicalPlan*> ppList;
    GPUDevInfo devInfo = ResourceManager::getSingleGPUInfo();

    LogicalPlan* lp = LogicalPlan::getLogicalPlan(networkID);
    PhysicalPlan* pp = NULL;
    SNEW(pp, PhysicalPlan, SNPROP(lossLayer));
    SASSUME0(pp != NULL);

    pp->networkID = networkID;
    pp->refCount = 0;
    for (int i = 0; i < PLANTYPE_MAX; i++) {
        pp->planTypeRCMap[(PlanType)i] = 0;
    }

    for (int i = 0; i < lp->ppDefs.size(); i++) {
        PlanDef planDef = lp->ppDefs[i];
        PlanAlloc planAlloc;
        planAlloc.nodeID = devInfo.nodeID;
        planAlloc.devID = devInfo.devID;
       
        if (pp->allocMap.find(planDef.layerID) == pp->allocMap.end()) {
            pp->allocMap[planDef.layerID] = planAlloc;
        }

        SASSERT0(pp->planMap.find(planDef.planID) == pp->planMap.end());
        pp->planMap[planDef.planID] = planDef;

        SASSERT0(pp->depRefMap.find(planDef.planID) == pp->depRefMap.end());
        pp->depRefMap[planDef.planID] = planDef.depCount;

        if (planDef.depCount == 0) {
            pp->readyQueue.push_back(planDef.planID);
        }

        pp->refCount += 1;
        SASSUME0(planDef.planType < PlanType::PLANTYPE_MAX);
        pp->planTypeRCMap[planDef.planType] += 1;
    }

    pp->dopID = 0;

    ppList.push_back(pp);

    // (2) make PlanInfo
    PlanInfo *planInfo = NULL;
    SNEW(planInfo, PlanInfo);
    SASSUME0(planInfo != NULL);

    planInfo->networkID = networkID;
    planInfo->dopCount = 1;
    planInfo->doneCount = 0;

    planInfo->epochCount = SNPROP(epochs);
    planInfo->miniBatchCount = SNPROP(miniBatch);

    if (isTest) {
        planInfo->epochCount =
            min((int)planInfo->epochCount, (int)SPARAM(PLAN_OPT_TEST_MAX_EPOCH_COUNT));
        planInfo->miniBatchCount =
            min((int)planInfo->miniBatchCount, (int)SPARAM(PLAN_OPT_TEST_MAX_MINIBATCH_COUNT));
    }

    planInfo->curEpochIndex = 0;
    planInfo->curMiniBatchIndex = 0;

    // (3) insert plan
    PhysicalPlan::insertPlan(networkID, ppList, planInfo);

    // (4) set context
    WorkContext::updatePlan(0, true);
}

void PlanOptimizer::setMultiGPUPlanContext(string networkID, bool isTest) { 
    SASSERT0(false);
}

void PlanOptimizer::setMultiNodePlanContext(string networkID, bool isTest) { 
    SASSERT0(false);
}

void PlanOptimizer::setVerticalSplitPlanContext(string networkID, bool isTest) { 
    SASSERT0(false);
}

void PlanOptimizer::setHorizontalSplitPlanContext(string networkID, bool isTest) { 
    SASSERT0(false);
}

void PlanOptimizer::setPlanContext(string networkID, int option, bool isTest) {
    PlanInfo planInfoMap;
    PhysicalPlan* physicalPlan;

    switch (option) {
        case PLAN_OPT_SINGLE_GPU:
            setSingleGPUPlanContext(networkID, isTest);
            break;

        case PLAN_OPT_MULTI_GPU:
            setMultiGPUPlanContext(networkID, isTest);
            break;

        case PLAN_OPT_MULTI_NODE:
            setMultiNodePlanContext(networkID, isTest);
            break;

        case PLAN_OPT_VERTICAL_SPLIT:
            setVerticalSplitPlanContext(networkID, isTest);
            break;

        case PLAN_OPT_HORIZONTAL_SPLIT:
            setHorizontalSplitPlanContext(networkID, isTest);
            break;

        default:
            SASSERT(false, "invalid plan option. option=%d", option);
            break;
    }

    PhysicalPlan::allocateTensor(networkID);
    PhysicalPlan::loadNetwork();

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->setBuilt();

    if (SNPROP(startIterNum) > 0) {
        SNPROP(iterations) = SNPROP(startIterNum);
      
        PhysicalPlan::setCurProgress(networkID, SNPROP(startIterNum));

        SNPROP(currentStep) = 0;
        for (int i = 0; i < SNPROP(stepValue).size(); i++) {
            if (SNPROP(startIterNum) >= SNPROP(stepValue)[i]) {
                SNPROP(currentStep) = SNPROP(currentStep) + 1;
            } else {
                break;
            }
        }
    } else {
        SNPROP(iterations) = 0;
    }
}

void PlanOptimizer::unsetPlanContext(string networkID) {
    PhysicalPlan::removePlan(networkID);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->unsetBuilt();
}
