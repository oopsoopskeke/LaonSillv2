/**
 * @file PlanBuilderTest.cpp
 * @date 2017-06-05
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PlanBuilderTest.h"
#include "PlanBuilder.h"
#include "common.h"
#include "StdOutLog.h"
#include "BaseLayer.h"
#include "LogicalPlan.h"

using namespace std;

bool PlanBuilderTest::runBuildTest() {
    PlanBuilder* pb = new PlanBuilder();

    pb->makeLayer(0, "input", (int)Layer<float>::ILSVRCInput, {}, {"data", "label"});
    pb->addLayerProp(0, "imageDir", "\"/data/ilsvrc12_train\"");
    pb->addLayerProp(0, "resizeImage", "true");
    pb->addLayerProp(0, "resizedImageRow", "224");
    pb->addLayerProp(0, "resizedImageCol", "224");

    pb->makeLayer(1, "conv1", (int)Layer<float>::Conv, {"data"}, {"conv1"});
    pb->addLayerProp(1, "filterDim.rows", "7");
    pb->addLayerProp(1, "filterDim.cols", "7");
    pb->addLayerProp(1, "filterDim.channels", "3");
    pb->addLayerProp(1, "filterDim.filters", "4");
    pb->addLayerProp(1, "filterDim.stride", "4");

    pb->makeLayer(2, "relu1", (int)Layer<float>::Relu, {"conv1"}, {"conv1"});

    pb->makeLayer(3, "fc1", (int)Layer<float>::FullyConnected, {"conv1"}, {"fc1"});
    pb->addLayerProp(3, "nOut", "1000");
    pb->addLayerProp(3, "weightFiller.type", "\"Gaussian\"");

    pb->makeLayer(6, "softmax", (int)Layer<float>::SoftmaxWithLoss, {"fc1", "label"},
        {"loss"});
    pb->addLayerProp(6, "innerLayer",
            "[ "
                "{ "
                    "\"name\" : \"softmaxInnerLayer\","
                    "\"id\" : 7000,"
                    "\"layer\" : \"Softmax\","
                    "\"input\" : [\"prob\"],"
                    "\"output\" : [\"softmaxInnerLayer1\" ] "
                "} "
            "]");

    pb->addNetworkProp("batchSize", "4");
    pb->addNetworkProp("epochs", "4");
    pb->addNetworkProp("lossLayer", "[\"loss\"]");
    pb->addNetworkProp("gamma", "0.1");
    pb->addNetworkProp("miniBatch", "50");

    string networkID = pb->load();

    LogicalPlan::printPlanDef(networkID);

    return true;
}

bool PlanBuilderTest::runTest() {
    bool result = runBuildTest();
    if (result) {
        STDOUT_LOG("*  - simple plan build test is success");
    } else {
        STDOUT_LOG("*  - simple plan build test is failed");
        return false;
    }

    return true;
}
