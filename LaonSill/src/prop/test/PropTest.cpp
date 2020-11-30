/**
 * @file PropTest.cpp
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string>

#include "PropTest.h"
#include "common.h"
#include "PropMgmt.h"
#include "StdOutLog.h"
#include "BaseLayer.h"
#include "WorkContext.h"

using namespace std;

bool PropTest::runSimpleLayerPropTest() {
    // (1) register layer prop
    string networkID = "9d20ddf9-b90f-4a6b-a455-e6aa3dd28c53";
    int layerID = 3;

    _ConvPropLayer *convProp = new _ConvPropLayer();
    LayerProp* newProp = new LayerProp(networkID, layerID, (int)Layer<float>::Conv,
        (void*)convProp);
    PropMgmt::insertLayerProp(newProp);

    _NetworkProp *networkProp = new _NetworkProp();
    PropMgmt::insertNetworkProp(networkID, networkProp);

    // (2) set layer prop and run
    WorkContext::updateLayer(networkID, layerID);

    STDOUT_LOG("initial filter dim strides & pads value : %d, %d\n",
        SLPROP(Conv, filterDim).stride, SLPROP(Conv, filterDim).pad);
    SLPROP(Conv, filterDim).stride = 2;
    SLPROP(Conv, filterDim).pad = 1;
    STDOUT_LOG("changed filter dim strides & pads value : %d, %d\n",
        SLPROP(Conv, filterDim).stride, SLPROP(Conv, filterDim).pad);

    // (3) clean up layer prop
    PropMgmt::removeLayerProp(networkID);
    PropMgmt::removeNetworkProp(networkID);

    return true;
}

bool PropTest::runSimpleNetworkPropTest() {
    // (1) register network prop
    string networkID = "5e59b0fb-ca99-434f-bfc0-f7b00570745c";
    int layerID = 45;

    _ConvPropLayer *convProp = new _ConvPropLayer();

    LayerProp* newProp = new LayerProp(networkID, layerID, (int)Layer<float>::Conv,
        (void*)convProp);
    PropMgmt::insertLayerProp(newProp);

    _NetworkProp *networkProp = new _NetworkProp();
    PropMgmt::insertNetworkProp(networkID, networkProp);

    // (2) set network prop and run
    WorkContext::updateLayer(networkID, layerID);

    STDOUT_LOG("initial batchSize value : %u\n", SNPROP(batchSize));
    SNPROP(batchSize) = 128;
    STDOUT_LOG("changed batchSize value : %u\n", SNPROP(batchSize));

    // (3) clean up layer prop
    PropMgmt::removeLayerProp(networkID);
    PropMgmt::removeNetworkProp(networkID);

    return true;
}

bool PropTest::runTest() {
    bool result = runSimpleLayerPropTest();
    if (result) {
        STDOUT_LOG("*  - simple layer prop test is success");
    } else {
        STDOUT_LOG("*  - simple layer prop test is failed");
        return false;
    }

    result = runSimpleNetworkPropTest();
    if (result) {
        STDOUT_LOG("*  - simple network prop test is success");
    } else {
        STDOUT_LOG("*  - simple network prop test is failed");
        return false;
    }
    
    return true;
}
