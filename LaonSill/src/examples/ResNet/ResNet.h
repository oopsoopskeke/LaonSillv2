/**
 * @file ResNet.h
 * @date 2018-02-26
 * @author jongheon kim
 * @brief 
 * @details
 */

#ifndef RESNET_H
#define RESNET_H

#include "Network.h"

template<typename Dtype>
class ResNet {
public: 
    ResNet() {}
    virtual ~ResNet() {}

    static void run();
private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* RESNET_H */
