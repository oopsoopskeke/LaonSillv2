/**
 * @file LeNet.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef LENET_H
#define LENET_H

#include "Network.h"

template<typename Dtype>
class LeNet {
public: 
    LeNet() {}
    virtual ~LeNet() {}

    static void run();
private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* LENET_H */
