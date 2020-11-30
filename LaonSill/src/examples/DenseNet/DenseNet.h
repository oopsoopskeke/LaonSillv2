/**
 * @file DenseNet.h
 * @date 2018-01-24
 * @author soung park
 * @brief
 * @details
 */

#ifndef DENSENET_H
#define DENSENET_H

#include "Network.h"

template<typename Dtype>
class DenseNet {
public:
	DenseNet() {}
    virtual ~DenseNet() {}

    static void run();
private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* DENSENET_H */
