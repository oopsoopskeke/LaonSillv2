/**
 * @file ZFNet.h
 * @date 2017-04-20
 * @author Jongha Lim
 * @brief 
 * @details
 */

#ifndef ZFNET_H
#define ZFNET_H

#include "Network.h"

template<typename Dtype>
class ZFNet {
public: 
    ZFNet() {}
    virtual ~ZFNet() {}

    static void run();
private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* ZFNET_H */
