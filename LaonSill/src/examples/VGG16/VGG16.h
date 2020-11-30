/**
 * @file VGG16.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef VGG16_H
#define VGG16_H

#include "Network.h"

template<typename Dtype>
class VGG16 {
public: 
    VGG16() {}
    virtual ~VGG16() {}

    static void run();
private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* VGG16_H */
