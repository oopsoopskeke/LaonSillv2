/**
 * @file GAN.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef GAN_H
#define GAN_H 

#include "Network.h"

template<typename Dtype>
class GAN {
public: 
    GAN() {}
    virtual ~GAN() {}

    static void run();
private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* GAN_H */
