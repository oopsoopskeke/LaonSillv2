/**
 * @file FRCNN.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef FRCNN_H
#define FRCNN_H

#include "Network.h"

template<typename Dtype>
class FRCNN {
public: 
    FRCNN() {}
    virtual ~FRCNN() {}

    static void run();
private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* FRCNN_H */
