/**
 * @file SSD.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef SSD_H
#define SSD_H

#include "Network.h"

template<typename Dtype>
class SSD {
public: 
    SSD() {}
    virtual ~SSD() {}

    static void run();
private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* SSD_H */
