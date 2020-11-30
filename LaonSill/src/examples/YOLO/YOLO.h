/**
 * @file YOLO.h
 * @date 2017-12-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef YOLO_H
#define YOLO_H 

#include "Network.h"

template<typename Dtype>
class YOLO {
public: 
    YOLO() {}
    virtual ~YOLO() {}
    static void run();

private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* YOLO_H */
