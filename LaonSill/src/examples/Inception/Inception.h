/**
 * @file Inception.h
 * @date 2018-02-26
 * @author jongheon kim
 * @brief 
 * @details
 */

#ifndef INCEPTION_H
#define INCEPTION_H

#include "Network.h"

template<typename Dtype>
class Inception {
public: 
    Inception() {}
    virtual ~Inception() {}

    static void run();
private:
    static void setLayerTrain(Network<Dtype>* lc, bool train);
};

#endif /* INCEPTION_H */
