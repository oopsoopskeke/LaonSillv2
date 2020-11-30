/**
 * @file MeasureLayer.h
 * @date 2017-11-02
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef MEASURELAYER_H
#define MEASURELAYER_H 

#include <vector>

#include "common.h"
#include "BaseLayer.h"
#include "EnumDef.h"
#include "PropMgmt.h"

template<typename Dtype>
class MeasureLayer : public Layer<Dtype> {
public: 
    MeasureLayer() : Layer<Dtype>() {}
    virtual ~MeasureLayer() {}

	virtual void reshape() {
		Layer<Dtype>::reshape();
	}
	virtual void feedforward() {
		Layer<Dtype>::feedforward();
	}
	virtual void backpropagation() {
		Layer<Dtype>::backpropagation();
	}
	virtual Dtype measure() = 0;
    virtual Dtype measureAll() = 0;
};
#endif /* MEASURELAYER_H */
