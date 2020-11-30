/**
 * @file YOLOPassThruLayer.cpp
 * @date 2018-01-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "YOLOPassThruLayer.h"
#include "MemoryMgmt.h"
#include "PropMgmt.h"
#include "SysLog.h"

template <typename Dtype>
YOLOPassThruLayer<Dtype>::YOLOPassThruLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::YOLOPassThru;
}


template <typename Dtype>
YOLOPassThruLayer<Dtype>::~YOLOPassThruLayer() {
}

template class YOLOPassThruLayer<float>;
