/**
 * @file YOLORegionLayer.cpp
 * @date 2018-01-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "YOLORegionLayer.h"
#include "MemoryMgmt.h"
#include "PropMgmt.h"
#include "SysLog.h"

template <typename Dtype>
YOLORegionLayer<Dtype>::YOLORegionLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::YOLORegion;

    SNEW(this->anchorSet, Data<Dtype>, SLPROP_BASE(name) + "_anchorSet");
    SASSUME0(this->anchorSet != NULL);
}


template <typename Dtype>
YOLORegionLayer<Dtype>::~YOLORegionLayer() {
    SFREE(this->anchorSet);
}

template class YOLORegionLayer<float>;
