/**
 * @file GlobalPoolingLayer.cpp
 * @date 2017-12-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "GlobalPoolingLayer.h"
#include "PropMgmt.h"

using namespace std;

template <typename Dtype>
GlobalPoolingLayer<Dtype>::GlobalPoolingLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::GlobalPooling;
}


template <typename Dtype>
GlobalPoolingLayer<Dtype>::~GlobalPoolingLayer() {
}

template class GlobalPoolingLayer<float>;
