/*
 * CudaUtils.h
 *
 *  Created on: Mar 10, 2017
 *      Author: jkim
 */

#ifndef CUDAUTILS_H_
#define CUDAUTILS_H_

#include "common.h"

template <typename Dtype>
void soooa_sub_channel_mean(const int N, const uint32_t singleChannelSize, const Dtype* mean,
		Dtype* data);

template <typename Dtype>
void soooa_add_channel_mean(const int N, const uint32_t singleChannelSize, const Dtype* mean,
		Dtype* data);


template <typename Dtype>
void soooa_bound_data(const int N, const uint32_t singleChannelSize, const Dtype* bound,
		Dtype* data);



#endif /* CUDAUTILS_H_ */
