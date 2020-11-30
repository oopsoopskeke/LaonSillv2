/*
 * RoIPoolingLayer.cpp
 *
 *  Created on: Dec 1, 2016
 *      Author: jkim
 */

#include <cfloat>

#include "RoIPoolingLayer.h"
#include "Cuda.h"
#include "MathFunctions.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

#define ROIPOOLINGLAYER_LOG 0


template <typename Dtype>
RoIPoolingLayer<Dtype>::RoIPoolingLayer()
: Layer<Dtype>(),
  maxIdx("maxIdx") {
	this->type = Layer<Dtype>::RoIPooling;

	assert(SLPROP(RoIPooling, pooledW) > 0 &&
			"pooledW must be > 0");
	assert(SLPROP(RoIPooling, pooledH) > 0 &&
			"pooledH must be > 0");

}

template <typename Dtype>
RoIPoolingLayer<Dtype>::~RoIPoolingLayer() {}

template <typename Dtype>
void RoIPoolingLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const std::vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;

		// 수정: inputData[0]든, inputData[1]이든 변경시 outputData의 shape은 변경된다.
		//if (i == 0) {
			this->channels = this->_inputData[0]->channels();
			this->height = this->_inputData[0]->height();
			this->width = this->_inputData[0]->width();

			const std::vector<uint32_t> outputDataShape =
                { (uint32_t)this->_inputData[1]->height(), this->channels, SLPROP(RoIPooling, pooledH),
                    SLPROP(RoIPooling, pooledW) };

			this->_outputData[0]->reshape(outputDataShape);
			this->maxIdx.reshape(outputDataShape);

#if ROIPOOLINGLAYER_LOG
			printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
						this->name.c_str(),
						outputDataShape[0],
						outputDataShape[1],
						outputDataShape[2],
						outputDataShape[3]);
#endif
		//}
	}
}


template <typename Dtype>
__global__ void ROIPoolForward(
		const int nthreads,
		const Dtype* bottom_data,
		const Dtype spatial_scale,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const Dtype* bottom_rois,
		Dtype* top_data,
		int* argmax_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		// (n, c, ph, pw) is an element in the pooled output
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;

		bottom_rois += n * 5;
		int roi_batch_ind = bottom_rois[0];
		int roi_start_w = round(bottom_rois[1] * spatial_scale);
		int roi_start_h = round(bottom_rois[2] * spatial_scale);
		int roi_end_w = round(bottom_rois[3] * spatial_scale);
		int roi_end_h = round(bottom_rois[4] * spatial_scale);

		// Force malformed ROIs to be 1x1
		int roi_width = max(roi_end_w - roi_start_w + 1, 1);
		int roi_height = max(roi_end_h - roi_start_h + 1, 1);
		Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
		Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

		int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
		int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
		int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)	* bin_size_h));
		int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)	* bin_size_w));

		// Add roi offsets and clip to input boundaries
		hstart = min(max(hstart + roi_start_h, 0), height);
		hend = min(max(hend + roi_start_h, 0), height);
		wstart = min(max(wstart + roi_start_w, 0), width);
		wend = min(max(wend + roi_start_w, 0), width);
		bool is_empty = (hend <= hstart) || (wend <= wstart);

		// Define an empty pooling region to be zero
		Dtype maxval = is_empty ? 0 : -FLT_MAX;
		// If nothing is pooled, argmax = -1 causes nothing to be backprop'd
		int maxidx = -1;
		bottom_data += (roi_batch_ind * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				int bottom_index = h * width + w;
				if (bottom_data[bottom_index] > maxval) {
					maxval = bottom_data[bottom_index];
					maxidx = bottom_index;
				}
			}
		}
		top_data[index] = maxval;
		argmax_data[index] = maxidx;
	}
}

template <typename Dtype>
void RoIPoolingLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* inputData = this->_inputData[0]->device_data();
	const Dtype* inputRois = this->_inputData[1]->device_data();
	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	int* argmaxData = this->maxIdx.mutable_device_data();
	uint32_t count = this->_outputData[0]->getCount();

	ROIPoolForward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
	      count, inputData, SLPROP(RoIPooling, spatialScale), this->channels, this->height, this->width,
	      SLPROP(RoIPooling, pooledH), SLPROP(RoIPooling, pooledW), inputRois, outputData, argmaxData);
	CUDA_POST_KERNEL_CHECK;
}




template <typename Dtype>
__global__ void ROIPoolBackward(
		const int nthreads,
		const Dtype* top_diff,
		const int* argmax_data,
		const int num_rois,
		const Dtype spatial_scale,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		Dtype* bottom_diff,
		const Dtype* bottom_rois) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		// (n, c, h, w) coords in bottom data
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;

		Dtype gradient = 0;
		// Accumulate gradient over all ROIs that pooled this element
		for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
			const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
			int roi_batch_ind = offset_bottom_rois[0];
			// Skip if ROI's batch index doesn't match n
			if (n != roi_batch_ind) {
				continue;
			}

			int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
			int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
			int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
			int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

			// Skip if ROI doesn't include (h, w)
			const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
					h >= roi_start_h && h <= roi_end_h);
			if (!in_roi) {
				continue;
			}

			int offset = (roi_n * channels + c) * pooled_height * pooled_width;
			const Dtype* offset_top_diff = top_diff + offset;
			const int* offset_argmax_data = argmax_data + offset;
			// Compute feasible set of pooled units that could have pooled
			// this bottom unit

			// Force malformed ROIs to be 1x1
			int roi_width = max(roi_end_w - roi_start_w + 1, 1);
			int roi_height = max(roi_end_h - roi_start_h + 1, 1);

			Dtype bin_size_h = static_cast<Dtype>(roi_height)
			/ static_cast<Dtype>(pooled_height);
			Dtype bin_size_w = static_cast<Dtype>(roi_width)
			/ static_cast<Dtype>(pooled_width);

			int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
			int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
			int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
			int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

			phstart = min(max(phstart, 0), pooled_height);
			phend = min(max(phend, 0), pooled_height);
			pwstart = min(max(pwstart, 0), pooled_width);
			pwend = min(max(pwend, 0), pooled_width);

			for (int ph = phstart; ph < phend; ++ph) {
				for (int pw = pwstart; pw < pwend; ++pw) {
					if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
						gradient += offset_top_diff[ph * pooled_width + pw];
					}
				}
			}
		}
		bottom_diff[index] = gradient;
	}
}





template <typename Dtype>
void RoIPoolingLayer<Dtype>::backpropagation() {
	//if (!propDown)

	const Dtype* inputRois = this->_inputData[1]->device_data();
	const Dtype* outputGrad = this->_outputData[0]->device_grad();
	Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
	const int count = this->_inputData[0]->getCount();

	soooa_gpu_set(count, Dtype(0.), inputGrad);
	const int* argmaxData = this->maxIdx.device_data();

	ROIPoolBackward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
	      count, outputGrad, argmaxData, this->_outputData[0]->batches(),
	      SLPROP(RoIPooling, spatialScale), this->channels, this->height, this->width,
	      SLPROP(RoIPooling, pooledH), SLPROP(RoIPooling, pooledW), inputGrad, inputRois);
	CUDA_POST_KERNEL_CHECK;
}



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* RoIPoolingLayer<Dtype>::initLayer() {
	RoIPoolingLayer* layer = NULL;
	SNEW(layer, RoIPoolingLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void RoIPoolingLayer<Dtype>::destroyLayer(void* instancePtr) {
    RoIPoolingLayer<Dtype>* layer = (RoIPoolingLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void RoIPoolingLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 2);
	} else {
		SASSERT0(index < 1);
	}

    RoIPoolingLayer<Dtype>* layer = (RoIPoolingLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool RoIPoolingLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    RoIPoolingLayer<Dtype>* layer = (RoIPoolingLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void RoIPoolingLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	RoIPoolingLayer<Dtype>* layer = (RoIPoolingLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void RoIPoolingLayer<Dtype>::backwardTensor(void* instancePtr) {
	RoIPoolingLayer<Dtype>* layer = (RoIPoolingLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void RoIPoolingLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool RoIPoolingLayer<Dtype>::checkShape(std::vector<TensorShape> inputShape,
        std::vector<TensorShape> &outputShape) {

	if (inputShape.size() != 2)
        return false;

    TensorShape outputShape1;
    outputShape1.N = inputShape[1].H;
    outputShape1.C = inputShape[0].C;
    outputShape1.H = SLPROP(RoIPooling, pooledH);
    outputShape1.W = SLPROP(RoIPooling, pooledW);
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t RoIPoolingLayer<Dtype>::calcGPUSize(std::vector<TensorShape> inputShape) {
    const int count = inputShape[1].H * inputShape[0].C * 
        SLPROP(RoIPooling, pooledH) * SLPROP(RoIPooling, pooledW);

    size_t size = ALIGNUP(sizeof(Dtype) * count, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    return size;
}

template class RoIPoolingLayer<float>;
