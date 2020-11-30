/*
 * InterpolationLayer.cpp
 *
 *  Created on: Aug 7, 2017
 *      Author: jkim
 */

#include "InterpolationLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "MathFunctions.h"
#include "MemoryMgmt.h"


using namespace std;

template <typename Dtype>
InterpolationLayer<Dtype>::InterpolationLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Interpolation;

	this->padBeg = SLPROP(Interpolation, padBeg);
	this->padEnd = SLPROP(Interpolation, padEnd);
	SASSERT(this->padBeg <= 0, "Only supports non-pos padding (cropping) for now.");
	SASSERT(this->padEnd <= 0, "Only supports non-pos padding (cropping) for now.");
}

template <typename Dtype>
InterpolationLayer<Dtype>::~InterpolationLayer() {

}

template <typename Dtype>
void InterpolationLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	this->batches = this->_inputData[0]->batches();
	this->channels = this->_inputData[0]->channels();
	this->heightIn = this->_inputData[0]->height();
	this->widthIn = this->_inputData[0]->width();
	this->heightInEff = this->heightIn + this->padBeg + this->padEnd;
	this->widthInEff = this->widthIn + this->padBeg + this->padEnd;

	if (SLPROP(Interpolation, shrinkFactor) != 1 &&
			SLPROP(Interpolation, zoomFactor) == 1) {
		const int shrinkFactor = SLPROP(Interpolation, shrinkFactor);
		SASSERT(shrinkFactor >= 1, "Shrink factor must be positive.");
		this->heightOut = (this->heightInEff - 1) / shrinkFactor + 1;
		this->widthOut = (this->widthInEff - 1) / shrinkFactor + 1;
	} else if (SLPROP(Interpolation, zoomFactor) != 1 &&
			SLPROP(Interpolation, shrinkFactor) == 1) {
		const int zoomFactor = SLPROP(Interpolation, zoomFactor);
		SASSERT(zoomFactor >= 1, "Zoom factor must be positive.");
		this->heightOut = this->heightInEff + (this->heightInEff - 1) * (zoomFactor - 1);
		this->widthOut = this->widthInEff + (this->widthInEff - 1) * (zoomFactor - 1);
	} else if (SLPROP(Interpolation, height) > 0 && SLPROP(Interpolation, width) > 0) {
		this->heightOut = SLPROP(Interpolation, height);
		this->widthOut = SLPROP(Interpolation, width);
	} else if (SLPROP(Interpolation, shrinkFactor) != 1 &&
			SLPROP(Interpolation, zoomFactor) != 1) {
		const int shrinkFactor = SLPROP(Interpolation, shrinkFactor);
		const int zoomFactor = SLPROP(Interpolation, zoomFactor);
		SASSERT(shrinkFactor >= 1, "Shrink factor must be positive.");
		SASSERT(zoomFactor >= 1, "Zoom factor must be positive.");
		this->heightOut = (this->heightInEff - 1) / shrinkFactor + 1;
		this->widthOut = (this->widthInEff - 1) / shrinkFactor + 1;
		this->heightOut = this->heightOut + (this->heightOut - 1) * (zoomFactor - 1);
		this->widthOut = this->widthOut + (this->widthOut - 1) * (zoomFactor - 1);
	} else {
		SASSERT(false, "Invalid Interpolation Layer Properies setting.");
	}
	SASSERT(this->heightInEff > 0, "height sould be positive.");
	SASSERT(this->widthInEff > 0, "width sould be positive.");
	SASSERT(this->heightOut > 0, "height sould be positive.");
	SASSERT(this->widthOut > 0, "width sould be positive.");
	this->_outputData[0]->reshape({(uint32_t)this->batches, (uint32_t)this->channels,
		(uint32_t)this->heightOut, (uint32_t)this->widthOut});
	this->_inputShape[0] = this->_inputData[0]->getShape();
}

template <typename Dtype>
void InterpolationLayer<Dtype>::feedforward() {
	reshape();
	soooa_gpu_interp2<Dtype, false>(this->batches * this->channels,
			this->_inputData[0]->device_data(), -this->padBeg, -this->padBeg,
			this->heightInEff, this->widthInEff, this->heightIn, this->widthIn,
			this->_outputData[0]->mutable_device_data(), 0, 0,
			this->heightOut, this->widthOut, this->heightOut, this->widthOut);
}

template <typename Dtype>
void InterpolationLayer<Dtype>::backpropagation() {
	if (!SLPROP_BASE(propDown)[0]) {
		return;
	}
	soooa_gpu_set(this->_inputData[0]->getCount(), Dtype(0),
			this->_inputData[0]->mutable_device_grad());
	soooa_gpu_interp2_backward<Dtype, false>(this->batches * this->channels,
			this->_inputData[0]->mutable_device_grad(), -this->padBeg, -this->padBeg,
			this->heightInEff, this->widthInEff, this->heightIn, this->widthIn,
			this->_outputData[0]->device_grad(), 0, 0,
			this->heightOut, this->widthOut, this->heightOut, this->widthOut);
}





/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* InterpolationLayer<Dtype>::initLayer() {
	InterpolationLayer* layer = NULL;
	SNEW(layer, InterpolationLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void InterpolationLayer<Dtype>::destroyLayer(void* instancePtr) {
    InterpolationLayer<Dtype>* layer = (InterpolationLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void InterpolationLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

	if (isInput) {
		SASSERT0(index < 1);
	} else {
		SASSERT0(index < 1);
	}

    InterpolationLayer<Dtype>* layer = (InterpolationLayer<Dtype>*)instancePtr;
    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool InterpolationLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    InterpolationLayer<Dtype>* layer = (InterpolationLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void InterpolationLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	InterpolationLayer<Dtype>* layer = (InterpolationLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void InterpolationLayer<Dtype>::backwardTensor(void* instancePtr) {
	InterpolationLayer<Dtype>* layer = (InterpolationLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void InterpolationLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool InterpolationLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

	const int padBeg = SLPROP(Interpolation, padBeg);
	const int padEnd = SLPROP(Interpolation, padEnd);

    if (padBeg > 0 || padEnd > 0) {
        return false;
    }

	const int batches = inputShape[0].N;
	const int channels = inputShape[0].C;
	const int heightIn = inputShape[0].H;
	const int widthIn = inputShape[0].W;
	const int heightInEff = heightIn + padBeg + padEnd;
	const int widthInEff = widthIn + padBeg + padEnd;
    int heightOut, widthOut;

	if (SLPROP(Interpolation, shrinkFactor) != 1 &&
			SLPROP(Interpolation, zoomFactor) == 1) {
		const int shrinkFactor = SLPROP(Interpolation, shrinkFactor);
        if (shrinkFactor < 1) {
		    //SASSERT(shrinkFactor >= 1, "Shrink factor must be positive.");
            return false;
        }
		heightOut = (heightInEff - 1) / shrinkFactor + 1;
		widthOut = (widthInEff - 1) / shrinkFactor + 1;
	} else if (SLPROP(Interpolation, zoomFactor) != 1 &&
			SLPROP(Interpolation, shrinkFactor) == 1) {
		const int zoomFactor = SLPROP(Interpolation, zoomFactor);
        if (zoomFactor < 1) {
		    //SASSERT(zoomFactor >= 1, "Zoom factor must be positive.");
            return false;
        }
		heightOut = heightInEff + (heightInEff - 1) * (zoomFactor - 1);
		widthOut = widthInEff + (widthInEff - 1) * (zoomFactor - 1);
	} else if (SLPROP(Interpolation, height) > 0 && SLPROP(Interpolation, width) > 0) {
		heightOut = SLPROP(Interpolation, height);
		widthOut = SLPROP(Interpolation, width);
	} else if (SLPROP(Interpolation, shrinkFactor) != 1 &&
			SLPROP(Interpolation, zoomFactor) != 1) {
		const int shrinkFactor = SLPROP(Interpolation, shrinkFactor);
		const int zoomFactor = SLPROP(Interpolation, zoomFactor);
        if (shrinkFactor < 1 || zoomFactor < 1) {
            //SASSERT(shrinkFactor >= 1, "Shrink factor must be positive.");
            //SASSERT(zoomFactor >= 1, "Zoom factor must be positive.");
            return false;
        }
		heightOut = (heightInEff - 1) / shrinkFactor + 1;
		widthOut = (widthInEff - 1) / shrinkFactor + 1;
		heightOut = heightOut + (heightOut - 1) * (zoomFactor - 1);
		widthOut = widthOut + (widthOut - 1) * (zoomFactor - 1);
	} else {
		//SASSERT(false, "Invalid Interpolation Layer Properies setting.");
        return false;
	}

    if (heightInEff <= 0 || widthInEff <=0 || heightOut <= 0 || widthOut <= 0) {
        //SASSERT(this->heightInEff > 0, "height sould be positive.");
        //SASSERT(this->widthInEff > 0, "width sould be positive.");
        //SASSERT(this->heightOut > 0, "height sould be positive.");
        //SASSERT(this->widthOut > 0, "width sould be positive.");
        return false;
    }

    TensorShape outputShape1;
    outputShape1.N = batches;
    outputShape1.C = channels;
    outputShape1.H = heightOut;
    outputShape1.W = widthOut;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t InterpolationLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class InterpolationLayer<float>;
