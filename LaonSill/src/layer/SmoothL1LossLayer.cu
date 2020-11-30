/*
 * SmoothL1LossLayer.cpp
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#include <vector>

#include "SmoothL1LossLayer.h"
#include "MathFunctions.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "MemoryMgmt.h"

#define SMOOTHL1LOSSLAYER_LOG 0

using namespace std;

template <typename Dtype>
int SmoothL1LossLayer<Dtype>::INNER_ID = 11010;


template <typename Dtype>
SmoothL1LossLayer<Dtype>::SmoothL1LossLayer()
: SmoothL1LossLayer(NULL) {}

template <typename Dtype>
SmoothL1LossLayer<Dtype>::SmoothL1LossLayer(_SmoothL1LossPropLayer* prop)
: LossLayer<Dtype>(),
  diff("diff"),
  errors("errors"),
  ones("ones") {
	this->type = Layer<Dtype>::SmoothL1Loss;
	if (prop) {
		this->prop = NULL;
		SNEW(this->prop, _SmoothL1LossPropLayer);
		SASSUME0(this->prop != NULL);
		*(this->prop) = *(prop);
	} else {
		this->prop = NULL;
	}
}

template <typename Dtype>
SmoothL1LossLayer<Dtype>::~SmoothL1LossLayer() {
    if (this->prop != NULL)
        SFREE(this->prop);
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		this->hasWeights = (this->_inputData.size() >= 3);
		if (this->hasWeights) {
			SASSERT(this->_inputData.size() == 4,
					"If weights are used, must specify both inside and outside weights");
		}

		this->_outputData[0]->reshape({1, 1, 1, 1});
		this->_outputData[0]->mutable_host_grad()[0] = GET_PROP(prop, Loss, lossWeight);
#if SMOOTHL1LOSSLAYER_LOG
		printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
				GET_PROP(prop, Loss, name).c_str(), 1, 1, 1, 1);
#endif
	}

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;

		// rpn_bbox_pred
		if (i == 0) {
			this->diff.reshape(inputDataShape);
			this->errors.reshape(inputDataShape);
			// vector of ones used to sum
			this->ones.reshape(inputDataShape);
			this->ones.reset_host_data(false, 1.0f);

		}
		// rpn_bbox_targets
		else if (i == 1) {
			// XXX: FullyConnectedLayer의 output이 (batches, 1, rows, 1)의 현 구조를 반영,
			// 강제로 bbox_targets의 shape를 조정
			if (this->_inputData[0]->getShape() != this->_inputData[1]->getShape()) {
				this->_inputData[1]->reshape({this->_inputData[1]->getShape(2), 1,
					this->_inputData[1]->getShape(3), 1});
				assert(this->_inputData[0]->getShape() == this->_inputData[1]->getShape());
			}
			//assert(this->_inputData[0]->channels() == this->_inputData[1]->channels());
			//assert(this->_inputData[0]->height() == this->_inputData[1]->height());
			//assert(this->_inputData[0]->width() == this->_inputData[1]->width());
		}
		// rpn_bbox_inside_weights
		else if (i == 2) {
			if (this->hasWeights) {
				if (this->_inputData[0]->getShape() != this->_inputData[2]->getShape()) {
					this->_inputData[2]->reshape({this->_inputData[2]->getShape(2), 1,
						this->_inputData[2]->getShape(3), 1});
					assert(this->_inputData[0]->getShape() ==
							this->_inputData[2]->getShape());
				}
				//assert(this->_inputData[0]->channels() == this->_inputData[2]->channels());
				//assert(this->_inputData[0]->height() == this->_inputData[2]->height());
				//assert(this->_inputData[0]->width() == this->_inputData[2]->width());
			}
		}
		// rpn_bbox_outside_weights
		else if (i == 3) {
			if (this->hasWeights) {
				if (this->_inputData[0]->getShape() != this->_inputData[3]->getShape()) {
					this->_inputData[3]->reshape({this->_inputData[3]->getShape(2), 1,
						this->_inputData[3]->getShape(3), 1});
					assert(this->_inputData[0]->getShape() ==
							this->_inputData[3]->getShape());
				}
				//assert(this->_inputData[0]->channels() == this->_inputData[3]->channels());
				//assert(this->_inputData[0]->height() == this->_inputData[3]->height());
				//assert(this->_inputData[0]->width() == this->_inputData[3]->width());
			}
		}
	}
}


template <typename Dtype>
__global__ void SmoothL1Forward(const uint32_t n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f(x) = 0.5 * (sigma2 * x)^2          if |x| < 1 / sigma2 / sigma2
  //        |x| - 0.5 / sigma2 / sigma2    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = 0.5 * val * val * sigma2;
    } else {
      out[index] = abs_val - 0.5 / sigma2;
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::feedforward() {
	reshape();

	const uint32_t count = this->_inputData[0]->getCount();
	// prediction (inputData[0]) - target (inputData[1]) => diff
	soooa_gpu_sub(
			count,
			this->_inputData[0]->device_data(),
			this->_inputData[1]->device_data(),
			this->diff.mutable_device_data());		// d := b0 - b1

#if SMOOTHL1LOSSLAYER_LOG
	this->_printOn();
	this->_inputData[0]->print_data();
	this->_inputData[1]->print_data();
	this->diff.print_data();
	this->_printOff();
#endif

	if (this->hasWeights) {

#if SMOOTHL1LOSSLAYER_LOG
	this->_printOn();
	this->_inputData[2]->print_data();
	this->diff.print_data();
	this->_printOff();
#endif
		// apply "inside" weights
		soooa_gpu_mul(
				count,
				this->_inputData[2]->device_data(),
				this->diff.device_data(),
				this->diff.mutable_device_data());	// d := w_in * (b0 - b1)

#if SMOOTHL1LOSSLAYER_LOG
	this->_printOn();
	this->diff.print_data();
	this->_printOff();
#endif

	}

	// smoothL1Forward
	const float sigma2 = GET_PROP(prop, SmoothL1Loss, sigma) * GET_PROP(prop, SmoothL1Loss, sigma);
	SmoothL1Forward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
	      count, this->diff.device_data(), this->errors.mutable_device_data(), sigma2);
	CUDA_POST_KERNEL_CHECK;

#if SMOOTHL1LOSSLAYER_LOG
	this->_printOn();
	this->diff.print_data();
	this->errors.print_data();
	this->_printOff();
#endif

	if (this->hasWeights) {

#if SMOOTHL1LOSSLAYER_LOG
	this->_printOn();
	this->_inputData[3]->print_data();
	this->errors.print_data();
	this->_printOff();
#endif

		// apply "outside" weights
		soooa_gpu_mul(
				count,
				this->_inputData[3]->device_data(),
				this->errors.device_data(),
				this->errors.mutable_device_data());	// d := w_out * SmoothL1(w_in * (b0 - b1))

#if SMOOTHL1LOSSLAYER_LOG
	this->_printOn();
	this->errors.print_data();
	this->_printOff();
#endif
	}

	const uint32_t firstAxis = GET_PROP(prop, SmoothL1Loss, firstAxis);
	const float lossWeight = GET_PROP(prop, Loss, lossWeight);
	Dtype loss;
	soooa_gpu_dot(count, this->ones.device_data(), this->errors.device_data(), &loss);
	this->_outputData[0]->mutable_host_data()[0] = loss * Dtype(lossWeight) /
			this->_inputData[0]->getShape(firstAxis);
	//this->_outputData[0]->mutable_host_data()[0] = loss * Dtype(this->lossWeight);
	//cout << "smoothl1loss: " << this->_outputData[0]->host_data()[0] << endl;
}


template <typename Dtype>
__global__ void SmoothL1Backward(const uint32_t n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f'(x) = sigma2 * sigma2 * x         if |x| < 1 / sigma2 / sigma2
  //       = sign(x)                   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = sigma2 * val;
    } else {
      out[index] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
}


template <typename Dtype>
void SmoothL1LossLayer<Dtype>::backpropagation() {
	// after forwards, diff holds w_in * (b0 - b1)
	const float sigma2 = GET_PROP(prop, SmoothL1Loss, sigma) * GET_PROP(prop, SmoothL1Loss, sigma);
	const uint32_t count = this->diff.getCount();
	SmoothL1Backward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
			count, this->diff.device_data(), this->diff.mutable_device_data(), sigma2);
	CUDA_POST_KERNEL_CHECK;

	const vector<bool> propDown = GET_PROP(prop, SmoothL1Loss, propDown);
	const uint32_t firstAxis = GET_PROP(prop, SmoothL1Loss, firstAxis);
	for (uint32_t i = 0; i < 2; i++) {
		if (propDown[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			// XXX: caffe, top[0]->cpu_diff()[0]에 대해서 set하는 부분을 찾을 수 없고
			// 현재 특수한 값이 들어 있는 것이 아닌 1의 값이 들어있어 상수 1.0f으로 대체
			//const Dtype alpha = sign * this->_outputData[0]->host_grad()[0] /
			//		this->_inputData[i]->batches();

			const Dtype alpha = sign * GET_PROP(prop, Loss, lossWeight) /
					this->_inputData[i]->getShape(firstAxis);
			soooa_gpu_axpby(
					count,
					alpha,
					this->diff.device_data(),
					Dtype(0),
					this->_inputData[i]->mutable_device_grad());

			//this->_printOn();
			//this->_inputData[i]->print_grad({}, false, -1);
			//this->_printOff();

			if (this->hasWeights) {
				// Scale by "inside" weight
				soooa_gpu_mul(
						count,
						this->_inputData[2]->device_data(),
						this->_inputData[i]->device_grad(),
						this->_inputData[i]->mutable_device_grad());
				// Scale by "outside" weight
				soooa_gpu_mul(
						count,
						this->_inputData[3]->device_data(),
						this->_inputData[i]->device_grad(),
						this->_inputData[i]->mutable_device_grad());
			}
		}
	}

	/*
	if (GET_PROP(prop, SmoothL1Loss, name) == "rpn_loss_bbox") {
		this->_printOn();
		this->_inputData[i]->print_grad({}, false);
		this->_printOff();
	}
	*/
}

template <typename Dtype>
Dtype SmoothL1LossLayer<Dtype>::cost() {
	return this->_outputData[0]->host_data()[0];
}





/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* SmoothL1LossLayer<Dtype>::initLayer() {
	SmoothL1LossLayer* layer = NULL;
	SNEW(layer, SmoothL1LossLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::destroyLayer(void* instancePtr) {
    SmoothL1LossLayer<Dtype>* layer = (SmoothL1LossLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 4);
	} else {
		SASSERT0(index == 0);
	}

    SmoothL1LossLayer<Dtype>* layer = (SmoothL1LossLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool SmoothL1LossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    SmoothL1LossLayer<Dtype>* layer = (SmoothL1LossLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	SmoothL1LossLayer<Dtype>* layer = (SmoothL1LossLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::backwardTensor(void* instancePtr) {
	SmoothL1LossLayer<Dtype>* layer = (SmoothL1LossLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool SmoothL1LossLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 4)
        return false;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 1;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t SmoothL1LossLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    const size_t inputCount = tensorCount(inputShape[0]);

    uint64_t size = 0;
    size += ALIGNUP(sizeof(Dtype) * inputCount, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    size += ALIGNUP(sizeof(Dtype) * inputCount, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
    size += ALIGNUP(sizeof(Dtype) * inputCount, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    return size;
}


template class SmoothL1LossLayer<float>;





