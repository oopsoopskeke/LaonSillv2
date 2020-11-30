/**
 * @file PoolingLayer.h
 * @date 2016/5/23
 * @author jhkim
 * @brief
 * @details
 */


#ifndef LAYER_POOLINGLAYER_H_
#define LAYER_POOLINGLAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "Pooling.h"
#include "PoolingFactory.h"
#include "LayerFunc.h"

/**
 * @brief 풀링 레이어
 * @details Max 풀링, Average 풀링 제공
 *          padding에 관한 옵션을 제공하지 않고 있고
 *          GoogLeNet에 따라 Max 풀링의 경우 padding을 기본으로, Average 풀링의 경우 
 *          Non padding을 기본으로 하고 있다.
 */
template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
public:
	PoolingLayer();
	virtual ~PoolingLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

protected:
	cudnnTensorDescriptor_t inputTensorDesc;	///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t outputTensorDesc;	///< cudnn 출력 데이터(n-D 데이터셋) 구조 정보
    Pooling<Dtype>* pooling_fn;
    bool globalPooling;
    pool_dim poolDim;
    PoolingType poolingType;

public:
    /****************************************************************************
     * layer callback functions
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
    static bool checkShape(std::vector<TensorShape> inputShape,
            std::vector<TensorShape> &outputShape);
    static uint64_t calcGPUSize(std::vector<TensorShape> inputShape);
};

#endif /* LAYER_POOLINGLAYER_H_ */
