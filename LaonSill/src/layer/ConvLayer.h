/**
 * @file	ConvLayer.h
 * @date	2016/5/23
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_CONVLAYER_H_
#define LAYER_CONVLAYER_H_

#include <stddef.h>
#include <iostream>
#include <map>
#include <string>

#include "common.h"
#include "Util.h"
#include "BaseLayer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"
#include "LayerFunc.h"

/**
 * @brief 컨볼루션 레이어
 * @details
 */
template <typename Dtype>
class ConvLayer : public LearnableLayer<Dtype> {
public:
	ConvLayer();
	virtual ~ConvLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	
    //
	virtual void update();
    void applyChanges(LearnableLayer<Dtype> *targetLayer);
    void syncParams(LearnableLayer<Dtype> *targetLayer);

protected:
	void _computeFiltersConvolutionData();
	void _computeFiltersGrad();
	void _computeBiasesGrad();
	void _computeInputGrad();

	enum ParamType {
		Filter = 0,
		Bias = 1
	};

protected:
	cudnnTensorDescriptor_t inputTensorDesc;	///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t outputTensorDesc;	///< cudnn 출력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t biasTensorDesc;		///< cudnn bias 구조 정보 구조체
	cudnnFilterDescriptor_t filterDesc;			///< cudnn filter 구조 정보 구조체
	cudnnConvolutionDescriptor_t convDesc;		///< cudnn 컨볼루션 연산 정보 구조체
	cudnnConvolutionFwdAlgo_t convFwdAlgo;		///< cudnn 컨볼루션 포워드 알고리즘 열거형 
	cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;	///< cudnn filter 백워드 열거형
	cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;	///< cudnn data 백워드 알고리즘 열거형

	size_t workspaceSize;	///< cudnn forward, backward에 필요한 작업공간 GPU 메모리 사이즈
	void *d_workspace;		///< cudnn forward, backward에 필요한 작업공간 장치 메모리 포인터


public:
    bool deconv;
    int deconvExtraCell;
    bool biasTerm;

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

#endif /* LAYER_CONVLAYER_H_ */
