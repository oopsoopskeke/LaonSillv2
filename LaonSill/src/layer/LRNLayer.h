/**
 * @file	LRNLayer.h
 * @date	2016/5/25
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_LRNLAYER_H_
#define LAYER_LRNLAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LayerConfig.h"
#include "LayerFunc.h"

/**
 * @brief Local Response Normalization 레이어
 * @details 입력값의 row x column 상의 값들에 대해 인접 채널의 동일 위치값들을 
 *          이용(ACROSS CHANNEL)하여 정규화하는 레이어 
 *          (WITHIN CHANNEL과 같이 한 채널 내에서 정규화하는 방법도 있으나 아직 사용하지 않아
 *           별도 파라미터로 기능을 제공하지 않음)
 *          'http://caffe.berkeleyvision.org/tutorial/layers.html'의 Local Response 
 *          Normalization (LRN) 항목 참고 (1+(α/n)∑ixi^2)^β의 수식으로 계산
 */
template <typename Dtype>
class LRNLayer : public Layer<Dtype> {
public:
	LRNLayer();
	virtual ~LRNLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

protected:
	cudnnTensorDescriptor_t inputTensorDesc;	///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t outputTensorDesc;	///< cudnn 출력 데이터(n-D 데이터셋) 구조 정보
	cudnnLRNDescriptor_t lrnDesc;				///< cudnn LRN 연산 정보 구조체

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

#endif /* LAYER_LRNLAYER_H_ */
