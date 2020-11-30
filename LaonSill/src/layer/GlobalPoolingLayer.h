/**
 * @file GlobalPoolingLayer.h
 * @date 2017-12-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef GLOBALPOOLING_H
#define GLOBALPOOLING_H 

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

/**
 * @brief   global pooling 레이어
 * @details 현재는 avg pooling만 지원하고 있음. YOLO 구현땜에 만듬. 
 *          모든 이미지 인풋데이터를 1개의 global average 값으로 뱉어내는 레이어.
 *          결국 필터개수만큼의 아웃풋이 나오게 된다.
 */
template<typename Dtype>
class GlobalPoolingLayer : public Layer<Dtype> {
public: 
    GlobalPoolingLayer();
    virtual ~GlobalPoolingLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

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

#endif /* GLOBALPOOLING_H */
