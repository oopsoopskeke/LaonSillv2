/**
 * @file YOLOLossLayer.h
 * @date 2017-04-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef YOLOLOSSLAYER_H
#define YOLOLOSSLAYER_H 

#include "common.h"
#include "LossLayer.h"
#include "LayerConfig.h"
#include "LayerFunc.h"

#if 0
// for Worker
typedef struct yoloJobPack_s {
    float top;
    float left;
    float bottom;
    float right;
    float score;
    int labelIndex;
} yoloJobPack;
#endif

template<typename Dtype>
class YOLOLossLayer : public LossLayer<Dtype> {
public: 
    YOLOLossLayer();
    virtual ~YOLOLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

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

private:
    Data<Dtype>* objIdxVec;
};

#endif /* YOLOLOSSLAYER_H */
