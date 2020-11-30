/**
 * @file YOLORegionLayer.h
 * @date 2018-01-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

#ifndef YOLOREGIONLAYER_H
#define YOLOREGIONLAYER_H 

template<typename Dtype>
class YOLORegionLayer : public Layer<Dtype> {
public: 
    YOLORegionLayer();
    virtual ~YOLORegionLayer();

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

private:
    Data<Dtype>* anchorSet;
};
#endif /* YOLOREGIONLAYER_H */
