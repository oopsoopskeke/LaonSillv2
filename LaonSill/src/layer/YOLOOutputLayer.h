/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * @ file      YOLOOutputLayer.h
 * @ date      2018-02-06
 * @ author    SUN
 * @ brief
 * @ details
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

#ifndef YOLOOUTPUTLAYER_H
#define YOLOOUTPUTLAYER_H

template<typename Dtype>
class YOLOOutputLayer : public Layer<Dtype> {
public:
    YOLOOutputLayer();
    virtual ~YOLOOutputLayer();

	virtual void reshape();
	virtual void feedforward();

private:
    void YOLOOutputForward(const Dtype* inputData, const int batch,
            const int side1, const int side2, const int dim);

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

#endif /* YOLOOUTPUTLAYER_H */
