/**
 * @file BatchNormLayer.h
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef BATCHNORMLAYER_H
#define BATCHNORMLAYER_H 

#include "common.h"
#include "BaseLayer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"
#include "LayerFunc.h"

template <typename Dtype>
class BatchNormLayer : public LearnableLayer<Dtype> {
public: 
    BatchNormLayer();
    virtual ~BatchNormLayer();

	virtual void update();
	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

private:
    void applyChanges(LearnableLayer<Dtype> *targetLayer);
    void syncParams(LearnableLayer<Dtype> *targetLayer);

    int         depth;

    Data<Dtype>    *meanSet;            // mean
    Data<Dtype>    *varSet;             // variance
    Data<Dtype>    *normInputSet;       // normalized input value

    void        computeNormInputGrad();
    void        computeVarianceGrad();
    void        computeMeanGrad();
    void        computeInputGrad();
    void        computeScaleGrad();
    void        computeShiftGrad();
public:
    void        setTrain(bool train);

protected:
    enum ParamType {
        Gamma = 0,
        Beta = 1,
        GlobalMean,
        GlobalVar,
        GlobalCount
    };

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
#endif /* BATCHNORMLAYER_H */
