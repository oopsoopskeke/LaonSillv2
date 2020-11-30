/**
 * @file NoiseInputLayer.h
 * @date 2017-02-16
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef NOISEINPUTLAYER_H
#define NOISEINPUTLAYER_H 

#include <string>

#include "common.h"
#include "InputLayer.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class NoiseInputLayer : public InputLayer<Dtype> {
public: 
    NoiseInputLayer();
    virtual ~NoiseInputLayer();

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

    void setRegenerateNoise(bool regenerate);

protected:
    bool prepareUniformArray();
    void prepareLinearTranMatrix();

    int batchSize;

    Dtype* uniformArray;
    Dtype* linearTransMatrix;

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

#endif /* NOISEINPUTLAYER_H */
