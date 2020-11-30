/**
 * @file LiveDataInputLayer.h
 * @date 2017-12-11
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DATALIVEINPUTLAYER_H
#define DATALIVEINPUTLAYER_H 

#include "InputLayer.h"
#include "DataTransformer.h"
#include "LayerFunc.h"

#define LIVEDATAINPUTLAYER_TEST 0

template <typename Dtype>
class LiveDataInputLayer : public InputLayer<Dtype> {
public: 
    LiveDataInputLayer();
    virtual ~LiveDataInputLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	virtual void reshape();

    virtual void feedImage(const int channels, const int height, const int width,
            float* image);

private:
    DataTransformer<Dtype> dataTransformer;
    uint32_t height;
    uint32_t width;
#if LIVEDATAINPUTLAYER_TEST
    std::vector<std::string> testList;
    int cur;
#endif

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

#endif /* DATALIVEINPUTLAYER_H */
