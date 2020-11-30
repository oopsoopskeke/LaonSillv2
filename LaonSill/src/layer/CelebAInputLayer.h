/**
 * @file CelebAInputLayer.h
 * @date 2017-02-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef CELEBAINPUTLAYER_H
#define CELEBAINPUTLAYER_H 

#include <string>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "InputLayer.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class CelebAInputLayer : public InputLayer<Dtype> {
public: 
    CelebAInputLayer();
    virtual ~CelebAInputLayer();

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

protected:
    int         imageCount;

    int         imageRow;
    int         imageCol;
    int         imageChannel;

    void        fillImagePaths();
    void        loadImages(int baseIdx);
    void        loadPixels(cv::Mat image, int imageIndex);
    void        shuffleImages();

    std::vector<int> imageIndexes;
    std::vector<std::string> imagePaths;
    Dtype*      images; 
    int         currentBatchIndex;

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
#endif /* CELEBAINPUTLAYER_H */
