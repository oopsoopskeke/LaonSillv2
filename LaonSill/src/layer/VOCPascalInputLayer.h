/**
 * @file VOCPascalInputLayer.h
 * @date 2017-04-18
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef VOCPASCALINPUTLAYER_H
#define VOCPASCALINPUTLAYER_H 

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "InputLayer.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

typedef struct VOCPascalMeta_s {
    std::string     imagePath;
    float           x;
    float           y;
    float           width;
    float           height;
    int             gridX;
    int             gridY;
    int             classID;
} VOCPascalMeta;

template<typename Dtype>
class VOCPascalInputLayer : public InputLayer<Dtype> {
public: 
    VOCPascalInputLayer();
    virtual ~VOCPascalInputLayer();

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

    void        fillMetas();
    void        loadImages(int baseIdx);
    void        loadLabels(int baseIdx);
    void        loadPixels(cv::Mat image, int imageIndex);
    void        shuffleImages();

    Dtype*                      images;
    Dtype*                      labels;

    std::vector<VOCPascalMeta>  metas;
    std::vector<int>            metaIndexes;
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

#endif /* VOCPASCALINPUTLAYER_H */
