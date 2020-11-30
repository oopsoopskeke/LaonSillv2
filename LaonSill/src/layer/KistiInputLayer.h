/**
 * @file KistiInputLayer.h
 * @date 2017-03-28
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef KISTIINPUTLAYER_H
#define KISTIINPUTLAYER_H 

#include <string>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "InputLayer.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

#define KISTIDATA_DEFAULT_RESIZED_ROW_SIZE       224
#define KISTIDATA_DEFAULT_RESIZED_COL_SIZE       224

typedef struct KistiData_s {
    std::string filePath;
    std::vector<int> labels;
} KistiData;

template<typename Dtype>
class KistiInputLayer : public InputLayer<Dtype> {
public: 
    KistiInputLayer();
    virtual ~KistiInputLayer();

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();
    void setTrain(bool train);

protected:
    int imageRow;
    int imageCol;
    int imageChannel;

public:
    std::map<std::string, int> keywordMap;
    std::vector<KistiData> trainData;
    std::vector<KistiData> testData;
protected:

    void registerData(std::string filePath, bool isTrainData);
    void prepareKeywordMap();
    void prepareData();
    void loadPixels(cv::Mat image, int imageIndex);
    void loadImages(int batchIndex);
    void loadLabels(int batchIndex);
    void shuffleImages();

    Dtype *images;
    Dtype *labels;
    int currentBatchIndex;

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
#endif /* KISTIINPUTLAYER_H */
