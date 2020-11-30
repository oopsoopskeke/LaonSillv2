/*
 * RoITestInputLayer.h
 *
 *  Created on: Dec 16, 2016
 *      Author: jkim
 */

#ifndef ROITESTINPUTLAYER_H_
#define ROITESTINPUTLAYER_H_

#include <opencv2/highgui/highgui.hpp>

#include "frcnn_common.h"
#include "InputLayer.h"
#include "IMDB.h"
#include "LayerFunc.h"

template <typename Dtype>
class RoITestInputLayer : public InputLayer<Dtype> {
public:
	RoITestInputLayer();
	virtual ~RoITestInputLayer();

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	virtual void reshape();
	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

private:
	void getNextMiniBatch();

	void imDetect(cv::Mat& im);
	float getBlobs(cv::Mat& im);
	float getImageBlob(cv::Mat& im);
	void imToBlob(cv::Mat& im);

	IMDB* combinedRoidb(const std::string& imdb_name);
	IMDB* getRoidb(const std::string& imdb_name);
	IMDB* getImdb(const std::string& imdb_name);


public:
	IMDB* imdb;
	std::vector<uint32_t> perm;
	uint32_t cur;

	std::vector<cv::Scalar> boxColors;

	bool isMeasureAP;


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

#endif /* ROITESTINPUTLAYER_H_ */


















