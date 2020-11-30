/*
 * RoITestVideoInputLayer.h
 *
 *  Created on: May 30, 2017
 *      Author: jkim
 */

#ifndef ROITESTVIDEOINPUTLAYER_H_
#define ROITESTVIDEOINPUTLAYER_H_

#include <opencv2/highgui/highgui.hpp>

#include "frcnn_common.h"
#include "InputLayer.h"
#include "IMDB.h"
#include "LayerFunc.h"

template <typename Dtype>
class RoITestVideoInputLayer : public InputLayer<Dtype> {
public:
	RoITestVideoInputLayer();
	virtual ~RoITestVideoInputLayer();

	int getNumTrainData();
	int getNumTestData();
	void shuffleTrainDataSet();

	virtual void reshape();
	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

private:
	void getNextMiniBatch();
	float getImageBlob(cv::Mat& im);
	void imToBlob(cv::Mat& im);

	IMDB* combinedRoidb(const std::string& imdb_name);
	IMDB* getRoidb(const std::string& imdb_name);


public:
	std::vector<cv::Scalar> boxColors;
	cv::VideoCapture cap;


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

#endif /* ROITESTVIDEOINPUTLAYER_H_ */
