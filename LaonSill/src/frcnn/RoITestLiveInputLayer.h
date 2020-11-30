/*
 * RoITestLiveInputLayer.h
 *
 *  Created on: Jul 13, 2017
 *      Author: jkim
 */

#ifndef ROITESTLIVEINPUTLAYER_H_
#define ROITESTLIVEINPUTLAYER_H_

#include <opencv2/highgui/highgui.hpp>

#include "frcnn_common.h"
#include "InputLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class RoITestLiveInputLayer : public InputLayer<Dtype> {
public:
	RoITestLiveInputLayer();
	virtual ~RoITestLiveInputLayer();

    int getNumTrainData();
    virtual void feedImage(const int channels, const int height, const int width,
            float* image);

	virtual void reshape();
	virtual void feedforward();

private:
	void getNextMiniBatch();
	float getBlobs(cv::Mat& im);
	float getImageBlob(cv::Mat& im);


private:
	int channels;
	int height;
	int width;
	float* image;


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

#endif /* ROITESTLIVEINPUTLAYER_H_ */
