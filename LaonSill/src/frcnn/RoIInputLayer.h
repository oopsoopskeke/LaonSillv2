/*
 * RoIInputLayer.h
 *
 *  Created on: Nov 11, 2016
 *      Author: jkim
 */

#ifndef ROIINPUTLAYER_H_
#define ROIINPUTLAYER_H_

#include <opencv2/highgui/highgui.hpp>

#include "frcnn_common.h"
#include "DataSet.h"
#include "InputLayer.h"
#include "IMDB.h"
#include "SysLog.h"
#include "LayerFunc.h"




/**
 * @brief
 *
 * im_info (shape: (1, 1, 1, 3)
 * [0]: height, 원본 픽셀수에서 scale이 곱해진 수 (org: 375, scale: 1.333.., -> 500)
 * [1]: width, 원본 픽셀수에서 scale이 곱해진 수 (org: 500, scale: 1.333.., -> 667)
 * [2]: scale, 원본에 곱해지는 수 (ex. 1.3333)
 *
 * im_info (shape: (1, 1, # of boxes, 5))
 * [0]: x1, xmin, 픽셀수, scale 적용됨, 234.66..
 * [1]: y1, ymin, 280.00
 * [2]: x2, xmax, 316.00
 * [3]: y2, ymax, 450.67
 * [4]: label, 9
 *
 */
template <typename Dtype>
class RoIInputLayer : public InputLayer<Dtype> {
public:
	RoIInputLayer();
	virtual ~RoIInputLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

private:
	IMDB* getImdb();
	void getTrainingRoidb(IMDB* imdb);
	IMDB* getRoidb();
	IMDB* combinedRoidb();
	bool isValidRoidb(RoIDB& roidb);
	void filterRoidb(std::vector<RoIDB>& roidb);

	void shuffleRoidbInds();
	void getNextMiniBatch();
	void getNextMiniBatchInds(std::vector<uint32_t>& inds);
	void getMiniBatch(const std::vector<RoIDB>& roidb, const std::vector<uint32_t>& inds);
	std::vector<cv::Mat> getImageBlob(const std::vector<RoIDB>& roidb,
			const std::vector<uint32_t>& scaleInds, std::vector<float>& imScales);
	float prepImForBlob(cv::Mat& im, cv::Mat& imResized, const std::vector<float>& pixelMeans,
			const uint32_t targetSize, const uint32_t maxSize);
	void imListToBlob(std::vector<cv::Mat>& ims);


public:
	std::vector<std::vector<float>> bboxMeans;
	std::vector<std::vector<float>> bboxStds;
	IMDB* imdb;

	std::vector<uint32_t> perm;
	uint32_t cur;

	std::vector<std::vector<Data<Dtype>*>> proposalTargetDataList;

	std::vector<cv::Scalar> boxColors;


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

#endif /* ROIINPUTLAYER_H_ */
