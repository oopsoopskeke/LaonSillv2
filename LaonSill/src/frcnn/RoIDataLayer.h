/*
 * RoIDataLayer.h
 *
 *  Created on: Nov 11, 2016
 *      Author: jkim
 */

#ifndef ROIDATALAYER_H_
#define ROIDATALAYER_H_

#include <opencv2/highgui/highgui.hpp>

#include "frcnn_common.h"
#include "InputLayer.h"
#include "Datum.h"
#include "InputDataProvider.h"
#include "DataTransformer.h"
#include "DataReader.h"
#include "LayerFunc.h"






/**
 * @brief
 *
 * XXX: RoIInputLayer 대비, gt_boxes의 값이 1 ~ 2 pixel 정도 shift되는 효과 있음
 *
 *
 * im_info (shape: (1, 1, 1, 3)
 * [0]: height, 원본 픽셀수에서 scale이 곱해진 수 (org: 375, scale: 1.333.., -> 500)
 * [1]: width, 원본 픽셀수에서 scale이 곱해진 수 (org: 500, scale: 1.333.., -> 667)
 * [2]: scale, 원본에 곱해지는 수 (ex. 1.3333)
 *
 * gt_boxes (shape: (1, 1, # of boxes, 5))
 * [0]: x1, xmin, 픽셀수, scale 적용됨, 234.66..
 * [1]: y1, ymin, 280.00
 * [2]: x2, xmax, 316.00
 * [3]: y2, ymax, 450.67
 * [4]: label, 9
 *
 */
template <typename Dtype>
class RoIDataLayer : public InputLayer<Dtype> {
public:
	RoIDataLayer();
	virtual ~RoIDataLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

private:
	void shuffleRoidbInds();
	void load_batch();


public:
	InputPool* inputPool;
	std::vector<std::vector<float>> bboxMeans;
	std::vector<std::vector<float>> bboxStds;

	std::vector<std::vector<Data<Dtype>*>> proposalTargetDataList;

private:
	DataReader<class AnnotatedDatum> dataReader;
	DataTransformer<Dtype> dataTransformer;


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

#endif /* ROIDATALAYER_H_ */
