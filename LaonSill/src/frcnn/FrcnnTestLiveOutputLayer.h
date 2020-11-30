/*
 * FrcnnTestLiveOutputLayer.h
 *
 *  Created on: Jul 13, 2017
 *      Author: jkim
 */

#ifndef FRCNNTESTLIVEOUTPUTLAYER_H_
#define FRCNNTESTLIVEOUTPUTLAYER_H_

#include "BaseLayer.h"
#include "SysLog.h"
#include "frcnn_common.h"
#include "ssd_common.h"
#include "LayerFunc.h"

template <typename Dtype>
class FrcnnTestLiveOutputLayer : public Layer<Dtype> {
public:
	FrcnnTestLiveOutputLayer();
	virtual ~FrcnnTestLiveOutputLayer();

	virtual void reshape();
	virtual void feedforward();

private:
	void imDetect(std::vector<std::vector<Dtype>>& scores,
			std::vector<std::vector<Dtype>>& predBoxes);
	void testNet(std::vector<std::vector<Dtype>>& scores,
			std::vector<std::vector<Dtype>>& predBoxes);

	void fillClsScores(std::vector<std::vector<Dtype>>& scores, int clsInd,
			std::vector<Dtype>& clsScores);
	void fillClsBoxes(std::vector<std::vector<Dtype>>& boxes, int clsInd,
			std::vector<std::vector<Dtype>>& clsBoxes);

	void visDetection();

public:
	std::vector<cv::Scalar> boxColors;
	LabelMap<Dtype> labelMap;


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

#endif /* FRCNNTESTLIVEOUTPUTLAYER_H_ */
