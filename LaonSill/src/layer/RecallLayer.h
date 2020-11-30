/*
 * RecallLayer.h
 *
 *  Created on: Aug 9, 2017
 *      Author: jkim
 */

#ifndef RECALLLAYER_H_
#define RECALLLAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class RecallLayer : public Layer<Dtype> {
public:
	RecallLayer();
	virtual ~RecallLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

	void save(const float score, const int batchIndex);


private:
	int sampleCount[2];
	int ttCount[2];		// true-true count		label과 예측이 일치
	//int tfCount[2];		// true-false count		예측이 일치하지 않은 경우

	int missCount[2];



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

#endif /* RECALLLAYER_H_ */
