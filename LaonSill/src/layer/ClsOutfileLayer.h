/*
 * ClsOutfileLayer.h
 *
 *  Created on: Jan 11, 2018
 *      Author: jkim
 */

#ifndef CLSOUTFILELAYER_H_
#define CLSOUTFILELAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

/**
 * @brief PlantyNet의 무해 / 유해 두 가지 케이스에 대해서만 처리
 * 다른 곳에서 사용할 경우 복수 레이블로 일반화해야 함.
 *
 */
template <typename Dtype>
class ClsOutfileLayer : public Layer<Dtype> {
public:
	ClsOutfileLayer();
	virtual ~ClsOutfileLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	bool hasDataSetFile;
	std::string dataSetFile;
	std::string outFilePath;

	int testCount;
	std::vector<Dtype> scores;
	//std::vector<Dtype> labels;

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

#endif /* CLSOUTFILELAYER_H_ */
