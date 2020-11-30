/*
 * BatchNorm3Layer.h
 *
 *  Created on: Dec 18, 2017
 *      Author: jkim
 */

#ifndef BATCHNORM3LAYER_H_
#define BATCHNORM3LAYER_H_

#include "common.h"
#include "LearnableLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class BatchNorm3Layer : public LearnableLayer<Dtype> {
public:
	BatchNorm3Layer();
	virtual ~BatchNorm3Layer();

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

	virtual void update();
	void applyChanges(LearnableLayer<Dtype> *targetLayer);
	void syncParams(LearnableLayer<Dtype> *targetLayer);


private:
	bool useGlobalStats;
	Dtype movingAverageFraction;
	int channels;
	Dtype eps;

	Data<Dtype> mean;
	Data<Dtype> variance;
	Data<Dtype> temp;
	Data<Dtype> xNorm;

	Data<Dtype> batchSumMultiplier;
	Data<Dtype> numByChans;
	Data<Dtype> spatialSumMultiplier;

	std::vector<update_param> updatePolicies;


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


#endif /* BATCHNORM3LAYER_H_ */
