/*
 * BiasLayer.h
 *
 *  Created on: Jan 6, 2018
 *      Author: jkim
 */

#ifndef BIASLAYER_H_
#define BIASLAYER_H_

#include "common.h"
#include "LearnableLayer.h"
#include "LayerPropList.h"
#include "LayerFunc.h"

template <typename Dtype>
class BiasLayer : public LearnableLayer<Dtype> {
public:
	BiasLayer();
	BiasLayer(_BiasPropLayer* prop);
	virtual ~BiasLayer();

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

	virtual void update();
	void applyChanges(LearnableLayer<Dtype> *targetLayer);
	void syncParams(LearnableLayer<Dtype> *targetLayer);

private:
	Data<Dtype> biasMultiplier;
	int outerDim;
	int biasDim;
	int innerDim;
	int dim;

	std::vector<update_param> updatePolicies;

	_BiasPropLayer* prop;


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

public:
    static int INNER_ID;
};

#endif /* BIASLAYER_H_ */
