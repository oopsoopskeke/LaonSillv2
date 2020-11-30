/*
 * ProposalLayer.h
 *
 *  Created on: Nov 29, 2016
 *      Author: jkim
 */

#ifndef PROPOSALLAYER_H_
#define PROPOSALLAYER_H_

#include <vector>

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
public:
	ProposalLayer();
	virtual ~ProposalLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void _filterBoxes(std::vector<std::vector<float>>& boxes,
			const float minSize, std::vector<uint32_t>& keep);

private:
	uint32_t numAnchors;
	std::vector<std::vector<float>> anchors;


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

#endif /* PROPOSALLAYER_H_ */
