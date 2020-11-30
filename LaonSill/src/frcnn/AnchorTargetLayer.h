/*
 * AnchorTargetLayer.h
 *
 *  Created on: Nov 18, 2016
 *      Author: jkim
 */

#ifndef ANCHORTARGETLAYER_H_
#define ANCHORTARGETLAYER_H_




#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"


/**
 * Assign anchors to ground-truth targets. Produces anchor classification
 * labels and bounding-box regression targets.
 * 실제 input data에 대한 cls score, bbox pred를 계산, loss를 계산할 때 쓸 데이터를 생성한다.
 */
template <typename Dtype>
class AnchorTargetLayer : public Layer<Dtype> {
public:
	AnchorTargetLayer();
	virtual ~AnchorTargetLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void _computeTargets(const std::vector<std::vector<float>>& exRois,
			const std::vector<std::vector<float>>& gtRois,
			std::vector<std::vector<float>>& bboxTargets);

	void _unmap(const std::vector<int>& data, const uint32_t count,
			const std::vector<uint32_t>& indsInside, const int fill,
			std::vector<int>& result);
	void _unmap(const std::vector<std::vector<float>>& data, const uint32_t count,
			const std::vector<uint32_t>& indsInside,
			std::vector<std::vector<float>>& result);


protected:
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

#endif /* ANCHORTARGETLAYER_H_ */
































