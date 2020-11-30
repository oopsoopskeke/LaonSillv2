/*
 * ProposalTargetLayer.h
 *
 *  Created on: Nov 30, 2016
 *      Author: jkim
 */

#ifndef PROPOSALTARGETLAYER_H_
#define PROPOSALTARGETLAYER_H_

#include <vector>

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"


/**
 * Assign object detection proposals to ground-truth targets. Produces proposal
 * classification labels and bounding-box regression targets.
 */
template <typename Dtype>
class ProposalTargetLayer : public Layer<Dtype> {
public:
	ProposalTargetLayer();
	virtual ~ProposalTargetLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void initialize();
	void _sampleRois(
			const std::vector<std::vector<float>>& allRois,
			const std::vector<std::vector<float>>& gtBoxes,
			const uint32_t fgRoisPerImage,
			const uint32_t roisPerImage,
			std::vector<uint32_t>& labels,
			std::vector<std::vector<float>>& rois,
			std::vector<std::vector<float>>& bboxTargets,
			std::vector<std::vector<float>>& bboxInsideWeights);
	void _computeTargets(
			const std::vector<std::vector<float>>& exRois,
			const uint32_t exRoisOffset,
			const std::vector<std::vector<float>>& gtRois,
			const uint32_t gtRoisOffset,
			const std::vector<uint32_t>& labels,
			std::vector<std::vector<float>>& targets,
			const uint32_t targetOffset);
	void _getBboxRegressionLabels(
			const std::vector<std::vector<float>>& bboxTargetData,
			std::vector<std::vector<float>>& bboxTargets,
			std::vector<std::vector<float>>& bboxInsideWeights);



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

#endif /* PROPOSALTARGETLAYER_H_ */
