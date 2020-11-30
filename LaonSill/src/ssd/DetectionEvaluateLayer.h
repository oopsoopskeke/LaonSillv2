/*
 * DetectionEvaluateLayer.h
 *
 *  Created on: May 15, 2017
 *      Author: jkim
 */

#ifndef DETECTIONEVALUATELAYER_H_
#define DETECTIONEVALUATELAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "ssd_common.h"
#include "MeasureLayer.h"
#include "LayerFunc.h"

/*
 * @brief Generate the detection evaluation based on DetectionOutputLayer and
 * ground truth bounding box labels.
 *
 * Intended for use with MultiBox detection method.
 *
 * inputData Structure
 * intputData[0] (detection_out) (1 x 1 x # of detections x 7)
 * [0]: item_id (item index in batch)
 * [1]: label
 * [2]: score
 * [3]: xmin (normalized to [0.0 ~ 1.0] or original size (not scaled))
 * [4]: ymin
 * [5]: xmax
 * [6]: ymax
 *
 * inputData[1] (label) (1 x 1 x # of gts x 8)
 * [0]: item_id
 * [1]: label
 * [2]: N/A
 * [3]: xmin
 * [4]: ymin
 * [5]: xmax
 * [6]: ymax
 * [7]: difficult
 *
 * <if box location is normalized, should specify test_name_size file at layer config>
 *
 */
template <typename Dtype>
class DetectionEvaluateLayer : public MeasureLayer<Dtype> {
public:
	DetectionEvaluateLayer();
	virtual ~DetectionEvaluateLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	Dtype measure();
	Dtype measureAll();

private:
	void collectBatchResult();
	float testDetection();

private:
	std::vector<std::pair<int, int>> sizes;
	int count;
	bool useNormalizedBBox;

	std::map<int, std::vector<std::pair<float, int>>> truePos;
	std::map<int, std::vector<std::pair<float, int>>> falsePos;
	std::map<int, int> numPos;


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

#endif /* DETECTIONEVALUATELAYER_H_ */
