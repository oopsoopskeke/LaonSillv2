/*
 * DetectionOutputLayer.h
 *
 *  Created on: May 15, 2017
 *      Author: jkim
 */

#ifndef DETECTIONOUTPUTLAYER_H_
#define DETECTIONOUTPUTLAYER_H_

#include <boost/property_tree/ptree.hpp>

#include "common.h"
#include "BaseLayer.h"
#include "ssd_common.h"
#include "LayerFunc.h"

/*
 * @brief Generate the detection output based on location and confidence
 *        predictions by doing non maximum suppression.
 */
template <typename Dtype>
class DetectionOutputLayer : public Layer<Dtype> {
public:
	DetectionOutputLayer();
	virtual ~DetectionOutputLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:


private:
	/*
	int numClasses;
	bool shareLocation;
	int backgroundLabelId;

	// NonMaximumSuppressionParameter
	float nmsThreshold;
	int topK;
	float eta;

	std::string outputDirectory;
	std::string outputNamePrefix;
	std::string outputFormat;
	std::string labelMapFile;
	int numTestImage;

	std::string codeType;

	bool varianceEncodedInTarget;
	int keepTopK;
	float confidenceThreshold;
	bool visualize;
	float visualizeThresh;
	std::string saveFile;
	*/

	int numLocClasses;
	bool needSave;
	int numPriors;
	int nameCount;

	Data<Dtype> bboxPreds;						// mbox_loc과 동일 shape
	Data<Dtype> bboxPermute;					// !shareLocation인 경우 사용
	Data<Dtype> confPermute;					// mbox_conf_flatten과 동일 shape



	LabelMap<Dtype> labelMap;
	std::vector<std::string> names;				// test 이미지 이름 목록
	std::vector<std::pair<int, int>> sizes;		// test 이미지 height, width 목록

	std::map<int, std::string> labelToName;
	std::map<int, std::string> labelToDisplayName;

	boost::property_tree::ptree detections;


	Data<Dtype> temp;



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

#endif /* DETECTIONOUTPUTLAYER_H_ */
