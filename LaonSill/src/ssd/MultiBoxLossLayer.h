/*
 * MultiBoxLossLayer.h
 *
 *  Created on: Apr 27, 2017
 *      Author: jkim
 */

#ifndef MULTIBOXLOSSLAYER_H_
#define MULTIBOXLOSSLAYER_H_

#include "common.h"
#include "LossLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class MultiBoxLossLayer : public LossLayer<Dtype> {
public:
	MultiBoxLossLayer();
	virtual ~MultiBoxLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

private:
	void setLayerData(Layer<Dtype>* layer, const std::string& type,
			std::vector<Data<Dtype>*>& dataVec);
	Layer<Dtype>* buildLocLossLayer(const LocLossType locLossType);
	Layer<Dtype>* buildConfLossLayer(const ConfLossType confLossType);

private:
	/*
	std::string locLossType;
	std::string confLossType;
	Dtype locWeight;
	int numClasses;
	bool shareLocation;
	std::string matchType;
	Dtype overlapThreshold;
	bool usePriorForMatching;
	int backgroundLabelId;
	bool useDifficultGt;
	Dtype negPosRatio;
	Dtype negOverlap;
	std::string codeType;
	bool ignoreCrossBoundaryBbox;
	std::string miningType;
	bool doNegMining;
	bool encodeVarianceInTarget;
	bool bpInside;
	bool usePriorForNMS;
	Dtype nmsThresh;
	int topK;
	Dtype eta;
	int sampleSize;
	bool mapObjectToAgnostic;
	*/


	int locClasses;
	int numGt;
	int num;
	int numPriors;

	int numMatches;
	int numConf;
	std::vector<std::map<int, std::vector<int>>> allMatchIndices;
	std::vector<std::vector<int>> allNegIndices;

	// How to normalize the loss


	Layer<Dtype>* locLossLayer;
	//std::vector<Data<Dtype>*> locInputVec;
	//std::vector<Data<Dtype>*> locOutputVec;
	Data<Dtype> locPred;
	Data<Dtype> locGt;
	Data<Dtype> locLoss;

	Layer<Dtype>* confLossLayer;
	//std::vector<Data<Dtype>*> confInputVec;
	//std::vector<Data<Dtype>*> confOutputVec;
	Data<Dtype> confPred;
	Data<Dtype> confGt;
	Data<Dtype> confLoss;


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

#endif /* MULTIBOXLOSSLAYER_H_ */
