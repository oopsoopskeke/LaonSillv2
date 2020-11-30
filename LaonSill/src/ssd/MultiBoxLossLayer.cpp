/*
 * MultiBoxLossLayer.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: jkim
 */

#include "MultiBoxLossLayer.h"
#include "ssd_common.h"
#include "SmoothL1LossLayer.h"
#include "SoftmaxWithLossLayer.h"
#include "SysLog.h"
#include "BBoxUtil.h"
#include "MathFunctions.h"
#include "PropMgmt.h"
#include "EnumDef.h"
#include "InnerLayerFunc.h"
#include "PlanParser.h"
#include "MemoryMgmt.h"

#define MULTIBOXLOSSLAYER_LOG 0

using namespace std;

template <typename Dtype>
MultiBoxLossLayer<Dtype>::MultiBoxLossLayer()
: LossLayer<Dtype>(),
  locPred("locPred"),
  locGt("locGt"),
  locLoss("locLoss"),
  confPred("confPred"),
  confGt("confGt"),
  confLoss("confLoss") {
	this->type = Layer<Dtype>::MultiBoxLoss;

	vector<bool>& propDown = SLPROP_BASE(propDown);
	if (propDown.size() == 0) {
		propDown.push_back(true);
		propDown.push_back(true);
		propDown.push_back(false);
		propDown.push_back(false);
	}

	const int numClasses = SLPROP(MultiBoxLoss, numClasses);
	const bool shareLocation = SLPROP(MultiBoxLoss, shareLocation);
	const MiningType miningType = SLPROP(MultiBoxLoss, miningType);

	bool& doNegMining = SLPROP(MultiBoxLoss, doNegMining);


	// Get other parameters.
	SASSERT(numClasses >= 0, "Must provide numClasses > 0");
	doNegMining = (miningType != MiningType::MINING_NONE) ? true : false;
	this->locClasses = shareLocation ? 1 : numClasses;

	if (doNegMining) {
		SASSERT(shareLocation,
				"Currently only support negative mining if shareLocation is true.");
	}


    this->locLossLayer = NULL;
	const LocLossType locLossType = SLPROP(MultiBoxLoss, locLossType);
	this->locLossLayer = buildLocLossLayer(locLossType);

    this->confLossLayer = NULL;
	const ConfLossType confLossType = SLPROP(MultiBoxLoss, confLossType);
	this->confLossLayer = buildConfLossLayer(confLossType);
}



template <typename Dtype>
MultiBoxLossLayer<Dtype>::~MultiBoxLossLayer() {
    if (this->locLossLayer != NULL)
        SDELETE(this->locLossLayer);

    if (this->confLossLayer != NULL)
        SDELETE(this->confLossLayer);
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();
	bool inputShapeChanged = false;
	for (int i = 0; i < this->_inputData.size(); i++) {
		if (Layer<Dtype>::_isInputShapeChanged(0)) {
			inputShapeChanged = true;
			break;
		}
	}
	if (!inputShapeChanged) return;

	const int numClasses = SLPROP(MultiBoxLoss, numClasses);

	this->num = this->_inputData[0]->batches();
	this->numPriors = this->_inputData[2]->channels() / 4;
	this->numGt = this->_inputData[3]->height();

	//cout << "num=" << this->num << ", numPriors=" << this->numPriors <<
	//		", numGt=" << this->numGt << ", locClasses=" << this->locClasses << endl;
	//cout << "channel0=" << this->_inputData[0]->channels() << ", channel1=" << this->_inputData[1]->channels() << endl;

	SASSERT0(this->_inputData[0]->batches() == this->_inputData[1]->batches());
	SASSERT(this->numPriors * this->locClasses * 4 == this->_inputData[0]->channels(),
			"Number of priors must match number of location predictions.");
	SASSERT(this->numPriors * numClasses == this->_inputData[1]->channels(),
			"Number of priors must match number of confidence predictions.");

	this->_outputData[0]->reshape({1, 1, 1, 1});
	this->_outputData[0]->mutable_host_grad()[0] = SLPROP(Loss, lossWeight);
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::feedforward() {
#if 0
#endif
	reshape();

	const int sampleSize = SLPROP(MultiBoxLoss, sampleSize);

	const uint32_t backgroundLabelId = SLPROP(MultiBoxLoss, backgroundLabelId);
	const uint32_t numClasses = SLPROP(MultiBoxLoss, numClasses);

	const float overlapThreshold = SLPROP(MultiBoxLoss, overlapThreshold);
	const float negPosRatio = SLPROP(MultiBoxLoss, negPosRatio);
	const float negOverlap = SLPROP(MultiBoxLoss, negOverlap);
	const float locWeight = SLPROP(MultiBoxLoss, locWeight);

	const bool useDifficultGt = SLPROP(MultiBoxLoss, useDifficultGt);
	const bool shareLocation = SLPROP(MultiBoxLoss, shareLocation);
	const bool bpInside = SLPROP(MultiBoxLoss, bpInside);
	const bool doNegMining = SLPROP(MultiBoxLoss, doNegMining);
	const bool mapObjectToAgnostic = SLPROP(MultiBoxLoss, mapObjectToAgnostic);
	const bool usePriorForMatching = SLPROP(MultiBoxLoss, usePriorForMatching);
	const bool encodeVarianceInTarget = SLPROP(MultiBoxLoss, encodeVarianceInTarget);
	const bool ignoreCrossBoundaryBBox = SLPROP(MultiBoxLoss, ignoreCrossBoundaryBBox);
	const bool usePriorForNMS = SLPROP(MultiBoxLoss, usePriorForNMS);

	const MatchType matchType = SLPROP(MultiBoxLoss, matchType);
	const CodeType codeType = SLPROP(MultiBoxLoss, codeType);
	const ConfLossType confLossType = SLPROP(MultiBoxLoss, confLossType);
	const LocLossType locLossType = SLPROP(MultiBoxLoss, locLossType);
	const MiningType miningType = SLPROP(MultiBoxLoss, miningType);

	const NonMaximumSuppressionParam& nmsParam = SLPROP(MultiBoxLoss, nmsParam);



	const Dtype* locData = this->_inputData[0]->host_data();
	const Dtype* confData = this->_inputData[1]->host_data();
	const Dtype* priorData = this->_inputData[2]->host_data();
	const Dtype* gtData = this->_inputData[3]->host_data();


	this->allMatchIndices.clear();
	this->allNegIndices.clear();

	// Retrieve all ground truth
	// key: item_id (index in batch), value: gt bbox list belongs to item_id
	map<int, vector<NormalizedBBox>> allGtBBoxes;
	GetGroundTruth(gtData, this->numGt, backgroundLabelId, useDifficultGt, &allGtBBoxes);

#if 0//MULTIBOXLOSSLAYER_LOG
	cout << "GetGroundTruth()" << endl;
	for (auto it = allGtBBoxes.begin(); it != allGtBBoxes.end(); it++) {
		//cout << "for itemId: " << it->first << endl;
		for (int i = 0; i < it->second.size(); i++) {
			//cout << "\t" << i << endl;
			//it->second[i].print();
			cout << it->second[i].xmin << "," << it->second[i].ymin << ","
					<< it->second[i].xmax << "," << it->second[i].ymax << ","
					<< it->second[i].label << "," << it->second[i].difficult << ","
					<< it->second[i].score << "," << it->second[i].size << endl;
		}
	}
#endif

	// Retrieve all prior bboxes. It is same withing a batch since we assume all
	// images in a batch are of same dimension.
	// all prior bboxes
	vector<NormalizedBBox> priorBBoxes;
	// all prior variances
	vector<vector<float>> priorVariances;
	GetPriorBBoxes(priorData, this->numPriors, &priorBBoxes, &priorVariances);

#if 0//MULTIBOXLOSSLAYER_LOG
	cout << "GetPriorBBoxes()" << endl;
	for (int i = 0; i < priorBBoxes.size(); i++) {
		cout << priorBBoxes[i].xmin << "," << priorBBoxes[i].ymin << ","
				<< priorBBoxes[i].xmax << "," << priorBBoxes[i].ymax << ","
				<< priorBBoxes[i].label << "," << priorBBoxes[i].difficult << ","
				<< priorBBoxes[i].score << "," << priorBBoxes[i].size << endl;
		//priorBBoxes[i].print();
		//cout << "-----" << endl;
	}

	cout << "priorVariances ... " << endl;
	for (int i = 0; i < priorVariances.size(); i++) {
		cout << priorVariances[i][0] << "," << priorVariances[i][1] << ","
				<< priorVariances[i][2] << "," << priorVariances[i][3] << endl;
		//for (int j = 0; j < 4; j++) {
		//	cout << priorVariances[i][j] << ",";
			//cout << "-----" << endl;
		//}
		//cout << endl;
	}
#endif

	// Retrieve all predictions.
	// allLocPreds[0]: 첫번째 이미지의 prediction ...
	// allLocPreds[0][-1]: shareLocation==true인 경우 label은 오직 -1뿐,
	// 					   -1 key에 전체 prediction box list를 value로 contain.
	vector<LabelBBox> allLocPreds;
	GetLocPredictions(locData, this->num, this->numPriors, this->locClasses,
			shareLocation, &allLocPreds);

#if 0//MULTIBOXLOSSLAYER_LOG
	//this->_printOn();
	//this->_inputData[0]->print_data({}, false);
	//this->_printOff();
	cout << "GetLocPredictions()" << endl;
	for (int i = 0; i < allLocPreds.size(); i++) {

		/*
		LabelBBox& labelBBox = allLocPreds[i];
		for (LabelBBox::iterator it = labelBBox.begin(); it != labelBBox.end(); it++) {
			//cout << it->first << endl;
			//for (int j = 0; j < 10; j++) {
			//	it->second[j].print();
			//	cout << "-------" << endl;
			//}
		}
		*/
		map<int, vector<NormalizedBBox>>& locPred = allLocPreds[i];

		for (map<int, vector<NormalizedBBox>>::iterator it = locPred.begin(); it != locPred.end(); it++) {
			//cout << "for itemId: " << it->first << endl;
			for (int i = 0; i < it->second.size(); i++) {
				//cout << "\t" << i << endl;
				//it->second[i].print();
				cout << it->second[i].xmin << "," << it->second[i].ymin << ","
						<< it->second[i].xmax << "," << it->second[i].ymax << ","
						<< it->second[i].label << "," << it->second[i].difficult << ","
						<< it->second[i].score << "," << it->second[i].size << endl;
			}
		}
	}
#endif

	// Find matches between source bboxes and ground truth bboxes.
	// for each image in batch, (label : overlaps for each prior bbox)
	vector<map<int, vector<float>>> allMatchOverlaps;
	// allMatchOverlaps: batch내 각 이미지에 대한 (label=-1:prior bbox overlap)맵 리스트
	// allMatchIndices: batch내 각 이미지에 대한 최대 match gt index 맵 리스트 

	FindMatches(allLocPreds, allGtBBoxes, priorBBoxes, priorVariances, numClasses,
			shareLocation, matchType, overlapThreshold, usePriorForMatching,
			backgroundLabelId, codeType, encodeVarianceInTarget, ignoreCrossBoundaryBBox,
			&allMatchOverlaps, &this->allMatchIndices);

#if MULTIBOXLOSSLAYER_LOG
	cout << "FindMatches()" << endl;
	cout << "allMatchOverlaps: " << allMatchOverlaps.size() << endl;
	for (int i = 0; i < allMatchOverlaps.size(); i++) {
		map<int, vector<float>>& matchOverlaps = allMatchOverlaps[i];
		for (map<int, vector<float>>::iterator it = matchOverlaps.begin();
				it != matchOverlaps.end(); it++) {
			cout << it->first << ": " << it->second.size() << endl;
			for (int j = 0; j < it->second.size(); j++) {
				cout << it->second[j] << ",";
				if ((j+1) % 1000 == 0)
					cout << endl;
				//if (it->second[j] > 1e-6) {
					//cout << j << "\t\t" << it->second[j] << endl;
				//}
			}
			cout << endl;
		}
	}
	cout << "allMatchIndices: " << this->allMatchIndices.size() << endl;
	for (int i = 0; i < this->allMatchIndices.size(); i++) {
		int matchCount = 0;
		map<int, vector<int>>& matchIndices = this->allMatchIndices[i];
		for (map<int, vector<int>>::iterator it = matchIndices.begin();
				it != matchIndices.end(); it++) {
			cout << it->first << ": " << it->second.size() << endl;
			for (int j = 0; j < it->second.size(); j++) {
				cout << it->second[j] << ",";
				if ((j+1) % 1000 == 0)
					cout << endl;
				//if (it->second[j] > -1) {
					//cout << j << "\t\t" << it->second[j] << endl;
					//matchCount++;
				//}
			}
			cout << endl;
		}
		//cout << "match count: " << matchCount << endl;
	}
#endif

	this->numMatches = 0;
	int numNegs = 0;
	// Sample hard negative (and positive) examples based on mining type.
	// allNegInidices: batch내 이미지별 negative sample 리스트.
	MineHardExamples(*this->_inputData[1], allLocPreds, allGtBBoxes, priorBBoxes,
			priorVariances, allMatchOverlaps, numClasses, backgroundLabelId, usePriorForNMS,
			confLossType, miningType, locLossType, negPosRatio, negOverlap, codeType,
			encodeVarianceInTarget, nmsParam.nmsThreshold, nmsParam.topK, sampleSize,
			bpInside, usePriorForMatching, &this->numMatches, &numNegs,
			&this->allMatchIndices, &this->allNegIndices);

#if MULTIBOXLOSSLAYER_LOG
	// std::vector<std::vector<int>> allNegIndices;
	for (int i = 0; i < this->allNegIndices.size(); i++) {
		cout << i << "-----" << endl;
		for (int j = 0; j < this->allNegIndices[i].size(); j++) {
			cout << j << "\t\t" << this->allNegIndices[i][j] << endl;
		}
	}
	cout << "numNegs: " << numNegs << endl;
#endif

	// 
	if (this->numMatches >= 1) {
		// Form data to pass on to locLossLayer
		vector<uint32_t> locShape(4, 1);
		locShape[3] = this->numMatches * 4;
		this->locPred.reshape(locShape);		// {1, 1, 1, numMatches * 4}
		this->locGt.reshape(locShape);			// {1, 1, 1, numMatches * 4}
		Dtype* locPredData = this->locPred.mutable_host_data();
		Dtype* locGtData = this->locGt.mutable_host_data();
		EncodeLocPrediction(allLocPreds, allGtBBoxes, this->allMatchIndices, priorBBoxes,
				priorVariances, codeType, encodeVarianceInTarget, bpInside,
				usePriorForMatching, locPredData, locGtData);

#if MULTIBOXLOSSLAYER_LOG
		this->_printOn();
		this->locPred.print_data({}, false, -1);
		this->locGt.print_data({}, false, -1);
		this->_printOff();
#endif

		//this->locLossLayer->reshape();		->> feedforward에서 reshape하므로 삭제!

#if MULTIBOXLOSSLAYER_LOG
		this->locPred.print_shape();
		this->locGt.print_shape();
		this->locLoss.print_shape();
#endif

		this->locLossLayer->feedforward();
		//InnerLayerFunc::runForward(0, -1);
	} else {
		this->locLoss.mutable_host_data()[0] = 0;
	}


#if MULTIBOXLOSSLAYER_LOG
	this->_printOn();
	this->locLoss.print_data({}, false, -1);
	this->_printOff();
#endif


	// Form data to pass on to confLossLayer.
	if (doNegMining) {
		this->numConf = this->numMatches + numNegs;
	} else {
		this->numConf = this->num * this->numPriors;
	}

	if (this->numConf >= 1) {
		// Reshape the confidence data.
		vector<uint32_t> confShape(4, 1);
		if (confLossType == ConfLossType::SOFTMAX) {
			confShape[0] = this->numConf;
			this->confGt.reshape(confShape);	// {numConf,          1, 1, 1}
			confShape[1] = numClasses;
			this->confPred.reshape(confShape);	// {numConf, numClasses, 1, 1}
		} else {
			SASSERT(false, "Unknown confidence loss type.");
		}

		if (!doNegMining) {
			// Consider all scores.
			// Share data and grad with inputData[1]
			SASSERT0(this->confPred.getCount() == this->_inputData[1]->getCount());
			this->confPred.share_data(this->_inputData[1]);
		}
		Dtype* confPredData = this->confPred.mutable_host_data();
		Dtype* confGtData = this->confGt.mutable_host_data();
		soooa_set(this->confGt.getCount(), Dtype(backgroundLabelId), confGtData);
		EncodeConfPrediction(confData, this->num, this->numPriors, numClasses,
				backgroundLabelId, mapObjectToAgnostic, miningType,
				confLossType, this->allMatchIndices, this->allNegIndices, allGtBBoxes,
				confPredData, confGtData);

#if MULTIBOXLOSSLAYER_LOG
		this->_printOn();
		this->confGt.print_data({}, false, -1);
		this->confPred.print_data({}, false, -1);
		this->_printOff();
#endif

		//this->confLossLayer->reshape();	->> feedforward에서 reshap하므로 삭제!
		this->confLossLayer->feedforward();
		//InnerLayerFunc::runForward(0, -1);
	} else {
		this->confLoss.mutable_host_data()[0] = 0;
	}


	this->_outputData[0]->mutable_host_data()[0] = 0;
	const NormalizationMode normalizationMode = SLPROP(Loss, normalization);
	if (SLPROP_BASE(propDown)[0]) {
		Dtype normalizer = LossLayer<Dtype>::getNormalizer(normalizationMode, this->num,
				this->numPriors, this->numMatches);
		this->_outputData[0]->mutable_host_data()[0] +=
				locWeight * this->locLoss.host_data()[0] / normalizer;
	}
	if (SLPROP_BASE(propDown)[1]) {
		Dtype normalizer = LossLayer<Dtype>::getNormalizer(normalizationMode, this->num,
				this->numPriors, this->numMatches);
		this->_outputData[0]->mutable_host_data()[0] +=
				this->confLoss.host_data()[0] / normalizer;
	}
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::backpropagation() {
	SASSERT(!SLPROP_BASE(propDown)[2], "MultiBoxLossLayer cannot backpropagate to prior inputs.");
	SASSERT(!SLPROP_BASE(propDown)[3], "MultiBoxLossLayer cannot backpropagate to label inputs.");

	//this->_printOn();
	//this->_inputData[0]->print_data({}, false, -1);

	// Back propagate on location prediction.
	if (SLPROP_BASE(propDown)[0]) {
		Dtype* locInputGrad = this->_inputData[0]->mutable_host_grad();
		soooa_set(this->_inputData[0]->getCount(), Dtype(0), locInputGrad);
		//this->_inputData[0]->print_grad({}, false, -1);

		if (this->numMatches >= 1) {
			/*
			vector<bool> locPropDown;
			// Only back propagate on prediction, not ground truth.
			locPropDown.push_back(true);
			locPropDown.push_back(false);
			this->locLossLayer->_propDown = locPropDown;
			*/
			this->locLossLayer->backpropagation();
			//InnerLayerFunc::runBackward(0);


			// Scale gradient.
			const NormalizationMode normalizationMode = SLPROP(Loss, normalization);
			Dtype normalizer = LossLayer<Dtype>::getNormalizer(normalizationMode, this->num,
					this->numPriors, this->numMatches);
			Dtype lossWeight = this->_outputData[0]->host_grad()[0] / normalizer;
			soooa_gpu_scal(this->locPred.getCount(), lossWeight,
					this->locPred.mutable_device_grad());
			//this->locPred.print_grad({}, false, -1);

			// Copy gradient back to inputData[0]
			const Dtype* locPredGrad = this->locPred.host_grad();
			int count = 0;
			for (int i = 0; i < this->num; i++) {
				for (map<int, vector<int>>::iterator it = this->allMatchIndices[i].begin();
						it != this->allMatchIndices[i].end(); it++) {
					const int label = SLPROP(MultiBoxLoss, shareLocation) ? 0 : it->first;
					const vector<int>& matchIndex = it->second;
					for (int j = 0; j < matchIndex.size(); j++) {
						if (matchIndex[j] <= -1)  {
							continue;
						}
						// Copy the grad to the right place.
						int startIdx = this->locClasses * 4 * j + label * 4;
						soooa_copy<Dtype>(4, locPredGrad + count * 4, locInputGrad + startIdx);
						count++;
					}
				}
				locInputGrad += this->_inputData[0]->offset(1);
			}
			//this->_inputData[0]->print_grad({}, false, -1);
		}
	}
	//this->_printOff();

	// Back propagate on confidence prediction.
	if (SLPROP_BASE(propDown)[1]) {
		Dtype* confInputGrad = this->_inputData[1]->mutable_host_grad();
		soooa_set(this->_inputData[1]->getCount(), Dtype(0), confInputGrad);
		if (this->numConf >= 1) {
			/*
			vector<bool> confPropDown;
			// Only back propagate on prediction, not ground truth.
			confPropDown.push_back(true);
			confPropDown.push_back(false);
			this->confLossLayer->_propDown = confPropDown;
			*/
			this->confLossLayer->backpropagation();
			//InnerLayerFunc::runBackward(1);

			//this->_printOn();
			//this->locPred.print_grad({}, false, -1);
			//this->_printOff();

			// Scale gradient.
			const NormalizationMode normalizationMode = SLPROP(Loss, normalization);
			Dtype normalizer = LossLayer<Dtype>::getNormalizer(normalizationMode, this->num,
					this->numPriors, this->numMatches);
			Dtype lossWeight = this->_outputData[0]->host_grad()[0] / normalizer;
			soooa_gpu_scal(this->confPred.getCount(), lossWeight,
					this->confPred.mutable_device_grad());
			// Copy gradient back to inputData[1]
			const Dtype* confPredGrad = this->confPred.host_grad();
			if (SLPROP(MultiBoxLoss, doNegMining)) {
				int count = 0;
				for (int i = 0; i < this->num; i++) {
					// Copy matched (positive) bboxes scores' grad.
					const map<int, vector<int>>& matchIndices = this->allMatchIndices[i];
					for (map<int, vector<int>>::const_iterator it = matchIndices.begin();
							it != matchIndices.end(); it++) {
						const vector<int>& matchIndex = it->second;
						SASSERT0(matchIndex.size() == this->numPriors);
						for (int j = 0; j < this->numPriors; j++) {
							if (matchIndex[j] <= -1) {
								continue;
							}
							// Copy the grad to the right place.
							soooa_copy<Dtype>(SLPROP(MultiBoxLoss, numClasses),
									confPredGrad + count * SLPROP(MultiBoxLoss, numClasses),
									confInputGrad + j * SLPROP(MultiBoxLoss, numClasses));
							count++;
						}
					}
					// Copy negative bboxes scores' grad
					for (int n = 0; n < this->allNegIndices[i].size(); n++) {
						int j = this->allNegIndices[i][n];
						SASSERT0(j < this->numPriors);
						soooa_copy<Dtype>(SLPROP(MultiBoxLoss, numClasses),
								confPredGrad + count * SLPROP(MultiBoxLoss, numClasses),
								confInputGrad + j * SLPROP(MultiBoxLoss, numClasses));
						count++;
					}
					confInputGrad += this->_inputData[1]->offset(1);
				}
			} else {
				// The grad is already computed and stored.
				this->_inputData[1]->share_grad(&this->confPred);
			}
		}
	}

	// After backward, remove match statistics.
	this->allMatchIndices.clear();
	this->allNegIndices.clear();
}

template <typename Dtype>
Dtype MultiBoxLossLayer<Dtype>::cost() {
	return this->_outputData[0]->host_data()[0];
}



template <typename Dtype>
void MultiBoxLossLayer<Dtype>::setLayerData(Layer<Dtype>* layer, const std::string& type,
		std::vector<Data<Dtype>*>& dataVec) {

	/*
	if (type == "input") {
		for (int i = 0; i < dataVec.size(); i++) {
			layer->_inputs.push_back(dataVec[i]->_name);
			layer->_inputData.push_back(dataVec[i]);
		}
	} else if (type == "output") {
		for (int i = 0; i < dataVec.size(); i++) {
			layer->_outputs.push_back(dataVec[i]->_name);
			layer->_outputData.push_back(dataVec[i]);
		}
	} else {
		SASSERT(false, "invalid layer data type.");
	}
	*/
}



template <typename Dtype>
Layer<Dtype>* MultiBoxLossLayer<Dtype>::buildLocLossLayer(const LocLossType locLossType) {
	const float locWeight = SLPROP(MultiBoxLoss, locWeight);

	Layer<Dtype>* locLossLayer = NULL;

	vector<uint32_t> lossShape(4, 1);		// {1, 1, 1, 1}
	// Set up localization loss layer

	vector<Data<Dtype>*> locInputVec;
	vector<Data<Dtype>*> locOutputVec;
	// fake shape
	vector<uint32_t> locShape(4, 1);
	locShape[3] = 4;

	this->locPred.reshape(locShape);
	this->locGt.reshape(locShape);
	locInputVec.push_back(&this->locPred);
	locInputVec.push_back(&this->locGt);

	this->locLoss.reshape(lossShape);
	locOutputVec.push_back(&this->locLoss);

	switch(locLossType) {
	case SMOOTH_L1: {
		int innerSmoothl1lossId = SmoothL1LossLayer<Dtype>::INNER_ID;
		SmoothL1LossLayer<Dtype>::INNER_ID += 10;

		stringstream smoothl1lossDef;
		smoothl1lossDef << "{\n";
		smoothl1lossDef << "\t\"name\" : \"inner_smooth_l1_loss\",\n";
		smoothl1lossDef << "\t\"id\" : " << innerSmoothl1lossId << ",\n";
		smoothl1lossDef << "\t\"layer\" : \"SmoothL1Loss\",\n";
		smoothl1lossDef << "\t\"input\" : [\"locPred\", \"locGt\"],\n";
		smoothl1lossDef << "\t\"output\" : [\"locLoss\"],\n";
		smoothl1lossDef << "\t\"lossWeight\" : " << std::fixed << std::setprecision(5) << locWeight << ",\n";
		smoothl1lossDef << "\t\"propDown\" : [true, false]\n";
		//smoothl1lossDef << "\t\"\" : \"\",\n";
		smoothl1lossDef << "}\n";

		//cout << smoothl1lossDef.str() << endl;


		_SmoothL1LossPropLayer* prop = NULL;
		SNEW(prop, _SmoothL1LossPropLayer);
		SASSUME0(prop != NULL);
		Json::Reader reader;
		Json::Value layer;
		reader.parse(smoothl1lossDef, layer);

		vector<string> keys = layer.getMemberNames();
		string layerType = layer["layer"].asCString();

		for (int j = 0; j < keys.size(); j++) {
			string key = keys[j];
			Json::Value val = layer[key.c_str()];
			if (strcmp(key.c_str(), "layer") == 0) continue;
			if (strcmp(key.c_str(), "innerLayer") == 0) continue;

			PlanParser::setPropValue(val, true, layerType, key,  (void*)prop);
		}
		locLossLayer = NULL;
		SNEW(locLossLayer, SmoothL1LossLayer<Dtype>, prop);
		SASSUME0(locLossLayer != NULL);
		SDELETE(prop);
	}
		break;
	default:
		SASSERT(false, "Unknown localization loss type.");
		break;
	}

	//setLayerData(this->locLossLayer, "input", this->locInputVec);
	//setLayerData(this->locLossLayer, "output", this->locOutputVec);
	locLossLayer->_inputData = locInputVec;
	locLossLayer->_outputData = locOutputVec;

	return locLossLayer;
}

template <typename Dtype>
Layer<Dtype>* MultiBoxLossLayer<Dtype>::buildConfLossLayer(const ConfLossType confLossType) {
	Layer<Dtype>* confLossLayer = NULL;

	const int numClasses = SLPROP(MultiBoxLoss, numClasses);
	vector<uint32_t> lossShape(4, 1);		// {1, 1, 1, 1}

	vector<Data<Dtype>*> confInputVec;
	vector<Data<Dtype>*> confOutputVec;

	// Set up confidence loss layer.
	confInputVec.push_back(&this->confPred);
	confInputVec.push_back(&this->confGt);
	this->confLoss.reshape(lossShape);
	confOutputVec.push_back(&this->confLoss);

	switch(confLossType) {
	case SOFTMAX: {
		SASSERT(SLPROP(MultiBoxLoss, backgroundLabelId) >= 0,
				"backgroundLabelId should be within [0, numClasses) for Softmax.");
		SASSERT(SLPROP(MultiBoxLoss, backgroundLabelId) < numClasses,
				"backgroundLabelId should be within [0, numClasses) for Softmax.");

		int innerSoftmaxWithLossId = SoftmaxWithLossLayer<Dtype>::INNER_ID;
		SoftmaxWithLossLayer<Dtype>::INNER_ID += 10;

		stringstream softmaxWithLossDef;
		softmaxWithLossDef << "{\n";
		softmaxWithLossDef << "\t\"name\" : \"inner_softmax_with_loss\",\n";
		softmaxWithLossDef << "\t\"id\" : " << innerSoftmaxWithLossId << ",\n";
		softmaxWithLossDef << "\t\"layer\" : \"SoftmaxWithLoss\",\n";
		softmaxWithLossDef << "\t\"input\" : [\"confPred\", \"confGt\"],\n";
		softmaxWithLossDef << "\t\"output\" : [\"confLoss\"],\n";
		softmaxWithLossDef << "\t\"lossWeight\" : 1.0,\n";
		//softmaxWithLossDef << "\t\"normalization\" : NormalizationMode::NoNormalization,\n";
		softmaxWithLossDef << "\t\"normalization\" : \"NoNormalization\",\n";
		softmaxWithLossDef << "\t\"softmaxAxis\" : 1,\n";
		softmaxWithLossDef << "\t\"propDown\" : [true, false]\n";
		//softmaxWithLossDef << "\t\"\" : \"\",\n";
		softmaxWithLossDef << "}\n";

		//cout << softmaxWithLossDef.str() << endl;

		_SoftmaxWithLossPropLayer* prop = NULL;
		SNEW(prop, _SoftmaxWithLossPropLayer);
		SASSUME0(prop != NULL);
		Json::Reader reader;
		Json::Value layer;
		reader.parse(softmaxWithLossDef, layer);

		vector<string> keys = layer.getMemberNames();
		string layerType = layer["layer"].asCString();
		for (int j = 0; j < keys.size(); j++) {
			string key = keys[j];
			Json::Value val = layer[key.c_str()];
			if (strcmp(key.c_str(), "layer") == 0) continue;
			if (strcmp(key.c_str(), "innerLayer") == 0) continue;

			PlanParser::setPropValue(val, true, layerType, key,  (void*)prop);
		}

		confLossLayer = NULL;
		SNEW(confLossLayer, SoftmaxWithLossLayer<Dtype>, prop);
		SASSUME0(confLossLayer != NULL);
		SDELETE(prop);
	}
		break;
	default:
		SASSERT(false, "Unknown confidence loss type.");
		break;
	}

	//setLayerData(this->confLossLayer, "input", this->confInputVec);
	//setLayerData(this->confLossLayer, "output", this->confOutputVec);
	confLossLayer->_inputData = confInputVec;
	confLossLayer->_outputData = confOutputVec;

	return confLossLayer;
}








/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* MultiBoxLossLayer<Dtype>::initLayer() {
	MultiBoxLossLayer* layer = NULL;
	SNEW(layer, MultiBoxLossLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void MultiBoxLossLayer<Dtype>::destroyLayer(void* instancePtr) {
    MultiBoxLossLayer<Dtype>* layer = (MultiBoxLossLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void MultiBoxLossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 4);
	} else {
		SASSERT0(index < 1);
	}

    MultiBoxLossLayer<Dtype>* layer = (MultiBoxLossLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool MultiBoxLossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    MultiBoxLossLayer<Dtype>* layer = (MultiBoxLossLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void MultiBoxLossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	MultiBoxLossLayer<Dtype>* layer = (MultiBoxLossLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void MultiBoxLossLayer<Dtype>::backwardTensor(void* instancePtr) {
	MultiBoxLossLayer<Dtype>* layer = (MultiBoxLossLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void MultiBoxLossLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool MultiBoxLossLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 4)
        return false;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 1;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t MultiBoxLossLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    // inputShape외의 것으로 동적으로 결정 ... 
    return 0UL;
}

template class MultiBoxLossLayer<float>;
