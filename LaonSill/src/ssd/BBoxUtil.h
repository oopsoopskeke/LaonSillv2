/*
 * BBoxUtil.h
 *
 *  Created on: Apr 28, 2017
 *      Author: jkim
 */

#ifndef BBOXUTIL_H_
#define BBOXUTIL_H_

#include <map>
#include <vector>

#include "ssd_common.h"
#include "Data.h"
#include "Datum.h"
#include "EnumDef.h"
#include "LayerPropParam.h"

typedef std::map<int, std::vector<NormalizedBBox>> LabelBBox;


bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);



template <typename T>
bool SortScorePairAscend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2);

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2);


template <typename Dtype>
void GetGroundTruth(const Dtype* gtData, const int numGt, const int backgroundLabelId,
		const bool useDifficultGt, std::map<int, std::vector<NormalizedBBox>>* allGtBBoxes);

template <typename Dtype>
void GetGroundTruth(const Dtype* gtData, const int numGt, const int backgroundLabelId,
		const bool useDifficultGt, std::map<int, LabelBBox>* allGtBBoxes);




template <typename Dtype>
void GetPriorBBoxes(const Dtype* priorData, const int numPriors,
		std::vector<NormalizedBBox>* priorBBoxes,
		std::vector<std::vector<float>>* priorVariances);

template <typename Dtype>
void GetLocPredictions(const Dtype* locData, const int num, const int numPredsPerClass,
		const int numLocClasses, const bool shareLocation, std::vector<LabelBBox>* locPreds);

void FindMatches(const std::vector<LabelBBox>& allLocPreds,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const int numClasses, const bool shareLocation, const MatchType matchType,
		const float overlapThreshold, const bool usePriorForMatching,
		const int backgroundLabelId, const CodeType codeType,
		const bool encodeVarianceInTarget, const bool ignoreCrossBoundaryBBox,
		std::vector<std::map<int, std::vector<float>>>* allMatchOverlaps,
		std::vector<std::map<int, std::vector<int>>>* allMatchIndices);

void DecodeBBoxes(const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const CodeType codeType, const bool varianceEncodedInTarget,
		const bool clipBBox, const std::vector<NormalizedBBox>& bboxes,
		std::vector<NormalizedBBox>* decodeBBoxes);

void DecodeBBox(const NormalizedBBox& priorBBox, const std::vector<float>& priorVariances,
		const CodeType codeType, const bool varianceEncodedInTarget,
		const bool clipBBox, const NormalizedBBox& bbox, NormalizedBBox* decodeBBox);

void MatchBBox(const std::vector<NormalizedBBox>& gtBBoxes,
		const std::vector<NormalizedBBox>& predBBoxes, const int label,
		const MatchType matchType, const float overlapThreshold,
		const bool ignoreCrossBoundaryBBox, std::vector<int>* matchIndices,
		std::vector<float>* matchOverlaps);

float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);

template <typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized = true);

void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clipBBox);

bool IsCrossBoundaryBBox(const NormalizedBBox& bbox);

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		NormalizedBBox* intersectBBox);

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		const bool normalized = true);

template <typename Dtype>
Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2);





template <typename Dtype>
void EncodeLocPrediction(const std::vector<LabelBBox>& allLocPreds,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const CodeType codeType, const bool encodeVarianceInTarget,
		const bool bpInside, const bool usePriorForMatching,
		Dtype* locPredData, Dtype* locGtData);

void EncodeBBox(const NormalizedBBox& priorBBox, const std::vector<float>& priorVariance,
		const CodeType codeType, const bool encodeVarianceInTarget,
		const NormalizedBBox& bbox, NormalizedBBox* encodeBBox);

template <typename Dtype>
void MineHardExamples(Data<Dtype>& confData,
		const std::vector<LabelBBox>& allLocPreds,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const std::vector<std::map<int, std::vector<float>>>& allMatchOverlaps,
		const int numClasses, const int backgroundLabelId, const bool usePriorForNms,
		const ConfLossType confLossType, const MiningType miningType,
		const LocLossType locLossType, const float negPosRatio, const float negOverlap,
		const CodeType codeType, const bool encodeVarianceInTarget, const float nmsThresh,
		const int topK, const int sampleSize, const bool bpInside,
		const bool usePriorForMatching, int* numMatches, int* numNegs,
		std::vector<std::map<int, std::vector<int>>>* allMatchIndices,
		std::vector<std::vector<int>>* allNegIndices);

template <typename Dtype>
void EncodeConfPrediction(const Dtype* confData, const int num, const int numPriors,
		const int numClasses, const int backgroundLabelId, const bool mapObjectToAgnostic,
		const MiningType miningType, const ConfLossType confLossType,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::vector<std::vector<int>>& allNegIndices,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		Dtype* confPredData, Dtype* confGtData);

template <typename Dtype>
void GetConfidenceScores(const Dtype* confData, const int num, const int numPredsPerClass,
		const int numClasses, std::vector<std::map<int, std::vector<float>>>* confPreds);

// Decode all bboxes in a batch.
void DecodeBBoxesAll(const std::vector<LabelBBox>& allLocPreds,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const int num, const bool shareLocation,
		const int numLocClasses, const int backgroundLabelId,
		const CodeType codeType, const bool varianceEncodedInTarget,
		const bool clip, std::vector<LabelBBox>* allDecodeBBoxes);

void ApplyNMSFast(const std::vector<NormalizedBBox>& bboxes, const std::vector<float>& scores,
		const float scoreThreshold, const float nmsThreshold, const float eta,
		const int topK, std::vector<int>* indices);

template <typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
		const float scoreThreshold, const float nmsThreshold, const float eta,
		const int topK, std::vector<int>* indices);

void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
		NormalizedBBox* scaleBBox);

void OutputBBox(const NormalizedBBox& bbox, const std::pair<int, int>& imgSize,
		const bool hasResize, NormalizedBBox* outBBox);

std::vector<cv::Scalar> GetColors(const int n);

template <typename Dtype>
void VisualizeBBox(const std::vector<cv::Mat>& images, Data<Dtype>* detections,
		const float threshold, const std::vector<cv::Scalar>& colors,
		const std::map<int, std::string>& labelToDisplayName, const std::string& saveFile);

template <typename Dtype>
void GetDetectionResults(const Dtype* detData, const int numDet, const int backgroundLabelId,
		std::map<int, LabelBBox>* allDetections);


template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads, const Dtype* locData, const Dtype* priorData,
		const CodeType codeType, const bool varianceEncodedInTarget,
		const int numPriors, const bool shareLocation,
		const int numLocClasses, const int backgroundLabelId,
		const bool clipBBox, Dtype* bboxData);

template <typename Dtype>
void PermuteDataGPU(const int nthreads,
		const Dtype* data, const int numClasses, const int numData,
		const int numDim, Dtype* newData);


bool MeetEmitConstraint(const NormalizedBBox& srcBBox, const NormalizedBBox& bbox,
		const EmitConstraint& emitConstraint);

// Compute the coverage of bbox1 by bbox2.
float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Project bbox onto the coordinate system defined by srcBBox.
bool ProjectBBox(const NormalizedBBox& srcBBox, const NormalizedBBox& bbox,
		NormalizedBBox* projBBox);

// Locate bbox in the coordinate system defined by srcBBox.
void LocateBBox(const NormalizedBBox& srcBBox, const NormalizedBBox& bbox,
		NormalizedBBox* locBBox);

// Extrapolate the transformed bbox if heightScale and widthScale is
// explicitly provided, and it is only effective for FIT_SMALL_SIZE case.
void ExtrapolateBBox(const ResizeParam& param, const int height, const int width,
		const NormalizedBBox& cropBBox, NormalizedBBox* bbox);

void CumSum(const std::vector<std::pair<float, int>>& pairs, std::vector<int>* cumSum);

void ComputeAP(const std::vector<std::pair<float, int>>& tp, const int numPos,
		const std::vector<std::pair<float, int>>& fp, const std::string apVersion,
		std::vector<float>* prec, std::vector<float>* rec, float* ap);


#endif /* BBOXUTIL_H_ */
