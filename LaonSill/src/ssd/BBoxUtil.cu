#include <set>
#include <cmath>
#include <cfloat>
#include <csignal>
#include <algorithm>
#include <boost/iterator/counting_iterator.hpp>

#include "BBoxUtil.h"
#include "MathFunctions.h"
#include "SysLog.h"
#include "StdOutLog.h"


//using namespace std;



bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
	return bbox1.score < bbox2.score;
}

bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
	return bbox1.score > bbox2.score;
}



template <typename T>
bool SortScorePairAscend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2) {
	return pair1.first < pair2.first;
}

template bool SortScorePairAscend(const std::pair<float, int>& pair1,
		const std::pair<float, int>& pair2);
template bool SortScorePairAscend(const std::pair<float, std::pair<int, int>>& pair1,
		const std::pair<float, std::pair<int, int>>& pair2);



template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2) {
	return pair1.first > pair2.first;
}

template bool SortScorePairDescend(const std::pair<float, int>& pair1,
		const std::pair<float, int>& pair2);
template bool SortScorePairDescend(const std::pair<float, std::pair<int, int>>& pair1,
		const std::pair<float, std::pair<int, int>>& pair2);




template <typename Dtype>
__device__ Dtype Min(const Dtype x, const Dtype y) {
	return x < y ? x : y;
}


/**
 * Max 이름이 충돌나 일단 임시로 MaxDevice로 변경
 * (Max 이름이 어디에 정의되어 있는지 모르겠음)
 */
template <typename Dtype>
__device__ Dtype MaxDevice(const Dtype x, const Dtype y) {
	return x > y ? x : y;
}

template <typename Dtype>
__device__ void ClipBBoxGPU(const Dtype* bbox, Dtype* clip_bbox) {
	for (int i = 0; i < 4; ++i) {
		clip_bbox[i] = MaxDevice(Min(bbox[i], Dtype(1.)), Dtype(0.));
	}
}

template __device__ void ClipBBoxGPU(const float* bbox, float* clip_bbox);





template <typename Dtype>
void GetGroundTruth(const Dtype* gtData, const int numGt, const int backgroundLabelId,
		const bool useDifficultGt, std::map<int, std::vector<NormalizedBBox>>* allGtBboxes) {
	allGtBboxes->clear();
	for (int i = 0; i < numGt; i++) {
		int startIdx = i * 8;
		int itemId = gtData[startIdx];
		if (itemId == -1) {
			continue;
		}
		int label = gtData[startIdx + 1];
		SASSERT(backgroundLabelId != label,
				"Found background label in the dataset\nbackground label id: %d\nlabel:%d",
				backgroundLabelId, label);
		bool difficult = static_cast<bool>(gtData[startIdx + 7]);
		if (!useDifficultGt && difficult) {
			// Skip reading difficult ground truth
			continue;
		}
		NormalizedBBox bbox;
		bbox.label = label;
		bbox.xmin = gtData[startIdx + 3];
		bbox.ymin = gtData[startIdx + 4];
		bbox.xmax = gtData[startIdx + 5];
		bbox.ymax = gtData[startIdx + 6];
		bbox.difficult = difficult;

		float bboxSize = BBoxSize(bbox);
		bbox.size = bboxSize;
		(*allGtBboxes)[itemId].push_back(bbox);
	}
}

template void GetGroundTruth(const float* gtData, const int numGt,
		const int backgroundLabelId, const bool useDifficultGt,
		std::map<int, std::vector<NormalizedBBox>>* allGtBboxes);


template <typename Dtype>
void GetGroundTruth(const Dtype* gtData, const int numGt, const int backgroundLabelId,
		const bool useDifficultGt, std::map<int, LabelBBox>* allGtBBoxes) {

	allGtBBoxes->clear();
	for (int i = 0; i < numGt; i++) {
		int startIdx = i * 8;
		int itemId = gtData[startIdx];
		if (itemId == -1) {
            continue;
		}
		NormalizedBBox bbox;
		int label = gtData[startIdx + 1];
		if (backgroundLabelId == label) {
			SASSERT(backgroundLabelId != label, "Found background label in the dataset.");
		}
		bool difficult = static_cast<bool>(gtData[startIdx + 7]);
		if (!useDifficultGt && difficult) {
			// Skip reading difficult ground truth.
			continue;
		}
		bbox.xmin = gtData[startIdx + 3];
		bbox.ymin = gtData[startIdx + 4];
		bbox.xmax = gtData[startIdx + 5];
		bbox.ymax = gtData[startIdx + 6];
		bbox.difficult = difficult;
		float bboxSize = BBoxSize(bbox);
		bbox.size = bboxSize;
		(*allGtBBoxes)[itemId][label].push_back(bbox);
	}
}

template void GetGroundTruth(const float* gtData, const int numGt,
		const int backgroundLabelId, const bool useDifficultGt,
		std::map<int, LabelBBox>* allGtBBoxes);












template <typename Dtype>
void GetPriorBBoxes(const Dtype* priorData, const int numPriors,
		std::vector<NormalizedBBox>* priorBBoxes,
		std::vector<std::vector<float>>* priorVariances) {

	priorBBoxes->clear();
	priorVariances->clear();
	for (int i = 0; i < numPriors; i++) {
		int startIdx = i * 4;
		NormalizedBBox bbox;
		bbox.xmin = priorData[startIdx];
		bbox.ymin = priorData[startIdx + 1];
		bbox.xmax = priorData[startIdx + 2];
		bbox.ymax = priorData[startIdx + 3];

		float bboxSize = BBoxSize(bbox);
		bbox.size = bboxSize;
		priorBBoxes->push_back(bbox);
	}

	for (int i = 0; i < numPriors; i++) {
		int startIdx = (numPriors + i) * 4;
		std::vector<float> var;
		for (int j = 0; j < 4; j++) {
			var.push_back(priorData[startIdx + j]);
		}
		priorVariances->push_back(var);
	}
}

template void GetPriorBBoxes(const float* priorData, const int numPriors,
		std::vector<NormalizedBBox>* priorBBoxes, std::vector<std::vector<float>>* priorVariances);




template <typename Dtype>
void GetLocPredictions(const Dtype* locData, const int num, const int numPredsPerClass,
		const int numLocClasses, const bool shareLocation, std::vector<LabelBBox>* locPreds) {
	locPreds->clear();
	if (shareLocation) {
		SASSERT0(numLocClasses == 1);
	}
	locPreds->resize(num);
	for (int i = 0; i < num; i++) {
		LabelBBox& labelBBox = (*locPreds)[i];
		for (int p = 0; p < numPredsPerClass; p++) {
			int startIdx = p * numLocClasses * 4;
			for (int c = 0; c < numLocClasses; c++) {
				int label = shareLocation ? -1 : c;
				if (labelBBox.find(label) == labelBBox.end()) {
					labelBBox[label].resize(numPredsPerClass);
				}
				labelBBox[label][p].xmin = locData[startIdx + c * 4];
				labelBBox[label][p].ymin = locData[startIdx + c * 4 + 1];
				labelBBox[label][p].xmax = locData[startIdx + c * 4 + 2];
				labelBBox[label][p].ymax = locData[startIdx + c * 4 + 3];
			}
		}
		locData += numPredsPerClass * numLocClasses * 4;
	}
}

template void GetLocPredictions(const float* locData, const int num,
		const int numPredsPerClass, const int numLocClasses, const bool shareLocation,
		std::vector<LabelBBox>* locPreds);



/*
 * allLocPreds: batch내의 이미지별 prediction
 * allGtBBoxes: batch내의 이미지별 gt bboxes
 * prioBBoxes:  전체 scale에서의 prior boxes
 * priorVariances: 전체 scale에서의 prior variances
 *
 * allMatchOverlaps
 */
void FindMatches(const std::vector<LabelBBox>& allLocPreds,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const int numClasses, const bool shareLocation, const MatchType matchType,
		const float overlapThreshold, const bool usePriorForMatching,
		const int backgroundLabelId, const CodeType codeType,
		const bool encodeVarianceInTarget, const bool ignoreCrossBoundaryBBox,
		std::vector<std::map<int, std::vector<float>>>* allMatchOverlaps,
		std::vector<std::map<int, std::vector<int>>>* allMatchIndices) {

	SASSERT(numClasses > 0, "numClasses should not be less than 1.");
	const int locClasses = shareLocation ? 1 : numClasses;

	// Find the matches.
	// num은 batch size를 의미
	int num = allLocPreds.size();
	// batch내의 각 이미지에 대해
	for (int i = 0; i < num; i++) {
		// (label : pred bbox별 최대 overlap gt index)
		std::map<int, std::vector<int>> matchIndices;
		std::map<int, std::vector<float>> matchOverlpas;
		// Check if there is ground truth for current image.
		if (allGtBBoxes.find(i) == allGtBBoxes.end()) {
			// There is no gt for current image. All predictions are negative.
			allMatchIndices->push_back(matchIndices);
			allMatchOverlaps->push_back(matchOverlpas);
			continue;
		}
		// Find match between predictions and ground truth.
		// gtBBoxes: batch내 이미지 한장의 전체 gt boxes 
		const std::vector<NormalizedBBox>& gtBBoxes = allGtBBoxes.find(i)->second;
		// 현재 사용하지 않음. usePriorForMatching = true.
		if (!usePriorForMatching) {
			for (int c = 0; c < locClasses; c++) {
				int label = shareLocation ? -1 : c;
				if (!shareLocation && label == backgroundLabelId) {
					// Ignore background loc predictions.
					continue;
				}
				// Decode the prediction into bbox first.
				std::vector<NormalizedBBox> locBBoxes;
				bool clipBBox = false;
				DecodeBBoxes(priorBBoxes, priorVariances, codeType, encodeVarianceInTarget,
						clipBBox, allLocPreds[i].find(label)->second, &locBBoxes);
				MatchBBox(gtBBoxes, locBBoxes, label, matchType, overlapThreshold,
						ignoreCrossBoundaryBBox, &matchIndices[label], &matchOverlpas[label]);
			}
		} else {
			// Use prior bboxes to match against all ground truth.
			std::vector<int> tempMatchIndices;
			std::vector<float> tempMatchOverlaps;
			const int label = -1;
			// batch내 현재 이미지에 대해 gt와 prior box간의 match 정보 


			// for break point ...
			MatchBBox(gtBBoxes, priorBBoxes, label, matchType, overlapThreshold,
					ignoreCrossBoundaryBBox, &tempMatchIndices, &tempMatchOverlaps);
			// label == -1 케이스, 별도로 class별 처리를 하지 않음  
			if (shareLocation) {
				matchIndices[label] = tempMatchIndices;
				matchOverlpas[label] = tempMatchOverlaps;
			} else {
				// Get ground truth label for each ground truth bbox.
				std::vector<int> gtLabels;
				for (int g = 0; g < gtBBoxes.size(); g++) {
					gtLabels.push_back(gtBBoxes[g].label);
				}
				// Distribute the matching results to different locClass.
				for (int c = 0; c < locClasses; c++) {
					if (c == backgroundLabelId) {
						// Ignore background loc predictions.
						continue;
					}
					matchIndices[c].resize(tempMatchIndices.size(), -1);
					matchOverlpas[c] = tempMatchOverlaps;
					for (int m = 0; m < tempMatchIndices.size(); m++) {
						if (tempMatchIndices[m] > -1) {
							const int gtIdx = tempMatchIndices[m];
							SASSERT0(gtIdx < gtLabels.size());
							if (c == gtLabels[gtIdx]) {
								matchIndices[c][m] = gtIdx;
							}
						}
					}
				}
			}
		}
		allMatchIndices->push_back(matchIndices);
		allMatchOverlaps->push_back(matchOverlpas);
	}
}

void DecodeBBoxes(const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const CodeType codeType, const bool varianceEncodedInTarget,
		const bool clipBBox, const std::vector<NormalizedBBox>& bboxes,
		std::vector<NormalizedBBox>* decodeBBoxes) {
	SASSERT0(priorBBoxes.size() == priorVariances.size());
	SASSERT0(priorBBoxes.size() == bboxes.size());
	int numBBoxes = priorBBoxes.size();
	if (numBBoxes >= 1) {
		SASSERT0(priorVariances[0].size() == 4);
	}
	decodeBBoxes->clear();
	for (int i = 0; i < numBBoxes; i++) {
		NormalizedBBox decodeBBox;
		DecodeBBox(priorBBoxes[i], priorVariances[i], codeType, varianceEncodedInTarget,
				clipBBox, bboxes[i], &decodeBBox);
		decodeBBoxes->push_back(decodeBBox);
	}
}

void DecodeBBox(const NormalizedBBox& priorBBox, const std::vector<float>& priorVariances,
		const CodeType codeType, const bool varianceEncodedInTarget,
		const bool clipBBox, const NormalizedBBox& bbox, NormalizedBBox* decodeBBox) {

	if (codeType == CodeType::CORNER) {
		if (varianceEncodedInTarget) {
			// variance is encoded intarget, we simply need to add the offset predictions.
			decodeBBox->xmin = priorBBox.xmin + bbox.xmin;
			decodeBBox->ymin = priorBBox.ymin + bbox.ymin;
			decodeBBox->xmax = priorBBox.xmax + bbox.xmax;
			decodeBBox->ymax = priorBBox.ymax + bbox.ymax;
		} else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decodeBBox->xmin = priorBBox.xmin + priorVariances[0] * bbox.xmin;
			decodeBBox->ymin = priorBBox.ymin + priorVariances[1] * bbox.ymin;
			decodeBBox->xmax = priorBBox.xmax + priorVariances[2] * bbox.xmax;
			decodeBBox->ymax = priorBBox.ymax + priorVariances[3] * bbox.ymax;
		}
	} else if (codeType == CodeType::CENTER_SIZE) {
		float priorWidth = priorBBox.xmax - priorBBox.xmin;
		SASSERT0(priorWidth > 0);
		float priorHeight = priorBBox.ymax - priorBBox.ymin;
		SASSERT0(priorHeight > 0);
		float priorCenterX = (priorBBox.xmin + priorBBox.xmax) / 2.f;
		float priorCenterY = (priorBBox.ymin + priorBBox.ymax) / 2.f;

		float decodeBBoxCenterX;
		float decodeBBoxCenterY;
		float decodeBBoxWidth;
		float decodeBBoxHeight;

		if (varianceEncodedInTarget) {
			// variance is encoded in target, we simply need to restore the offset
			// predictions.
			decodeBBoxCenterX = bbox.xmin * priorWidth + priorCenterX;
			decodeBBoxCenterY = bbox.ymin * priorHeight + priorCenterY;
			decodeBBoxWidth = std::exp(bbox.xmax) * priorWidth;
			decodeBBoxHeight = std::exp(bbox.ymax) * priorHeight;
		} else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decodeBBoxCenterX = priorVariances[0] * bbox.xmin * priorWidth + priorCenterX;
			decodeBBoxCenterY = priorVariances[1] * bbox.ymin * priorHeight + priorCenterY;
			decodeBBoxWidth = std::exp(priorVariances[2] * bbox.xmax) * priorWidth;
			decodeBBoxHeight = std::exp(priorVariances[3] * bbox.ymax) * priorHeight;
		}

		decodeBBox->xmin = decodeBBoxCenterX - decodeBBoxWidth / 2.f;
		decodeBBox->ymin = decodeBBoxCenterY - decodeBBoxHeight / 2.f;
		decodeBBox->xmax = decodeBBoxCenterX + decodeBBoxWidth / 2.f;
		decodeBBox->ymax = decodeBBoxCenterY + decodeBBoxHeight / 2.f;
	} else if (codeType == CodeType::CORNER_SIZE) {
		float priorWidth = priorBBox.xmax - priorBBox.xmin;
		SASSERT0(priorWidth > 0);
		float priorHeight = priorBBox.ymax - priorBBox.ymin;
		SASSERT0(priorHeight > 0);
		if (varianceEncodedInTarget) {
			// variance is encoded in target, we simply need to add the offset predictions.
			decodeBBox->xmin = priorBBox.xmin + bbox.xmin * priorWidth;
			decodeBBox->ymin = priorBBox.ymin + bbox.ymin * priorHeight;
			decodeBBox->xmax = priorBBox.xmax + bbox.xmax * priorWidth;
			decodeBBox->ymax = priorBBox.ymax + bbox.ymax * priorHeight;
		} else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decodeBBox->xmin = priorBBox.xmin + priorVariances[0] * bbox.xmin * priorWidth;
			decodeBBox->ymin = priorBBox.ymin + priorVariances[1] * bbox.ymin * priorHeight;
			decodeBBox->xmax = priorBBox.xmax + priorVariances[2] * bbox.xmax * priorWidth;
			decodeBBox->ymax = priorBBox.ymax + priorVariances[3] * bbox.ymax * priorHeight;
		}
	} else {
		SASSERT(false, "Unknown LocLossType: %s", codeType);
	}
	float bboxSize = BBoxSize(*decodeBBox);
	decodeBBox->size = bboxSize;
	if (clipBBox) {
		ClipBBox(*decodeBBox, decodeBBox);
	}
}

// gtBBoxes: 현재 이미지의 gt boxes
// predBBoxes: 전체 prior boxes
// label
// matchIndices: 전체 predBBox별 최대 match gt box index
// matchOverlaps: 전체 predBBox별 최대 match gt box와의 overlap,
//                eps보다 커야하며 그 중 최대값을 저장
// 
void MatchBBox(const std::vector<NormalizedBBox>& gtBBoxes,
		const std::vector<NormalizedBBox>& predBBoxes, const int label,
		const MatchType matchType, const float overlapThreshold,
		const bool ignoreCrossBoundaryBBox, std::vector<int>* matchIndices,
		std::vector<float>* matchOverlaps) {
	int numPred = predBBoxes.size();
	matchIndices->clear();
	matchIndices->resize(numPred, -1);
	matchOverlaps->clear();
	matchOverlaps->resize(numPred, 0.);

	int numGt = 0;
	std::vector<int> gtIndices;
	if (label == -1) {
		// label -1 means comparing against all ground truth.
		numGt = gtBBoxes.size();
		for (int i = 0; i < numGt; i++) {
			gtIndices.push_back(i);
		}
	} else {
		// Count number of ground truth boxes which has the desired label.
		for (int i = 0; i < gtBBoxes.size(); i++) {
			if (gtBBoxes[i].label == label) {
				numGt++;
				gtIndices.push_back(i);
			}
		}
	}
	if (numGt == 0) {
		return;
	}

	// Store the positive overlap between predictions and ground truth.
	// map 구조 -> (pred box index : (gt index : overlap))
	// 모든 pred bbox와 gt bbox간의 overlap, 1e-6보다 큰 경우에 map에 저장.
	// 따라서 overlap 값에 따라 map에 해당 entry가 존재 하지 않을 수 있음.
	std::map<int, std::map<int, float>> overlaps;
	for (int i = 0; i < numPred; i++) {
		// ignoreCrossBoundarBBox = false
		if (ignoreCrossBoundaryBBox && IsCrossBoundaryBBox(predBBoxes[i])) {
			(*matchIndices)[i] = -2;
			continue;
		}
		for (int j = 0; j < numGt; j++) {
			// i번째 predBbox와 j번째 gtBox간의 overlap
			float overlap = JaccardOverlap(predBBoxes[i], gtBBoxes[gtIndices[j]]);
			if (overlap > 1e-6) {
				// i번째 predBBox에 대한 최대 overlap 갱신
				(*matchOverlaps)[i] = std::max((*matchOverlaps)[i], overlap);
				// i번째 predBBox와 j번째 gtBox간의 overlap 저장 
				// overlap이 eps이상인 경우에 대해서만 저장 
				overlaps[i][j] = overlap;

				//cout << "overlap of pred bbox " << i << " and gt bbox " << j << ": " <<
				//		overlap << endl;
			}
		}
	}

	/*
	if (numGt == 16) {
		for (std::map<int, std::map<int, float>>::iterator itr = overlaps.begin();
				itr != overlaps.end(); itr++) {
			std::cout << "predbbox#" << itr->first << ",";
			for (int i = 0; i < numGt; i++) {
				if (itr->second.find(i) != itr->second.end()) {
					std::cout << itr->second[i] << ",";
				} else {
					std::cout << 0 << ",";
				}
			}
			std::cout << std::endl;
		}
		exit(1);
	}
	*/

	/*
	if (numGt == 16) {
		for (int j = 0; j < numGt; j++) {
			if (overlaps[1277].find(j) != overlaps[1277].end()) {
				std::cout << "overlaps[1277][" << j << "]=" << overlaps[1277][j] << std::endl;
			}
		}
		for (int j = 0; j < numGt; j++) {
			if (overlaps[1334].find(j) != overlaps[1334].end()) {
				std::cout << "overlaps[1334][" << j << "]=" << overlaps[1334][j] << std::endl;
			}
		}
	}
	*/

	// Bipartite matching
	// gt 갯수만큼 수행
	// gt index pool
	std::vector<int> gtPool;
	for (int i = 0; i < numGt; i++) {
		gtPool.push_back(i);
	}


	/*
	for (int i = 0; i < gtPool.size(); i++) {
		std::cout << "gtPool[" << i << "]: " << gtPool[i] << std::endl;
	}
	*/





	// gt bbox들의 최대 matching pred bbox를 찾음.
	while (gtPool.size() > 0) {
		// Find the most overlapped gt and coresponding predictions.
		int maxIdx = -1;
		int maxGtIdx = -1;
		float maxOverlap = -1;
		// 각 predBox에 대한 최대 overlap의 gt 찾기 
		for (std::map<int, std::map<int, float>>::iterator it = overlaps.begin();
				it != overlaps.end(); it++) {
			// predBBox index
			int i= it->first;
			//std::cout << "for i=" << i << std::endl;
			if ((*matchIndices)[i] != -1) {
				// The prediction already has matched ground truth or is ignored.
				continue;
			}
			for (int p = 0; p < gtPool.size(); p++) {

				int j = gtPool[p];
				// overlap < eps인 경우 map에 추가되지 않았음. 
				if (it->second.find(j) == it->second.end()) {
					// No overlap between the i-th predcition and j-th ground truth.
					continue;
				}
				// Find the maximu overlapped pari.
				if (it->second[j] > maxOverlap) {
					// If the predction has not been matched to any ground truth,
					// and the overlap is larger than maximum overlap, update.

					/*
					if (numGt == 16 && i == 1277) {
						std::cout << "break for j =" << j << std::endl;
						std::cout << "matchIndices[1277]=" << (*matchIndices)[1277] << std::endl;
						std::cout << "matchOverlaps[1277]=" << (*matchOverlaps)[1277] << std::endl;
					}

					if (numGt == 16 && i == 1334) {
						std::cout << "break for j =" << j << std::endl;
						std::cout << "matchIndices[1334]=" << (*matchIndices)[1334] << std::endl;
						std::cout << "matchOverlaps[1334]=" << (*matchOverlaps)[1334] << std::endl;
					}
					*/
					//std::cout << "checking for i=" << i << ", j=" << j << std::endl;

					maxIdx = i;
					maxGtIdx = j;
					maxOverlap = it->second[j];
				}
			}
		}
		// 현재 gt bbox에 대해 matching되는 pred bbox를 찾지 못함.
		if (maxIdx == -1) {
			// Cannot find good match.
			break;
		} else {
			// 이전에 다른 경로를 통해 matching되지 않은 pred bbox여야 함.
			SASSERT0((*matchIndices)[maxIdx] == -1);
			(*matchIndices)[maxIdx] = gtIndices[maxGtIdx];
			(*matchOverlaps)[maxIdx] = maxOverlap;
			// Erase the ground truth.
			// 전체 predBBox와 특정 gt가 최대 overlap으로 찾아진 경우
			// 해당 gt는 다른 predBBox와 match하지 않도록 삭제
			gtPool.erase(std::find(gtPool.begin(), gtPool.end(), maxGtIdx));
			/*
			std::cout << "maxIdx=" << maxIdx << ", gtIdx=" << gtIndices[maxGtIdx] <<
					", maxGtIdx=" << maxGtIdx << ", maxOverlap=" << maxOverlap << std::endl;
					*/
		}
	}

	/*
	if (numGt == 16) {
		std::cout << "matchIndices[1277]=" << (*matchIndices)[1277] << std::endl;
		std::cout << "matchOverlaps[1277]=" << (*matchOverlaps)[1277] << std::endl;

		std::cout << "matchIndices[1334]=" << (*matchIndices)[1334] << std::endl;
		std::cout << "matchOverlaps[1334]=" << (*matchOverlaps)[1334] << std::endl;
		exit(1);
	}
	*/



	switch (matchType) {
	case BIPARTITE:
		// Already done.
		break;
	case PER_PREDICTION:
		// gt 최대 overlap pred bbox뿐 아니라
		// 최대는 아니어도 pred bbox 자신과 overlap되는 gt가 있는 경우
		// 해당 index와 overlap을 추가 업데이트.

		// Get most overlapped for the rest prediction bboxes.
		// overlap: (pred bbox idx : (gt idx : overlap)) 구조의 맵
		for (std::map<int, std::map<int, float>>::iterator it = overlaps.begin();
				it != overlaps.end(); it++) {
			int i = it->first;

			// gt와 최대 overlap으로 match된 pred bbox
			if ((*matchIndices)[i] != -1) {
				// Thre predction already has matched ground truth or is ignored.
				continue;
			}
			int maxGtIdx = -1;
			float maxOverlap = -1;
			for (int j = 0; j < numGt; j++) {
				if (it->second.find(j) == it->second.end()) {
					// No overlap between the i-th predction and j-th ground truth.
					continue;
				}
				// Find the maximum overlapped pair
				float overlap = it->second[j];
				if (overlap >= overlapThreshold && overlap > maxOverlap) {
					// If the predcition has not been matched to any ground truth,
					// and the overlap is larger than maximum overlap, update.
					maxGtIdx = j;
					maxOverlap = overlap;
				}
			}
			if (maxGtIdx != -1) {
				// Found a matched ground truth.
				SASSERT0((*matchIndices)[i] == -1);
				(*matchIndices)[i] = gtIndices[maxGtIdx];
				(*matchOverlaps)[i] = maxOverlap;
			}
		}
		break;
	default:
		SASSERT(false, "Unknown matching type: %s", matchType);
		break;
	}

	return;
}



float BBoxSize(const NormalizedBBox& bbox, const bool normalized) {
	if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin) {
		// If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		return 0;
	} else {
		if (bbox.size > 0) {
			return bbox.size;
		} else {
			float width = bbox.xmax - bbox.xmin;
			float height = bbox.ymax - bbox.ymin;
			if (normalized) {
				return width * height;
			} else {
				// If bbox is not within range [0, 1]
				return (width + 1) * (height + 1);
			}
		}
	}
}

template <typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized) {
	if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
		// If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		return Dtype(0.);
	} else {
		const Dtype width = bbox[2] - bbox[0];
		const Dtype height = bbox[3] - bbox[1];
		if (normalized) {
			return width * height;
		} else {
			// If bbox is not within range [0, 1].
			return (width + 1) * (height + 1);
		}
	}
}

template float BBoxSize(const float* bbox, const bool normalized);

void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clipBBox) {
	clipBBox->xmin = std::max(std::min(bbox.xmin, 1.f), 0.f);
	clipBBox->ymin = std::max(std::min(bbox.ymin, 1.f), 0.f);
	clipBBox->xmax = std::max(std::min(bbox.xmax, 1.f), 0.f);
	clipBBox->ymax = std::max(std::min(bbox.ymax, 1.f), 0.f);
	clipBBox->size = BBoxSize(*clipBBox);
	clipBBox->difficult = bbox.difficult;
}


bool IsCrossBoundaryBBox(const NormalizedBBox& bbox) {
	return bbox.xmin < 0 || bbox.xmin > 1 ||
			bbox.ymin < 0 || bbox.ymin > 1 ||
			bbox.xmax < 0 || bbox.xmax > 1 ||
			bbox.ymax < 0 || bbox.ymax > 1;
}

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		const bool normalized) {
	NormalizedBBox intersectBBox;
	IntersectBBox(bbox1, bbox2, &intersectBBox);
	float intersectWidth;
	float intersectHeight;

	if (normalized) {
		intersectWidth = intersectBBox.xmax - intersectBBox.xmin;
		intersectHeight = intersectBBox.ymax - intersectBBox.ymin;
	} else {
		intersectWidth = intersectBBox.xmax - intersectBBox.xmin + 1;
		intersectHeight = intersectBBox.ymax - intersectBBox.ymin + 1;
	}

	if (intersectWidth > 0 && intersectHeight > 0) {
		float intersectSize = intersectWidth * intersectHeight;
		float bbox1Size = BBoxSize(bbox1);
		float bbox2Size = BBoxSize(bbox2);
		return intersectSize / (bbox1Size + bbox2Size - intersectSize);
	} else {
		return 0.;
	}
}

template <typename Dtype>
Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2) {
	if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
		bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
		return Dtype(0.);
	} else {
		const Dtype interXMin = std::max(bbox1[0], bbox2[0]);
		const Dtype interYMin = std::max(bbox1[1], bbox2[1]);
		const Dtype interXMax = std::min(bbox1[2], bbox2[2]);
		const Dtype interYMax = std::min(bbox1[3], bbox2[3]);

		const Dtype interWidth = interXMax - interXMin;
		const Dtype interHeight = interYMax - interYMin;
		const Dtype interSize = interWidth * interHeight;

		const Dtype bbox1Size = BBoxSize(bbox1);
		const Dtype bbox2Size = BBoxSize(bbox2);

		return interSize / (bbox1Size + bbox2Size - interSize);
	}
}

template float JaccardOverlap(const float* bbox1, const float* bbox2);



void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		NormalizedBBox* intersectBBox) {
	if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin ||
			bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) {
		// Return [0, 0, 0, 0] if there is no intersection.
		intersectBBox->xmin = 0;
		intersectBBox->ymin = 0;
		intersectBBox->xmax = 0;
		intersectBBox->ymax = 0;
	} else {
		intersectBBox->xmin = std::max(bbox1.xmin, bbox2.xmin);
		intersectBBox->ymin = std::max(bbox1.ymin, bbox2.ymin);
		intersectBBox->xmax = std::min(bbox1.xmax, bbox2.xmax);
		intersectBBox->ymax = std::min(bbox1.ymax, bbox2.ymax);
	}
}

inline bool IsEligibleMining(const MiningType miningType, const int matchIdx,
		const float matchOverlap, const float negOverlap) {
	if (miningType == MiningType::MAX_NEGATIVE) {
		return matchIdx == -1 && matchOverlap < negOverlap;
	} else if (miningType == MiningType::HARD_EXAMPLE) {
		return true;
	} else {
		return false;
	}
}

int CountNumMatches(const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const int num) {
	int numMatches = 0;
	for (int i = 0; i < num; i++) {
		const std::map<int, std::vector<int>>& matchIndices = allMatchIndices[i];
		for (std::map<int, std::vector<int>>::const_iterator it = matchIndices.begin();
				it != matchIndices.end(); it++) {
			const std::vector<int>& matchIndex = it->second;
			for (int m = 0; m < matchIndex.size(); m++) {
				if (matchIndex[m] > -1) {
					numMatches++;
				}
			}
		}
	}
	return numMatches;
}

void GetTopKScoreIndex(const std::vector<float>& scores, const std::vector<int>& indices,
		const int topK, std::vector<std::pair<float, int>>* scoreIndexVec) {
	SASSERT0(scores.size() == indices.size());

	// Generate index score pairs.
	for (int i = 0; i < scores.size(); i++) {
		scoreIndexVec->push_back(std::make_pair(scores[i], indices[i]));
	}

	// Sort the score pair according to the scores in descending order
	std::stable_sort(scoreIndexVec->begin(), scoreIndexVec->end(), SortScorePairDescend<int>);

	// Keep topK scores if needed.
	if (topK > -1 && topK < scoreIndexVec->size()) {
		scoreIndexVec->resize(topK);
	}
}




void ApplyNMS(const std::vector<NormalizedBBox>& bboxes, const std::vector<float>& scores,
		const float threshold, const int topK, const bool reuseOverlaps,
		std::map<int, std::map<int, float>>* overlaps, std::vector<int>* indices) {
	// Sanity check.
	SASSERT(bboxes.size() == scores.size(), "bboxes and scores have different size.");

	// Get topK scores (with coreesponding indices).
	std::vector<int> idx(boost::counting_iterator<int>(0),
			boost::counting_iterator<int>(scores.size()));
	std::vector<std::pair<float, int>> scoreIndexVec;
	GetTopKScoreIndex(scores, idx, topK, &scoreIndexVec);

	// Do nms.
	indices->clear();
	while (scoreIndexVec.size() != 0) {
		// Get the current highest score box.
		int bestIdx = scoreIndexVec.front().second;
		const NormalizedBBox& bestBBox = bboxes[bestIdx];
		if (BBoxSize(bestBBox) < 1e-5) {
			// Erase small box.
			scoreIndexVec.erase(scoreIndexVec.begin());
			continue;
		}
		indices->push_back(bestIdx);
		// Erase the best box.
		scoreIndexVec.erase(scoreIndexVec.begin());

		if (topK > -1 && indices->size() >= topK) {
			// Stop if finding enough bboxes for nms.
			break;
		}

		// Compute overlap between bestBBox and other remaining bboxes.
		// Remove a bbox if the overlap with bestBBox is larger than nmsThreshold.
		for (std::vector<std::pair<float, int>>::iterator it = scoreIndexVec.begin();
				it != scoreIndexVec.end(); ) {
			int curIdx = it->second;
			const NormalizedBBox& curBBox = bboxes[curIdx];
			if (BBoxSize(curBBox) < 1e-5) {
				// Erase small box.
				it = scoreIndexVec.erase(it);
				continue;
			}
			float curOverlap = 0.;
			if (reuseOverlaps) {
				if (overlaps->find(bestIdx) != overlaps->end() &&
						overlaps->find(bestIdx)->second.find(curIdx) !=
								(*overlaps)[bestIdx].end()) {
					// Use the computed overlap.
					curOverlap = (*overlaps)[bestIdx][curIdx];
				} else if (overlaps->find(curIdx) != overlaps->end() &&
						overlaps->find(curIdx)->second.find(bestIdx) !=
								(*overlaps)[curIdx].end()) {
					// Use the computed overlap.
					curOverlap = (*overlaps)[curIdx][bestIdx];
				} else {
					curOverlap = JaccardOverlap(bestBBox, curBBox);
					// Store the overlap for future use.
					(*overlaps)[bestIdx][curIdx] = curOverlap;
				}
			} else {
				curOverlap = JaccardOverlap(bestBBox, curBBox);
			}

			// Remove it if necessary
			if (curOverlap > threshold) {
				it = scoreIndexVec.erase(it);
			} else {
				it++;
			}
		}
	}
}

void ApplyNMS(const std::vector<NormalizedBBox>& bboxes, const std::vector<float>& scores,
		const float threshold, const int topK, std::vector<int>* indices) {
	bool reuseOverlap = false;
	std::map<int, std::map<int, float>> overlaps;
	ApplyNMS(bboxes, scores, threshold, topK, reuseOverlap, &overlaps, indices);
}

void ApplyNMS(const bool* overlapped, const int num, std::vector<int>* indices) {
	std::vector<int> indexVec(boost::counting_iterator<int>(0),
			boost::counting_iterator<int>(num));
	// Do nms.
	indices->clear();
	while (indexVec.size() != 0) {
		// Get the current highest score box.
		int bestIdx = indexVec.front();
		indices->push_back(bestIdx);
		// Erase the best box.
		indexVec.erase(indexVec.begin());

		for (std::vector<int>::iterator it = indexVec.begin(); it != indexVec.end(); ) {
			int curIdx = *it;

			// Remove it if necessary
			if (overlapped[bestIdx * num + curIdx]) {
				it = indexVec.erase(it);
			} else {
				it++;
			}
		}
	}
}

template <typename Dtype>
__global__ void ComputeConfLossKernel(const int nthreads, const Dtype* conf_data,
		const int num_preds_per_class, const int num_classes, const int loss_type,
		const Dtype* match_data, Dtype* conf_loss_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int label = match_data[index];
		int num = index / num_preds_per_class;
		int p = index % num_preds_per_class;
		int start_idx = (num * num_preds_per_class + p) * num_classes;
		Dtype loss = 0;

		if (loss_type == 0) {
			// Compute softmax probability.
			Dtype prob = conf_data[start_idx + label];
			loss = -log(MaxDevice(prob, Dtype(FLT_MIN)));
		} else if (loss_type == 1) {
			int target = 0;
			for (int c = 0; c < num_classes; ++c) {
				if (c == label) {
					target = 1;
				} else {
					target = 0;
				}
				Dtype input = conf_data[start_idx + c];
				loss -= input * (target - (input >= 0)) -
						log(1 + exp(input - 2 * input * (input >= 0)));
			}
		}
		conf_loss_data[index] = loss;
	}
}






template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels, const int spatial_dim,
		const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, num * spatial_dim) {
		int n = index / spatial_dim;
		int s = index % spatial_dim;
		Dtype maxval = -FLT_MAX;
		for (int c = 0; c < channels; ++c) {
			maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
		}
		out[index] = maxval;
	}
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count, const int num, const int channels,
		const int spatial_dim, const Dtype* channel_data, const Dtype* channel_max,
		Dtype* data) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / channels / spatial_dim;
		int s = index % spatial_dim;
		data[index] = channel_data[index] - channel_max[n * spatial_dim + s];
	}
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, count) {
		out[index] = exp(data[index]);
	}
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim,
		const Dtype* data, Dtype* channel_sum) {
	CUDA_KERNEL_LOOP(index, num * spatial_dim) {
		int n = index / spatial_dim;
		int s = index % spatial_dim;
		Dtype sum = 0;
		for (int c = 0; c < channels; ++c) {
			sum += data[(n * channels + c) * spatial_dim + s];
		}
		channel_sum[index] = sum;
	}
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count, const int num, const int channels,
		const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / channels / spatial_dim;
		int s = index % spatial_dim;
		data[index] /= channel_sum[n * spatial_dim + s];
	}
}


template <typename Dtype>
void SoftMaxGPU(const Dtype* data, const int outerNum, const int channels, const int innerNum,
		Dtype* prob) {
	std::vector<uint32_t> shape(4, 1);
	shape[0] = outerNum;
	shape[1] = channels;
	shape[2] = innerNum;
	Data<Dtype> scale("scale", shape);
	Dtype* scaleData = scale.mutable_device_data();
	int count = outerNum * channels * innerNum;
	// We need to subtract the max to avoid numerical issues, compute the exp,
	// and the normalize.

	// compute max
	kernel_channel_max<Dtype><<<SOOOA_GET_BLOCKS(outerNum * innerNum),
			SOOOA_CUDA_NUM_THREADS>>>(outerNum, channels, innerNum, data, scaleData);

	// subtract
	kernel_channel_subtract<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
			count, outerNum, channels, innerNum, data, scaleData, prob);

	// exponentiate
	kernel_exp<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(count, prob, prob);

	// sum after exp
	kernel_channel_sum<Dtype><<<SOOOA_GET_BLOCKS(outerNum * innerNum),
	      SOOOA_CUDA_NUM_THREADS>>>(outerNum, channels, innerNum, prob, scaleData);

	// divide
	kernel_channel_div<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
			count, outerNum, channels, innerNum, scaleData, prob);
}




// confData: (numBatches, numPriors * numClasses)의 shape. confidence 예측값.
// num:
// ...jjjjj
// allConfLoss: 이미지별 각 prior box에 대한 conf loss 리스트.
template <typename Dtype>
void ComputeConfLossGPU(Data<Dtype>& confData, const int num,
		const int numPredsPerClass, const int numClasses, const int backgroundLabelId,
		const ConfLossType lossType,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		std::vector<std::vector<float>>* allConfLoss) {
	SASSERT0(backgroundLabelId < numClasses);
	// match: prior box와 matching된 gt box의 label (정답 label)을 담음 
	// batch내 전체 이미지의 각 prior box에 대한 정답값을 담음
	// matching된 gt box가 없는 경우 0이 기본값임. 0이 matching되지 않음을 의미.
	Data<Dtype> match("match", {uint32_t(num), uint32_t(numPredsPerClass), 1, 1});
	Dtype* matchData = match.mutable_host_data();
	for (int i = 0; i < num; i++) {
		const std::map<int, std::vector<int>>& matchIndices = allMatchIndices[i];
		// p번째 prior box에 대해 
		for (int p = 0; p < numPredsPerClass; p++) {
			// Get the label index.
			int label = backgroundLabelId;
			// shareLocation == true인 케이스에서 key에 해당하는 값은 -1 하나 뿐. 
			// 실제로 iteration을 돈다기 보다는 한 번만 실행됨.
			for (std::map<int, std::vector<int>>::const_iterator it = matchIndices.begin();
					it != matchIndices.end(); it++) {
				const std::vector<int>& matchIndex = it->second;
				SASSERT0(matchIndex.size() == numPredsPerClass);
				// p번째 prior box에 대해 matching된 gt box가 있는지 확인
				if (matchIndex[p] > -1) {
					SASSERT0(allGtBBoxes.find(i) != allGtBBoxes.end());
					const std::vector<NormalizedBBox>& gtBBoxes = allGtBBoxes.find(i)->second;
					SASSERT0(matchIndex[p] < gtBBoxes.size());
					label = gtBBoxes[matchIndex[p]].label;
					SASSERT0(label >= 0);
					SASSERT0(label != backgroundLabelId);
					SASSERT0(label < numClasses);
					// A prior can only be matched to one gt bbox.
					break;
				}
			}
			// p번째 prior box에 대한 정답 label을 matchData에 저장 
			matchData[i * numPredsPerClass + p] = label;
		}
	}
	// Get probability data.
	// 네트워크가 예측한 각 prior box별, class별 probability.
	const Dtype* confGpuData = confData.device_data();
	Data<Dtype> prob("prob");
	prob.reshapeLike(&confData);
	// lossType이 'SOFTMAX'인 경우 raw conf data대신 softmax를 취한 값을 conf data로 사용.
	if (lossType == ConfLossType::SOFTMAX) {
		Dtype* probData = prob.mutable_device_data();
		SoftMaxGPU(confData.device_data(), num * numPredsPerClass, numClasses, 1, probData);
		confGpuData = prob.device_data();
	}
	// Compute the loss.
	Data<Dtype> confLoss("confLoss", {uint32_t(num), uint32_t(numPredsPerClass), 1, 1});
	Dtype* confLossData = confLoss.mutable_device_data();
	const int numThreads = num * numPredsPerClass;

	int intLossType = 0;
	if (lossType == ConfLossType::SOFTMAX) intLossType = 0;
	else if (lossType == ConfLossType::LOGISTIC) intLossType = 1;
	else SASSERT0(false);

	// matching된 gt box가 있는 경우 해당 label에 대해서만 loss계산.
	// matching된 gt box가 없는 경우 background class에 대해서 loss계산. 
	ComputeConfLossKernel<Dtype><<<SOOOA_GET_BLOCKS(numThreads), SOOOA_CUDA_NUM_THREADS>>>(
			numThreads, confGpuData, numPredsPerClass, numClasses, intLossType,
			match.device_data(), confLossData);

	// Save the loss.
	allConfLoss->clear();
	const Dtype* lossData = confLoss.host_data();
	for (int i = 0; i < num; i++) {
		std::vector<float> confLoss(lossData, lossData + numPredsPerClass);
		allConfLoss->push_back(confLoss);
		lossData += numPredsPerClass;
	}
}

template void ComputeConfLossGPU(Data<float>& confData, const int num,
		const int numPredsPerClass, const int numClasses, const int backgroundLabelId,
		const ConfLossType lossType,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		std::vector<std::vector<float>>* allConfLoss);


// confData: mbox_conf
// allLocPreds: 이미지별 location prediction 리스트  
// allGtBBoxes: 이미지별 gt box 리스트 맵
// priorBBoxes: 전체 prior bboxes 리스트 
// priorVariances
// allMatchOverlaps:
// ...
// allMatchIndices:
// allNegIndices: Negative Example indices로 추정 
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
		const CodeType codeType, const bool encodeVarianceInTarget,
		const float nmsThreshold, const int topK, const int sampleSize, const bool bpInside,
		const bool usePriorForMatching, int* numMatches, int* numNegs,
		std::vector<std::map<int, std::vector<int>>>* allMatchIndices,
		std::vector<std::vector<int>>* allNegIndices) {

	// num images in batch 
	int num = allLocPreds.size();
	// batch내 전체 이미지에 대해서 gt에 대해 match된 prior box의 갯수. (label > -1) 
	*numMatches = CountNumMatches(*allMatchIndices, num);
	*numNegs = 0;
	int numPriors = priorBBoxes.size();
	SASSERT0(numPriors == priorVariances.size());

	SASSERT(numClasses >= 1, "numClasses should not be less than 1.");
	if (miningType == MiningType::MINING_NONE) {
		return;
	}

	bool hasNmsParam = true;
	if (topK <= 0) {
		hasNmsParam = false;
	}

	// Compute confidence losses based on matching results.
	// batch내 이미지별 prior box 전체에 대한 loss 리스트.
	// [num][num_priors]
	std::vector<std::vector<float>> allConfLoss;
	ComputeConfLossGPU(confData, num, numPriors, numClasses, backgroundLabelId, confLossType,
			*allMatchIndices, allGtBBoxes, &allConfLoss);

	// batch내 이미지별 prior box 전체에 대한 loss 리스트.
	// [num][num_priors]
	std::vector<std::vector<float>> allLocLoss;
	if (miningType == MiningType::HARD_EXAMPLE) {
		// Compute localization losses based on matching results.
		Data<Dtype> locPred("locPred");
		Data<Dtype> locGt("locGt");
		if (*numMatches != 0) {
			std::vector<uint32_t> locShape(4, 1);
			locShape[3] = *numMatches * 4;
			locPred.reshape(locShape);
			locGt.reshape(locShape);
			Dtype* locPredData = locPred.mutable_host_data();
			Dtype* locGtData = locGt.mutable_host_data();
			EncodeLocPrediction(allLocPreds, allGtBBoxes, *allMatchIndices, priorBBoxes,
					priorVariances, codeType, encodeVarianceInTarget, bpInside,
					usePriorForMatching, locPredData, locGtData);
		}
		ComputeLocLoss(locPred, locGt, *allMatchIndices, num, numPriors, locLossType,
				&allLocLoss);
	} else {
		// No localization loss.
		// 전체 loc loss를 0.f로 초기화.
		for (int i = 0; i < num; i++) {
			std::vector<float> locLoss(numPriors, 0.f);
			allLocLoss.push_back(locLoss);
		}
	}

	for (int i = 0; i < num; i++) {
		std::map<int, std::vector<int>>& matchIndices = (*allMatchIndices)[i];
		const std::map<int, std::vector<float>>& matchOverlaps = allMatchOverlaps[i];
		// loc + conf loss.
		const std::vector<float>& confLoss = allConfLoss[i];
		const std::vector<float>& locLoss = allLocLoss[i];
		std::vector<float> loss;
		std::transform(confLoss.begin(), confLoss.end(), locLoss.begin(),
				std::back_inserter(loss), std::plus<float>());
		// Pick negatives or hard examples based on loss.
		std::set<int> selIndices;
		std::vector<int> negIndices;
		for (std::map<int, std::vector<int>>::iterator it = matchIndices.begin();
				it != matchIndices.end(); it++) {
			const int label = it->first;
			int numSel = 0;
			// Get potential indices and loss pairs.
			// negative sample의 overlap, index pair 리스트
			std::vector<std::pair<float, int>> lossIndices;
			for (int m = 0; m < matchIndices[label].size(); m++) {
				// matchIdx == -1 && matchOverlap < negOverlap일때 true
				if (IsEligibleMining(miningType, matchIndices[label][m],
						matchOverlaps.find(label)->second[m], negOverlap)) {
					lossIndices.push_back(std::make_pair(loss[m], m));
					numSel++;
				}
			}
			if (miningType == MiningType::MAX_NEGATIVE) {
				int numPos = 0;
				for (int m = 0; m < matchIndices[label].size(); m++) {
					if (matchIndices[label][m] > -1) {
						numPos++;
					}
				}
				numSel = std::min(static_cast<int>(numPos * negPosRatio), numSel);
			} else if (miningType == MiningType::HARD_EXAMPLE) {
				SASSERT0(sampleSize > 0);
				numSel = std::min(sampleSize, numSel);
			}

			// XXX: nmsThreshold 테스트 통과하는지 확인
			// Select samples.
			if (hasNmsParam && nmsThreshold > 0) {
				// Do nms before selecting samples.
				std::vector<float> selLoss;
				std::vector<NormalizedBBox> selBBoxes;
				if (usePriorForNms) {
					for (int m = 0; m < matchIndices[label].size(); m++) {
						if (IsEligibleMining(miningType, matchIndices[label][m],
								matchOverlaps.find(label)->second[m], negOverlap)) {
							selLoss.push_back(loss[m]);
							selBBoxes.push_back(priorBBoxes[m]);
						}
					}
				} else {
					// Decode the prediction into bbox first.
					std::vector<NormalizedBBox> locBBoxes;
					bool clipBBox = false;
					DecodeBBoxes(priorBBoxes, priorVariances, codeType,
							encodeVarianceInTarget, clipBBox,
							allLocPreds[i].find(label)->second, &locBBoxes);
					for (int m  = 0; m < matchIndices[label].size(); m++) {
						if (IsEligibleMining(miningType, matchIndices[label][m],
								matchOverlaps.find(label)->second[m], negOverlap)) {
							selLoss.push_back(loss[m]);
							selBBoxes.push_back(locBBoxes[m]);
						}
					}
				}
				// Do non-maximum suppression based on the loss.
				std::vector<int> nmsIndices;
				ApplyNMS(selBBoxes, selLoss, nmsThreshold, topK, &nmsIndices);
				if (nmsIndices.size() < numSel) {
					STDOUT_LOG("not enought sample after nms: %d", nmsIndices.size());
				}
				// Pick top example indices after nms.
				numSel = std::min(static_cast<int>(nmsIndices.size()), numSel);
				for (int n = 0; n < numSel; n++) {
					selIndices.insert(lossIndices[nmsIndices[n]].second);
				}
			} else {
				// Pick top example indices based on loss.
				std::sort(lossIndices.begin(), lossIndices.end(),
						SortScorePairDescend<int>);
				for (int n = 0; n < numSel; n++) {
					selIndices.insert(lossIndices[n].second);
				}
			}
			// Update the matchIndices and select negIndices.
			// negative prior box indices 생성.
			for (int m = 0; m < matchIndices[label].size(); m++) {
				if (matchIndices[label][m] > -1) {
					if (miningType == MiningType::HARD_EXAMPLE &&
							selIndices.find(m) == selIndices.end()) {
						matchIndices[label][m] = -1;
						*numMatches -= 1;
					}
				} else if (matchIndices[label][m] == -1) {
					if (selIndices.find(m) != selIndices.end()) {
						negIndices.push_back(m);
						*numNegs += 1;
					}
				}
			}
		}
		allNegIndices->push_back(negIndices);
	}
}

template void MineHardExamples(Data<float>& confData,
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


// allLocPreds: 
template <typename Dtype>
void EncodeLocPrediction(const std::vector<LabelBBox>& allLocPreds,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const CodeType codeType, const bool encodeVarianceInTarget,
		const bool bpInside, const bool usePriorForMatching,
		Dtype* locPredData, Dtype* locGtData) {

	int num = allLocPreds.size();
	int count = 0;

	for (int i = 0; i < num; i++) {
		for (std::map<int, std::vector<int>>::const_iterator it = allMatchIndices[i].begin();
				it != allMatchIndices[i].end(); it++) {
			const int label = it->first;
			const std::vector<int>& matchIndex = it->second;
			SASSERT0(allLocPreds[i].find(label) != allLocPreds[i].end());
			const std::vector<NormalizedBBox>& locPred = allLocPreds[i].find(label)->second;
			for (int j = 0; j < matchIndex.size(); j++) {
				if (matchIndex[j] <= -1) {
					continue;
				}
				// Store encoded ground truth.
				const int gtIdx = matchIndex[j];
				SASSERT0(allGtBBoxes.find(i) != allGtBBoxes.end());
				SASSERT0(gtIdx < allGtBBoxes.find(i)->second.size());
				const NormalizedBBox& gtBBox = allGtBBoxes.find(i)->second[gtIdx];
				NormalizedBBox gtEncode;
				SASSERT0(j < priorBBoxes.size());


				EncodeBBox(priorBBoxes[j], priorVariances[j], codeType,
						encodeVarianceInTarget, gtBBox, &gtEncode);

				locGtData[count * 4] = gtEncode.xmin;
				locGtData[count * 4 + 1] = gtEncode.ymin;
				locGtData[count * 4 + 2] = gtEncode.xmax;
				locGtData[count * 4 + 3] = gtEncode.ymax;
				// Store location prediction.
				SASSERT0(j < locPred.size());
				if (bpInside) {
					NormalizedBBox matchBBox = priorBBoxes[j];
					if (!usePriorForMatching) {
						const bool clipBBox = false;
						DecodeBBox(priorBBoxes[j], priorVariances[j], codeType,
								encodeVarianceInTarget, clipBBox, locPred[j], &matchBBox);
					}
					// When a dimension of matchBBox is outside of image region, use
					// gtEncode to simulate zero gradient.
					locPredData[count * 4] = (matchBBox.xmin < 0 || matchBBox.xmin > 1) ?
							gtEncode.xmin : locPred[j].ymin;
					locPredData[count * 4 + 1] = (matchBBox.ymin < 0 || matchBBox.ymin > 1) ?
							gtEncode.ymin : locPred[j].ymin;
					locPredData[count * 4 + 2] = (matchBBox.xmax < 0 || matchBBox.xmax > 1) ?
							gtEncode.xmax : locPred[j].xmax;
					locPredData[count * 4 + 3] = (matchBBox.ymax < 0 || matchBBox.ymax > 1) ?
							gtEncode.ymax : locPred[j].ymax;
				} else {
					locPredData[count * 4] = locPred[j].xmin;
					locPredData[count * 4 + 1] = locPred[j].ymin;
					locPredData[count * 4 + 2] = locPred[j].xmax;
					locPredData[count * 4 + 3] = locPred[j].ymax;
				}
				if (encodeVarianceInTarget) {
					for (int k = 0; k < 4; k++) {
						SASSERT0(priorVariances[j][k] > 0);
						locPredData[count * 4 + k] /= priorVariances[j][k];
						locGtData[count * 4 + k] /= priorVariances[j][k];
					}
				}
				count++;
			}
		}
	}
}

template void EncodeLocPrediction(const std::vector<LabelBBox>& allLocPreds,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const CodeType codeType, const bool encodeVarianceInTarget,
		const bool bpInside, const bool usePriorForMatching,
		float* locPredData, float* locGtData);


void EncodeBBox(const NormalizedBBox& priorBBox, const std::vector<float>& priorVariance,
		const CodeType codeType, const bool encodeVarianceInTarget,
		const NormalizedBBox& bbox, NormalizedBBox* encodeBBox) {
	if (codeType == CodeType::CORNER) {
		if (encodeVarianceInTarget) {
			encodeBBox->xmin = bbox.xmin - priorBBox.xmin;
			encodeBBox->ymin = bbox.ymin - priorBBox.ymin;
			encodeBBox->xmax = bbox.xmax - priorBBox.xmax;
			encodeBBox->ymax = bbox.ymax - priorBBox.ymax;
		} else {
			// Encode variance in bbox.
			SASSERT0(priorVariance.size() == 4);
			for (int i = 0; i < priorVariance.size(); i++) {
				SASSERT0(priorVariance[i] > 0);
			}
			encodeBBox->xmin = (bbox.xmin - priorBBox.xmin) / priorVariance[0];
			encodeBBox->ymin = (bbox.ymin - priorBBox.ymin) / priorVariance[1];
			encodeBBox->xmax = (bbox.xmax - priorBBox.xmax) / priorVariance[2];
			encodeBBox->ymax = (bbox.ymax - priorBBox.ymax) / priorVariance[3];
		}
	} else if (codeType == CodeType::CENTER_SIZE) {
		float priorWidth = priorBBox.xmax - priorBBox.xmin;
		SASSERT0(priorWidth > 0);
		float priorHeight = priorBBox.ymax - priorBBox.ymin;
		SASSERT0(priorHeight > 0);
		float priorCenterX = (priorBBox.xmin + priorBBox.xmax) / 2.;
		float priorCenterY = (priorBBox.ymin + priorBBox.ymax) / 2.;

		float bboxWidth = bbox.xmax - bbox.xmin;
		SASSERT0(bboxWidth > 0);
		float bboxHeight = bbox.ymax - bbox.ymin;
		SASSERT0(bboxHeight > 0);
		float bboxCenterX = (bbox.xmin + bbox.xmax) / 2.;
		float bboxCenterY = (bbox.ymin + bbox.ymax) / 2.;

		if (encodeVarianceInTarget) {
			encodeBBox->xmin = (bboxCenterX - priorCenterX) / priorWidth;
			encodeBBox->ymin = (bboxCenterY - priorCenterY) / priorHeight;
			encodeBBox->xmax = std::log(bboxWidth / priorWidth);
			encodeBBox->ymax = std::log(bboxHeight / priorHeight);
		} else {
			// Encode variance in bbox.
			encodeBBox->xmin = (bboxCenterX - priorCenterX) / priorWidth / priorVariance[0];
			encodeBBox->ymin = (bboxCenterY - priorCenterY) / priorHeight / priorVariance[1];
			encodeBBox->xmax = std::log(bboxWidth / priorWidth) / priorVariance[2];
			encodeBBox->ymax = std::log(bboxHeight / priorHeight) / priorVariance[3];
		}
	} else if (codeType == CodeType::CORNER_SIZE) {
		float priorWidth = priorBBox.xmax - priorBBox.xmin;
		SASSERT0(priorWidth > 0);
		float priorHeight = priorBBox.ymax - priorBBox.ymin;
		SASSERT0(priorHeight > 0);
		if (encodeVarianceInTarget) {
			encodeBBox->xmin = (bbox.xmin - priorBBox.xmin) / priorWidth;
			encodeBBox->ymin = (bbox.ymin - priorBBox.ymin) / priorHeight;
			encodeBBox->xmax = (bbox.xmax - priorBBox.xmax) / priorWidth;
			encodeBBox->ymax = (bbox.ymax - priorBBox.ymax) / priorHeight;
		} else {
			// Encode variance in bbox.
			SASSERT0(priorVariance.size() == 4);
			for (int i = 0; i < priorVariance.size(); i++) {
				SASSERT0(priorVariance[i] > 0);
			}
			encodeBBox->xmin = (bbox.xmin - priorBBox.xmin) / priorWidth / priorVariance[0];
			encodeBBox->ymin = (bbox.ymin - priorBBox.ymin) / priorHeight / priorVariance[1];
			encodeBBox->xmax = (bbox.xmax - priorBBox.xmax) / priorWidth / priorVariance[2];
			encodeBBox->ymax = (bbox.ymax - priorBBox.ymax) / priorHeight / priorVariance[3];
		}
	} else {
		SASSERT(false, "Unknown LocLossType.");
	}
}


template <typename Dtype>
void ComputeLocLoss(Data<Dtype>& locPred, Data<Dtype>& locGt,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices, const int num,
		const int numPriors, const LocLossType locLossType,
		std::vector<std::vector<float>>* allLocLoss) {
	int locCount = locPred.getCount();
	SASSERT0(locCount == locGt.getCount());
	Data<Dtype> diff("diff");
	const Dtype* diffData = NULL;
	if (locCount != 0) {
		diff.reshape(locPred.getShape());
		soooa_gpu_sub(locCount, locPred.device_data(), locGt.device_data(), diff.mutable_device_data());
		diffData = diff.host_data();
	}
	int count = 0;
	for (int i = 0; i < num; i++) {
		std::vector<float> locLoss(numPriors, 0.f);
		for (std::map<int, std::vector<int>>::const_iterator it = allMatchIndices[i].begin();
				it != allMatchIndices[i].end(); it++) {
			const std::vector<int>& matchIndex = it->second;
			SASSERT0(numPriors == matchIndex.size());
			for (int j = 0; j < matchIndex.size(); j++) {
				if (matchIndex[j] <= -1) {
					continue;
				}
				Dtype loss = 0;
				for (int k = 0; k < 4; k++) {
					Dtype val = diffData[count * 4 + k];
					if (locLossType == LocLossType::SMOOTH_L1) {
						Dtype absVal = fabs(val);
						if (absVal < 1.) {
							loss += 0.5 * val * val;
						} else {
							loss += absVal - 0.5;
						}
					} else if (locLossType == LocLossType::L2) {
						loss += 0.5 * val * val;
					} else {
						SASSERT(false, "Unknown loc loss type.");
					}
				}
				locLoss[j] = loss;
				count++;
			}
		}
		allLocLoss->push_back(locLoss);
	}
}

template void ComputeLocLoss(Data<float>& locPred, Data<float>& locGt,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices, const int num,
		const int numPriors, const LocLossType locLossType,
		std::vector<std::vector<float>>* allLocLoss);



















template <typename Dtype>
void EncodeConfPrediction(const Dtype* confData, const int num, const int numPriors,
		const int numClasses, const int backgroundLabelId, const bool mapObjectToAgnostic,
		const MiningType miningType, const ConfLossType confLossType,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::vector<std::vector<int>>& allNegIndices,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		Dtype* confPredData, Dtype* confGtData) {

	SASSERT(numClasses > 1, "numClasses should not be less than 1.");
	if (mapObjectToAgnostic) {
		if (backgroundLabelId >= 0) {
			SASSERT0(numClasses == 2);
		} else {
			SASSERT0(numClasses == 1);
		}
	}
	bool doNegMining = (miningType != MiningType::MINING_NONE);
	int count = 0;
	for (int i = 0; i < num; i++) {
		if (allGtBBoxes.find(i) != allGtBBoxes.end()) {
			// Save matched (positive) bboxes scores and labels.
			const std::map<int, std::vector<int>>& matchIndices = allMatchIndices[i];
			for (std::map<int, std::vector<int>>::const_iterator it = matchIndices.begin();
					it != matchIndices.end(); it++) {
				const std::vector<int>& matchIndex = it->second;
				SASSERT0(matchIndex.size() == numPriors);
				for (int j = 0; j < numPriors; j++) {
					if (matchIndex[j] <= -1) {
						continue;
					}
					const int gtLabel = mapObjectToAgnostic ?
							backgroundLabelId + 1 :
							allGtBBoxes.find(i)->second[matchIndex[j]].label;
					int idx = doNegMining ? count : j;
					if (confLossType == ConfLossType::SOFTMAX) {
						confGtData[idx] = gtLabel;
					} else if (confLossType == ConfLossType::LOGISTIC) {
						confGtData[idx * numClasses + gtLabel] = 1;
					} else {
						SASSERT(false, "Unknown conf loss type.");
					}
					if (doNegMining) {
						// Copy scores for matched bboxes.
						soooa_copy<Dtype>(numClasses, confData + j * numClasses,
								confPredData + count * numClasses);
						count++;
					}
				}
			}
			// Go to next image
			if (doNegMining) {
				// Save negative bboxes scores and labels.
				for (int n = 0; n < allNegIndices[i].size(); n++) {
					int j = allNegIndices[i][n];
					SASSERT0(j < numPriors);
					soooa_copy<Dtype>(numClasses, confData + j * numClasses,
							confPredData + count * numClasses);
					if (confLossType == ConfLossType::SOFTMAX) {
						confGtData[count] = backgroundLabelId;
					} else if (confLossType == ConfLossType::LOGISTIC) {
						if (backgroundLabelId >= 0 && backgroundLabelId < numClasses) {
							confGtData[count * numClasses + backgroundLabelId] = 1;
						}
					} else {
						SASSERT(false, "Unknown conf loss type.");
					}
					count++;
				}
			}
		}
		if (doNegMining) {
			confData += numPriors * numClasses;
		} else {
			confGtData += numPriors;
		}
	}
}

template void EncodeConfPrediction(const float* confData, const int num, const int numPriors,
		const int numClasses, const int backgroundLabelId, const bool mapObjectToAgnostic,
		const MiningType miningType, const ConfLossType confLossType,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::vector<std::vector<int>>& allNegIndices,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		float* confPredData, float* confGtData);








template <typename Dtype>
void GetConfidenceScores(const Dtype* confData, const int num, const int numPredsPerClass,
		const int numClasses, std::vector<std::map<int, std::vector<float>>>* confPreds) {
	confPreds->clear();
	confPreds->resize(num);
	for (int i = 0; i < num; i++) {
		std::map<int, std::vector<float>>& labelScores = (*confPreds)[i];
		for (int p = 0; p < numPredsPerClass; p++) {
			int startIdx = p * numClasses;
			for (int c = 0; c < numClasses; c++) {
				labelScores[c].push_back(confData[startIdx + c]);
			}
		}
		confData += numPredsPerClass * numClasses;
	}
}

template void GetConfidenceScores(const float* confData, const int num,
		const int numPredsPerClass, const int numClasses,
		std::vector<std::map<int, std::vector<float>>>* confPreds);

template void GetConfidenceScores(const double* confData, const int num,
		const int numPredsPerClass, const int numClasses,
		std::vector<std::map<int, std::vector<float>>>* confPreds);








void DecodeBBoxesAll(const std::vector<LabelBBox>& allLocPreds,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const int num, const bool shareLocation,
		const int numLocClasses, const int backgroundLabelId,
		const CodeType codeType, const bool varianceEncodedInTarget,
		const bool clip, std::vector<LabelBBox>* allDecodeBBoxes) {

	SASSERT0(allLocPreds.size() == num);
	allDecodeBBoxes->clear();
	allDecodeBBoxes->resize(num);
	for (int i = 0; i < num; i++) {
		// Decode predictions into bboxes.
		LabelBBox& decodeBBoxes = (*allDecodeBBoxes)[i];
		for (int c = 0; c < numLocClasses; c++) {
			int label = shareLocation ? -1 : c;
			if (label == backgroundLabelId) {
				// Ignore background class.
				continue;
			}
			if (allLocPreds[i].find(label) == allLocPreds[i].end()) {
				// Something bad happend if there are no predictions for current label.
				SASSERT(false, "Could not find location predictions for label %d.", label);
			}
			const std::vector<NormalizedBBox>& labelLocPreds =
					allLocPreds[i].find(label)->second;
			DecodeBBoxes(priorBBoxes, priorVariances, codeType, varianceEncodedInTarget, clip,
					labelLocPreds, &(decodeBBoxes[label]));
		}
	}
}


void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int topK,
		std::vector<std::pair<float, int>>* scoreIndexVec) {
	// Generate index score pairs.
	for (int i = 0; i < scores.size(); i++) {
		if (scores[i] > threshold) {
			scoreIndexVec->push_back(std::make_pair(scores[i], i));
		}
	}

	// Sort the score pair according to the scores in descending order
	std::stable_sort(scoreIndexVec->begin(), scoreIndexVec->end(),
			SortScorePairDescend<int>);

	// Keep topK scores if needed.
	if (topK > -1 && topK < scoreIndexVec->size()) {
		scoreIndexVec->resize(topK);
	}
}

template <typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
		const int topK, std::vector<std::pair<Dtype, int>>* scoreIndexVec) {
	// Generate index score pairs.
	for (int i = 0; i < num; i++) {
		if (scores[i] > threshold) {
			scoreIndexVec->push_back(std::make_pair(scores[i], i));
		}
	}

	// Sort the score pair according to the scores in descending order
	std::stable_sort(scoreIndexVec->begin(), scoreIndexVec->end(),
			SortScorePairDescend<int>);

	// Keep topK scores if needed.
	if (topK > -1 && topK < scoreIndexVec->size()) {
		scoreIndexVec->resize(topK);
	}
}

template void GetMaxScoreIndex(const float* scores, const int num, const float threshold,
		const int topK, std::vector<std::pair<float, int>>* scoreIndexVec);


void ApplyNMSFast(const std::vector<NormalizedBBox>& bboxes, const std::vector<float>& scores,
		const float scoreThreshold, const float nmsThreshold,
		const float eta, const int topK, std::vector<int>* indices) {

	// Sanity check.
	SASSERT(bboxes.size() == scores.size(), "bboxes and scores have different size.");

	// Get topK scores (with corresponding indices).
	std::vector<std::pair<float, int>> scoreIndexVec;
	GetMaxScoreIndex(scores, scoreThreshold, topK, &scoreIndexVec);

	// Do nms.
	float adaptiveThreshold = nmsThreshold;
	indices->clear();
	while (scoreIndexVec.size() != 0) {
		const int idx = scoreIndexVec.front().second;
		bool keep = true;
		for (int k = 0; k < indices->size(); k++) {
			if (keep) {
				const int keptIdx = (*indices)[k];
				float overlap = JaccardOverlap(bboxes[idx], bboxes[keptIdx]);
				keep = overlap <= adaptiveThreshold;
			} else {
				break;
			}
		}
		if (keep) {
			indices->push_back(idx);
		}
		scoreIndexVec.erase(scoreIndexVec.begin());
		if (keep && eta < 1 && adaptiveThreshold > 0.5) {
			adaptiveThreshold *= eta;
		}
	}
}

template <typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
		const float scoreThreshold, const float nmsThreshold, const float eta,
		const int topK, std::vector<int>* indices) {
	// Get topK scores (with corresponding indices).
	std::vector<std::pair<float, int>> scoreIndexVec;
	GetMaxScoreIndex(scores, num, scoreThreshold, topK, &scoreIndexVec);

	// Do nms.
	float adaptiveThreshold = nmsThreshold;
	indices->clear();
	while (scoreIndexVec.size() != 0) {
		const int idx = scoreIndexVec.front().second;
		bool keep = true;
		for (int k = 0; k < indices->size(); k++) {
			if (keep) {
				const int keptIdx = (*indices)[k];
				Dtype overlap = JaccardOverlap(bboxes + idx * 4, bboxes + keptIdx * 4);
				keep = overlap <= adaptiveThreshold;
			} else {
				break;
			}
		}
		if (keep) {
			indices->push_back(idx);
		}
		scoreIndexVec.erase(scoreIndexVec.begin());
		if (keep && eta < 1 && adaptiveThreshold > 0.5) {
			adaptiveThreshold *= eta;
		}
	}
}

template void ApplyNMSFast(const float* bboxes, const float* scores, const int num,
		const float scoreThreshold, const float nmsThreshold, const float eta,
		const int topK, std::vector<int>* indices);



void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
		NormalizedBBox* scaleBBox) {
	scaleBBox->xmin = bbox.xmin * width;
	scaleBBox->ymin = bbox.ymin * height;
	scaleBBox->xmax = bbox.xmax * width;
	scaleBBox->ymax = bbox.ymax * height;
	scaleBBox->size = 0.f;
	bool normalized = !(width > 1 || height > 1);
	scaleBBox->size = BBoxSize(*scaleBBox, normalized);
	scaleBBox->difficult = bbox.difficult;
}


void OutputBBox(const NormalizedBBox& bbox, const std::pair<int, int>& imgSize,
		const bool hasResize, NormalizedBBox* outBBox) {
	// 현재 resize 지원하지 않음.
	SASSERT0(!hasResize);

	const int height = imgSize.first;
	const int width = imgSize.second;
	NormalizedBBox tempBBox = bbox;

	if (hasResize) {

	} else {
		// Clip the normalized bbox first.
		ClipBBox(tempBBox, &tempBBox);
		// Scale the bbox according to the original image size.
		ScaleBBox(tempBBox, height, width, outBBox);
	}
}

cv::Scalar HSV2RGB(const float h, const float s, const float v) {
	const int h_i = static_cast<int>(h * 6);
	const float f = h * 6 - h_i;
	const float p = v * (1 - s);
	const float q = v * (1 - f*s);
	const float t = v * (1 - (1 - f) * s);
	float r, g, b;
	switch (h_i) {
	case 0:
		r = v; g = t; b = p;
		break;
	case 1:
		r = q; g = v; b = p;
		break;
	case 2:
		r = p; g = v; b = t;
		break;
	case 3:
		r = p; g = q; b = v;
		break;
	case 4:
		r = t; g = p; b = v;
		break;
	case 5:
		r = v; g = p; b = q;
		break;
	default:
		r = 1; g = 1; b = 1;
		break;
	}
	return cv::Scalar(r * 255, g * 255, b * 255);
}



std::vector<cv::Scalar> GetColors(const int n) {
	std::vector<cv::Scalar> colors;
	cv::RNG rng(12345);
	const float goldenRatioConjugate = 0.618033988749895;
	const float s = 0.3;
	const float v = 0.99;
	for (int i = 0; i < n; i++) {
		const float h = std::fmod(rng.uniform(0.f, 1.f) + goldenRatioConjugate, 1.f);
		colors.push_back(HSV2RGB(h, s, v));
	}
	return colors;
}

static clock_t startClock = clock();
static cv::VideoWriter capOut;

template <typename Dtype>
void VisualizeBBox(const std::vector<cv::Mat>& images, Data<Dtype>* detections,
		const float threshold, const std::vector<cv::Scalar>& colors,
		const std::map<int, std::string>& labelToDisplayName, const std::string& saveFile) {

	// Retrieve detections.
	SASSERT0(detections->width() == 7);
	const int numDet = detections->height();
	const int numImg = images.size();
	if (numDet == 0 || numImg == 0) {
		return;
	}
	// Compute FPS.
	float fps = numImg / (static_cast<double>(clock() - startClock) / CLOCKS_PER_SEC);
	const Dtype* detectionsData = detections->host_data();
	const int width = images[0].cols;
	const int height = images[0].rows;
	std::vector<LabelBBox> allDetections(numImg);
	for (int i = 0; i < numDet; i++) {
		const int imgIdx = detectionsData[i * 7];
		SASSERT0(imgIdx < numImg);
		const int label = detectionsData[i * 7 + 1];
		const float score = detectionsData[i * 7 + 2];
		if (score < threshold) {
			continue;
		}
		NormalizedBBox bbox;
		bbox.xmin = detectionsData[i * 7 + 3] * width;
		bbox.ymin = detectionsData[i * 7 + 4] * height;
		bbox.xmax = detectionsData[i * 7 + 5] * width;
		bbox.ymax = detectionsData[i * 7 + 6] * height;
		bbox.score = score;
		allDetections[imgIdx][label].push_back(bbox);
	}

	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 1;
	int thickness = 2;
	int baseline = 0;
	char buffer[50];
	for (int i = 0; i < numImg; i++) {
		cv::Mat image = images[i];
		// Show FPS
		snprintf(buffer, sizeof(buffer), "FPS: %.2f", fps);
		cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness, &baseline);
		cv::rectangle(image, cv::Point(0, 0), cv::Point(text.width, text.height + baseline),
				CV_RGB(255, 255, 255), CV_FILLED);
		cv::putText(image, buffer, cv::Point(0, text.height + baseline / 2.),
				fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
		// Draw bboxes.
		for (std::map<int, std::vector<NormalizedBBox>>::iterator it = allDetections[i].begin();
				it != allDetections[i].end(); it++) {
			int label = it->first;
			std::string labelName = "Unknown";
			if (labelToDisplayName.find(label) != labelToDisplayName.end()) {
				labelName = labelToDisplayName.find(label)->second;
			}
			SASSERT0(label < colors.size());
			const cv::Scalar& color = colors[label];
			const std::vector<NormalizedBBox>& bboxes = it->second;
			for (int j = 0; j < bboxes.size(); j++) {
				cv::Point topLeftPt(bboxes[j].xmin, bboxes[j].ymin);
				cv::Point bottomRightPt(bboxes[j].xmax, bboxes[j].ymax);
				cv::rectangle(image, topLeftPt, bottomRightPt, color, 4);
				cv::Point bottomLeftPt(bboxes[j].xmin, bboxes[j].ymax);
				snprintf(buffer, sizeof(buffer), "%s: %.2f", labelName.c_str(),
						bboxes[j].score);
				cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
						&baseline);
				cv::rectangle(image, bottomLeftPt + cv::Point(0, 0),
						bottomLeftPt + cv::Point(text.width, -text.height - baseline),
						color, CV_FILLED);
				cv::putText(image, buffer, bottomLeftPt - cv::Point(0, baseline),
						fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
			}
		}
		// Save result if required.
		if (!saveFile.empty()) {
			if (!capOut.isOpened()) {
				cv::Size size(image.size().width, image.size().height);
				cv::VideoWriter outputVideo(saveFile, CV_FOURCC('D', 'I', 'V', 'X'),
						30, size, true);
				capOut = outputVideo;
			}
			capOut.write(image);
		}
		cv::imshow("detections", image);
		if (cv::waitKey(1) == 27) {
			raise(SIGINT);
		}
	}
	startClock = clock();
}

template void VisualizeBBox(const std::vector<cv::Mat>& images, Data<float>* detections,
		const float threshold, const std::vector<cv::Scalar>& colors,
		const std::map<int, std::string>& labelToDisplayName, const std::string& saveFile);










template <typename Dtype>
void GetDetectionResults(const Dtype* detData, const int numDet, const int backgroundLabelId,
		std::map<int, LabelBBox>* allDetections) {
	allDetections->clear();
	for (int i = 0; i < numDet; i++) {
		int startIdx = i * 7;
		int itemId = detData[startIdx];
		if (itemId == -1) {
			continue;
		}
		int label = detData[startIdx + 1];
		SASSERT(backgroundLabelId != label,
				"Found background label in the detection results.");
		NormalizedBBox bbox;
		bbox.score = detData[startIdx + 2];
		bbox.xmin = detData[startIdx + 3];
		bbox.ymin = detData[startIdx + 4];
		bbox.xmax = detData[startIdx + 5];
		bbox.ymax = detData[startIdx + 6];
		float bboxSize = BBoxSize(bbox);
		bbox.size = bboxSize;
		(*allDetections)[itemId][label].push_back(bbox);
	}
}

template void GetDetectionResults(const float* detData, const int numDet,
		const int backgroundLabelId, std::map<int, LabelBBox>* allDetections);



template <typename Dtype>
__global__ void DecodeBBoxesKernel(const int nthreads, const Dtype* loc_data,
		const Dtype* prior_data, const int code_type,
		const bool variance_encoded_in_target, const int num_priors,
		const bool share_location, const int num_loc_classes, const int background_label_id,
		const bool clip_bbox, Dtype* bbox_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int i = index % 4;
		const int c = (index / 4) % num_loc_classes;
		const int d = (index / 4 / num_loc_classes) % num_priors;
		if (!share_location && c == background_label_id) {
			// Ignore background class if not share_location.
			return;
		}
		const int pi = d * 4;
		const int vi = pi + num_priors * 4;
		//if (code_type == PriorBoxParameter_CodeType_CORNER) {
		if (code_type == 0) {
			if (variance_encoded_in_target) {
				// variance is encoded in target, we simply need to add the offset
				// predictions.
				bbox_data[index] = prior_data[pi + i] + loc_data[index];
				} else {
				// variance is encoded in bbox, we need to scale the offset accordingly.
				bbox_data[index] =
				prior_data[pi + i] + loc_data[index] * prior_data[vi + i];
			}
		//} else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
		} else if (code_type == 1) {
			const Dtype p_xmin = prior_data[pi];
			const Dtype p_ymin = prior_data[pi + 1];
			const Dtype p_xmax = prior_data[pi + 2];
			const Dtype p_ymax = prior_data[pi + 3];
			const Dtype prior_width = p_xmax - p_xmin;
			const Dtype prior_height = p_ymax - p_ymin;
			const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
			const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

			const Dtype xmin = loc_data[index - i];
			const Dtype ymin = loc_data[index - i + 1];
			const Dtype xmax = loc_data[index - i + 2];
			const Dtype ymax = loc_data[index - i + 3];

			Dtype decode_bbox_center_x, decode_bbox_center_y;
			Dtype decode_bbox_width, decode_bbox_height;
			if (variance_encoded_in_target) {
				// variance is encoded in target, we simply need to retore the offset
				// predictions.
				decode_bbox_center_x = xmin * prior_width + prior_center_x;
				decode_bbox_center_y = ymin * prior_height + prior_center_y;
				decode_bbox_width = exp(xmax) * prior_width;
				decode_bbox_height = exp(ymax) * prior_height;
			} else {
				// variance is encoded in bbox, we need to scale the offset accordingly.
				decode_bbox_center_x =
				prior_data[vi] * xmin * prior_width + prior_center_x;
				decode_bbox_center_y =
				prior_data[vi + 1] * ymin * prior_height + prior_center_y;
				decode_bbox_width =
				exp(prior_data[vi + 2] * xmax) * prior_width;
				decode_bbox_height =
				exp(prior_data[vi + 3] * ymax) * prior_height;
			}

			switch (i) {
			case 0:
				bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;
				break;
			case 1:
				bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;
				break;
			case 2:
				bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;
				break;
			case 3:
				bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;
				break;
			}
		//} else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
		} else if (code_type == 2) {
			const Dtype p_xmin = prior_data[pi];
			const Dtype p_ymin = prior_data[pi + 1];
			const Dtype p_xmax = prior_data[pi + 2];
			const Dtype p_ymax = prior_data[pi + 3];
			const Dtype prior_width = p_xmax - p_xmin;
			const Dtype prior_height = p_ymax - p_ymin;
			Dtype p_size;
			if (i == 0 || i == 2) {
				p_size = prior_width;
			} else {
				p_size = prior_height;
			}
			if (variance_encoded_in_target) {
				// variance is encoded in target, we simply need to add the offset
				// predictions.
				bbox_data[index] = prior_data[pi + i] + loc_data[index] * p_size;
			} else {
				// variance is encoded in bbox, we need to scale the offset accordingly.
				bbox_data[index] =
				prior_data[pi + i] + loc_data[index] * prior_data[vi + i] * p_size;
			}
		} else {
			// Unknown code type.
		}
		if (clip_bbox) {
			bbox_data[index] = max(min(bbox_data[index], Dtype(1.)), Dtype(0.));
		}
	}
}


template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads, const Dtype* locData, const Dtype* priorData,
		const CodeType codeType, const bool varianceEncodedInTarget,
		const int numPriors, const bool shareLocation,
		const int numLocClasses, const int backgroundLabelId,
		const bool clipBBox, Dtype* bboxData) {

	int codeTypeInt = -1;
	if (codeType == CodeType::CORNER) codeTypeInt = 0;
	else if (codeType == CodeType::CENTER_SIZE) codeTypeInt = 1;
	else if (codeType == CodeType::CORNER_SIZE) codeTypeInt = 2;

	DecodeBBoxesKernel<Dtype><<<SOOOA_GET_BLOCKS(nthreads), SOOOA_CUDA_NUM_THREADS>>>(
			nthreads, locData, priorData, codeTypeInt, varianceEncodedInTarget, numPriors,
			shareLocation, numLocClasses, backgroundLabelId, clipBBox, bboxData);
	  CUDA_POST_KERNEL_CHECK;
}

template void DecodeBBoxesGPU(const int nthreads, const float* locData, const float* priorData,
		const CodeType codeType, const bool varianceEncodedInTarget,
		const int numPriors, const bool shareLocation,
		const int numLocClasses, const int backgroundLabelId,
		const bool clipBBox, float* bboxData);

template <typename Dtype>
__global__ void PermuteDataKernel(const int nthreads, const Dtype* data,
		const int num_classes, const int num_data, const int num_dim, Dtype* new_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int i = index % num_dim;
		const int c = (index / num_dim) % num_classes;
		const int d = (index / num_dim / num_classes) % num_data;
		const int n = index / num_dim / num_classes / num_data;
		const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
		new_data[new_index] = data[index];
	}
}

template <typename Dtype>
void PermuteDataGPU(const int nthreads,
		const Dtype* data, const int numClasses, const int numData,
		const int numDim, Dtype* newData) {
	PermuteDataKernel<Dtype><<<SOOOA_GET_BLOCKS(nthreads), SOOOA_CUDA_NUM_THREADS>>>(
			nthreads, data, numClasses, numData, numDim, newData);
	CUDA_POST_KERNEL_CHECK;
}

template void PermuteDataGPU(const int nthreads,
		const float* data, const int numClasses, const int numData,
		const int numDim, float* newData);


// Project bbox onto the coordinate system defined by src_bbox.
bool ProjectBBox(const NormalizedBBox& srcBBox, const NormalizedBBox& bbox,
		NormalizedBBox* projBBox) {
	if (bbox.xmin >= srcBBox.xmax || bbox.xmax <= srcBBox.xmin ||
			bbox.ymin >= srcBBox.ymax || bbox.ymax <= srcBBox.ymin) {
		return false;
	}
	float srcWidth = srcBBox.xmax - srcBBox.xmin;
	float srcHeight = srcBBox.ymax - srcBBox.ymin;
	projBBox->xmin = (bbox.xmin - srcBBox.xmin) / srcWidth;
	projBBox->ymin = (bbox.ymin - srcBBox.ymin) / srcHeight;
	projBBox->xmax = (bbox.xmax - srcBBox.xmin) / srcWidth;
	projBBox->ymax = (bbox.ymax - srcBBox.ymin) / srcHeight;
	projBBox->difficult = bbox.difficult;
	ClipBBox(*projBBox, projBBox);
	if (BBoxSize(*projBBox) > 0) {
		return true;
	} else {
		return false;
	}
}

void LocateBBox(const NormalizedBBox& srcBBox, const NormalizedBBox& bbox,
		NormalizedBBox* locBBox) {
	float srcWidth = srcBBox.xmax - srcBBox.xmin;
	float srcHeight = srcBBox.ymax - srcBBox.ymin;
	locBBox->xmin = srcBBox.xmin + bbox.xmin * srcWidth;
	locBBox->ymin = srcBBox.ymin + bbox.ymin * srcHeight;
	locBBox->xmax = srcBBox.xmin + bbox.xmax * srcWidth;
	locBBox->ymax = srcBBox.ymin + bbox.ymax * srcHeight;
	locBBox->difficult = bbox.difficult;
}

float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
	NormalizedBBox intersectBBox;
	IntersectBBox(bbox1, bbox2, &intersectBBox);
	float intersectSize = BBoxSize(intersectBBox);
	if (intersectSize > 0) {
		float bbox1Size = BBoxSize(bbox1);
		return intersectSize / bbox1Size;
	} else {
		return 0.f;
	}
}

bool MeetEmitConstraint(const NormalizedBBox& srcBBox, const NormalizedBBox& bbox,
		const EmitConstraint& emitConstraint) {
	EmitType emitType = emitConstraint.emitType;
	if (emitType == EmitType::CENTER) {
		float xcenter = (bbox.xmin + bbox.xmax) / 2;
		float ycenter = (bbox.ymin + bbox.ymax) / 2;
		if (xcenter >= srcBBox.xmin && xcenter <= srcBBox.xmax &&
				ycenter >= srcBBox.ymin && ycenter <= srcBBox.ymax) {
			return true;
		} else {
			return false;
		}
	} else if (emitType == EmitType::MIN_OVERLAP) {
		float bboxCoverage = BBoxCoverage(bbox, srcBBox);
		return bboxCoverage > emitConstraint.emitOverlap;
	} else {
		SASSERT(false, "Unknown emit type.");
		return false;
	}
}

void ExtrapolateBBox(const ResizeParam& param, const int height, const int width,
		const NormalizedBBox& cropBBox, NormalizedBBox* bbox) {
	float heightScale = param.heightScale;
	float widthScale = param.widthScale;
	if (heightScale > 0 && widthScale > 0 &&
			param.resizeMode == ResizeMode::FIT_SMALL_SIZE) {
		float origAspect = static_cast<float>(width) / height;
		float resizeHeight = param.height;
		float resizeWidth = param.width;
		float resizeAspect = resizeWidth / resizeHeight;
		if (origAspect < resizeAspect) {
			resizeHeight = resizeWidth / origAspect;
		} else {
			resizeWidth = resizeHeight * origAspect;
		}
		float cropHeight = resizeHeight * (cropBBox.ymax - cropBBox.ymin);
		float cropWidth = resizeWidth * (cropBBox.xmax - cropBBox.xmin);
		SASSERT0(cropWidth >= widthScale);
		SASSERT0(cropHeight >= heightScale);
		bbox->xmin = bbox->xmin * cropWidth / widthScale;
		bbox->xmax = bbox->xmax * cropWidth / widthScale;
		bbox->ymin = bbox->ymin * cropHeight / heightScale;
		bbox->ymax = bbox->ymax * cropHeight / heightScale;
	}

}


void CumSum(const std::vector<std::pair<float, int>>& pairs, std::vector<int>* cumSum) {
	// Sort the pairs based on first item of the pair.
	std::vector<std::pair<float, int>> sortPairs = pairs;
	std::stable_sort(sortPairs.begin(), sortPairs.end(), SortScorePairDescend<int>);

	cumSum->clear();
	for (int i = 0; i < sortPairs.size(); i++) {
		if (i == 0) {
			cumSum->push_back(sortPairs[i].second);
		} else {
			cumSum->push_back(cumSum->back() + sortPairs[i].second);
		}
	}
}

void ComputeAP(const std::vector<std::pair<float, int>>& tp, const int numPos,
		const std::vector<std::pair<float, int>>& fp, const std::string apVersion,
		std::vector<float>* prec, std::vector<float>* rec, float* ap) {
	const float eps = 1e-6;
	SASSERT(tp.size() == fp.size(), "tp must have same size as fp.");
	const int num = tp.size();
	// Make sure that tp and fp have complement value.
	for (int i = 0; i < num; i++) {
		SASSERT0(std::fabs(tp[i].first - fp[i].first) <= eps);
		SASSERT0(tp[i].second == 1 - fp[i].second);
	}
	prec->clear();
	rec->clear();
	*ap = 0;
	if (tp.size() == 0 || numPos == 0) {
		return;
	}

	// Compute cumSum of tp.
	std::vector<int> tpCumSum;
	CumSum(tp, &tpCumSum);
	SASSERT0(tpCumSum.size() == num);

	// Compute cumSum of fp.
	std::vector<int> fpCumSum;
	CumSum(fp, &fpCumSum);
	SASSERT0(fpCumSum.size() == num);

	// Compute precision.
	for (int i = 0; i < num; i++) {
		prec->push_back(static_cast<float>(tpCumSum[i]) / (tpCumSum[i] + fpCumSum[i]));
	}

	// Compute recall.
	for (int i = 0; i < num; i++) {
		SASSERT0(tpCumSum[i] <= numPos);
		rec->push_back(static_cast<float>(tpCumSum[i]) / numPos);
	}

	if (apVersion == "11point") {
		// VOC2007 style for computing AP.
		std::vector<float> maxPrecs(11, 0.f);
		int startIdx = num - 1;
		for (int j = 10; j >= 0; j--) {
			for (int i = startIdx; i >= 0; i--) {
				if ((*rec)[i] < j / 10.f) {
					startIdx = i;
					if (j > 0) {
						maxPrecs[j - 1] = maxPrecs[j];
					}
					break;
				} else {
					if (maxPrecs[j] < (*prec)[i]) {
						maxPrecs[j] = (*prec)[i];
					}
				}
			}
		}
		for (int j = 10; j >= 0; j--) {
			*ap += maxPrecs[j] / 11;
		}
	} else if (apVersion == "MaxIntegral") {
		// VOC2012 or ILSVRC style for computing AP.
		float curRec = rec->back();
		float curPrec = prec->back();
		for (int i = num - 2; i >= 0; i--) {
			curPrec = std::max<float>((*prec)[i], curPrec);
			if (fabs(curRec - (*rec)[i]) > eps) {
				*ap += curPrec * fabs(curRec - (*rec)[i]);
			}
			curRec = (*rec)[i];
		}
		*ap += curRec * curPrec;
	} else if (apVersion == "Integral") {
		// Natural integral.
		float prevRec = 0.f;
		for (int i = 0; i < num; i++) {
			if (fabs((*rec)[i] - prevRec) > eps) {
				*ap += (*prec)[i] * fabs((*rec)[i] - prevRec);
			}
			prevRec = (*rec)[i];
		}
	} else {
		STDOUT_LOG("Unknown apVersion: %s", apVersion.c_str());
	}
}















































