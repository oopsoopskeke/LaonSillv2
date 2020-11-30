/*
 * RoIDBUtil.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef ROIDBUTIL_H_
#define ROIDBUTIL_H_

#include <algorithm>
#include <iostream>
#include <ostream>
#include <cstdint>
#include <vector>

#include "frcnn_common.h"
#include "BboxTransformUtil.h"
#include "RoIDB.h"
#include "SysLog.h"
#include "Perf.h"

#define ROIDBUTIL_LOG 0


class RoIDBUtil {
public:
	static void addBboxRegressionTargets(std::vector<RoIDB>& roidb,
			std::vector<std::vector<float>>& means, std::vector<std::vector<float>>& stds) {
		// Add information needed to train bounding-box regressors.
		SASSERT0(roidb.size() > 0);

		const uint32_t numImages = roidb.size();
		// Infer numfer of classes from the number of columns in gt_overlaps
		const uint32_t numClasses = roidb[0].gt_overlaps[0].size();

		for (uint32_t i = 0; i < numImages; i++) {
			RoIDBUtil::computeTargets(roidb[i]);
		}

		SASSERT0(TRAIN_BBOX_NORMALIZE_TARGETS_PRECOMPUTED);
		np_tile(TRAIN_BBOX_NORMALIZE_MEANS, numClasses, means);
#if ROIDBUTIL_LOG
		print2dArray("bbox target means", means);
#endif
		np_tile(TRAIN_BBOX_NORMALIZE_STDS, numClasses, stds);
#if ROIDBUTIL_LOG
		print2dArray("bbox target stdeves", stds);
#endif


		// XXX: 정답 box로 target을 구했기 때문에 target값이 모두 0,
		// 따라서 표준 mean, std로 normalize해봤자 항상 0,
		// 현재로는 의미가 없어서 빼두지만 custom mean, std를 사용할 경우
		// 적절히 처리를 해줘야 할 것.
		// 까먹을 가능성도 있는데 !!!
#if 0
		// Normalize targets
		std::cout << "Normalizing targets" << std::endl;
		for (uint32_t i = 0; i < numImages; i++) {
			std::vector<std::vector<float>>& targets = roidb[i].bbox_targets;
			std::vector<uint32_t> clsInds;
			for (uint32_t j = 1; j < numClasses; j++) {
				np_where_s(targets, static_cast<float>(j), (uint32_t)0, clsInds);

				for (uint32_t k = 0; k < clsInds.size(); k++) {
					targets[k][1] = (targets[k][1] - means[j][0]) / stds[j][0];
					targets[k][2] = (targets[k][2] - means[j][1]) / stds[j][1];
					targets[k][3] = (targets[k][3] - means[j][2]) / stds[j][2];
					targets[k][4] = (targets[k][4] - means[j][3]) / stds[j][3];
				}
			}
#if ROIDBUTIL_LOG
			print2dArray("bbox_targets", targets);
#endif
		}
#endif
	}

	static void computeTargets(RoIDB& roidb) {
#if ROIDBUTIL_LOG
		roidb.print();
#endif

		// Compute bounding-box regression targets for an image.
		// Indices of ground-truth ROIs
		std::vector<uint32_t> gtInds;
		// XXX: 1.0f float compare check
		np_where_s(roidb.max_overlaps, EQ, 1.0f, gtInds);
		if (gtInds.size() < 1) {
			// Bail if the image has no ground-truth ROIs
		}
		// Indices of examples for which we try to make predictions
		std::vector<uint32_t> exInds;
		np_where_s(roidb.max_overlaps, GE, TRAIN_BBOX_THRESH, exInds);

		// Get IoU overlap between each ex ROI and gt ROI
		std::vector<std::vector<float>> ex_gt_overlaps;
		bboxOverlaps(roidb.boxes, gtInds, 0, exInds, 0, ex_gt_overlaps);
#if ROIDBUTIL_LOG
		print2dArray("ex_gt_overlaps", ex_gt_overlaps);
#endif

		// Find which gt ROI each ex ROI has max overlap with:
		// this will be the ex ROI's gt target
		std::vector<uint32_t> gtAssignment;
		np_argmax(ex_gt_overlaps, 1, gtAssignment);
		std::vector<uint32_t> gtRoisInds;
		std::vector<std::vector<uint32_t>> gtRois;
		py_arrayElemsWithArrayInds(gtInds, gtAssignment, gtRoisInds);
		py_arrayElemsWithArrayInds(roidb.boxes, gtRoisInds, gtRois);
#if ROIDBUTIL_LOG
		print2dArray("gt_rois", gtRois);
#endif
		std::vector<std::vector<uint32_t>> exRois;
		py_arrayElemsWithArrayInds(roidb.boxes, exInds, exRois);
#if ROIDBUTIL_LOG
		print2dArray("ex_rois", exRois);
#endif

		const uint32_t numRois = roidb.boxes.size();
		const uint32_t numEx = exInds.size();
		std::vector<std::vector<float>>& targets = roidb.bbox_targets;
		targets.resize(numRois);
		for (uint32_t i = 0; i < numRois; i++) {
			targets[i].resize(5);
			// XXX: init to zero ... ?
		}
#if ROIDBUTIL_LOG
		print2dArray("targets", targets);
#endif

		for (uint32_t i = 0; i < numEx; i++) {
			targets[i][0] = roidb.max_classes[i];
		}


		BboxTransformUtil::bboxTransform(exRois, 0, gtRois, 0, targets, 1);
#if ROIDBUTIL_LOG
		print2dArray("targets", targets);
		roidb.print();
#endif
	}

	static void bboxOverlaps(const std::vector<std::vector<uint32_t>>& rois,
			const std::vector<uint32_t>& gt_inds, const uint32_t gtOffset,
			const std::vector<uint32_t>& ex_inds, const uint32_t exOffset,
			std::vector<std::vector<float>>& result) {

		const uint32_t numEx = ex_inds.size();
		const uint32_t numGt = gt_inds.size();

		result.resize(numEx);
		for (uint32_t i = 0; i < numEx; i++) {
			result[i].resize(numGt);
			for (uint32_t j = 0; j < numGt; j++) {
				result[i][j] = iou(rois[ex_inds[i]], exOffset, rois[gt_inds[j]], gtOffset);
			}
		}
	}

	static void bboxOverlaps(const std::vector<std::vector<float>>& ex,
			const uint32_t exOffset,
			const std::vector<std::vector<float>>& gt,
			const uint32_t gtOffset,
			std::vector<std::vector<float>>& result) {

		const uint32_t numEx = ex.size();
		const uint32_t numGt = gt.size();

		result.resize(numEx);
		for (uint32_t i = 0; i < numEx; i++) {
			result[i].resize(numGt);
			for (uint32_t j = 0; j < numGt; j++) {
				result[i][j] = iou(ex[i], exOffset, gt[j], gtOffset);
			}
		}
	}

	template <typename Dtype>
	static float iou(const std::vector<Dtype>& box1, const uint32_t box1Offset,
			const std::vector<Dtype>& box2, const uint32_t box2Offset) {
		float iou = 0.0f;
		Dtype left, right, top, bottom;
		left	= std::max(box1[box1Offset+0], box2[box2Offset+0]);
		right	= std::min(box1[box1Offset+2], box2[box2Offset+2]);
		top		= std::max(box1[box1Offset+1], box2[box2Offset+1]);
		bottom	= std::min(box1[box1Offset+3], box2[box2Offset+3]);

		if(left <= right &&
				top <= bottom) {
			float i = float((right - left + 1) * (bottom - top + 1));
			float u = float(
					(box1[box1Offset+2]-box1[box1Offset+0] + 1) *
					(box1[box1Offset+3]-box1[box1Offset+1] + 1) +
					(box2[box2Offset+2]-box2[box2Offset+0] + 1) *
					(box2[box2Offset+3]-box2[box2Offset+1] + 1) -
					i);
			iou = i / u;
		}
		return iou;
	}
};



#endif /* ROIDBUTIL_H_ */
