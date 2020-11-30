/*
 * BboxTransformUtil.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef BBOXTRANSFORMUTIL_H_
#define BBOXTRANSFORMUTIL_H_


#include "frcnn_common.h"

#define BBOXTRANSFORMUTIL_LOG 0


struct BboxTransformUtil {
public:
	template <typename Dtype>
	static void bboxTransform(const std::vector<std::vector<Dtype>>& ex_rois,
			const uint32_t ex_rois_offset,
			const std::vector<std::vector<Dtype>>& gt_rois,
			const uint32_t gt_rois_offset,
			std::vector<std::vector<float>>& result,
			const uint32_t resultOffset = 0) {
		assert(ex_rois.size() == gt_rois.size());

		float ex_width, ex_height, ex_ctr_x, ex_ctr_y;
		float gt_width, gt_height, gt_ctr_x, gt_ctr_y;

		const uint32_t numRois = ex_rois.size();
		for (uint32_t i = 0; i < numRois; i++) {
			 ex_width	= ex_rois[i][ex_rois_offset+2] - ex_rois[i][ex_rois_offset+0] + 1.0f;
			 ex_height	= ex_rois[i][ex_rois_offset+3] - ex_rois[i][ex_rois_offset+1] + 1.0f;
			 ex_ctr_x	= ex_rois[i][ex_rois_offset+0] + 0.5f * ex_width;
			 ex_ctr_y	= ex_rois[i][ex_rois_offset+1] + 0.5f * ex_height;

			 gt_width	= gt_rois[i][gt_rois_offset+2] - gt_rois[i][gt_rois_offset+0] + 1.0f;
			 gt_height	= gt_rois[i][gt_rois_offset+3] - gt_rois[i][gt_rois_offset+1] + 1.0f;
			 gt_ctr_x	= gt_rois[i][gt_rois_offset+0] + 0.5f * gt_width;
			 gt_ctr_y	= gt_rois[i][gt_rois_offset+1] + 0.5f * gt_height;

			 result[i][resultOffset+0] = (gt_ctr_x - ex_ctr_x) / ex_width;
			 result[i][resultOffset+1] = (gt_ctr_y - ex_ctr_y) / ex_height;
			 result[i][resultOffset+2] = std::log(gt_width / ex_width);
			 result[i][resultOffset+3] = std::log(gt_height / ex_height);
		}
	}


	template <typename Dtype>
	static void bboxTransformInv(std::vector<std::vector<Dtype>>& boxes,
			Data<Dtype>* deltas, std::vector<std::vector<Dtype>>& result) {

		if (boxes.size() == 0) {
			result.clear();
			return;
		}

		const Dtype* deltaPtr = deltas->host_data();

		uint32_t numDeltaElems = 0;
		int shapeIdx;
		for (shapeIdx = deltas->getShape().size()-1; shapeIdx >= 0; shapeIdx--) {
			if (deltas->getShape()[shapeIdx] > 1) {
				numDeltaElems = deltas->getShape()[shapeIdx];
				break;
			}
		}
		assert(shapeIdx >= 0);

		const uint32_t numClasses = numDeltaElems / boxes[0].size();
		const uint32_t numBoxes = boxes.size();
		result.resize(numBoxes);
		for (uint32_t i = 0; i < numBoxes; i++)
			result[i].resize(numDeltaElems);

		float width, height, ctrX, ctrY;
		float dx, dy, dw, dh;
		float predCtrX, predCtrY, predW, predH;

		uint32_t deltaElemIndex;
		for (uint32_t i = 0; i < numBoxes; i++) {
			std::vector<Dtype>& box = boxes[i];
			std::vector<Dtype>& predBox = result[i];
			//predBox.resize(numDeltaElems);

			width = box[2] - box[0] + 1.0f;
			height = box[3] - box[1] + 1.0f;
			ctrX = box[0] + 0.5 * width;
			ctrY = box[1] + 0.5 * height;


#if BBOXTRANSFORMUTIL_LOG
			std::cout << "width: " << width << std::endl;
			std::cout << "height: " << height << std::endl;
			std::cout << "ctrX: " << ctrX << std::endl;
			std::cout << "ctrY: " << ctrY << std::endl;

			float dxs[21];
			float dys[21];
			float dws[21];
			float dhs[21];

			float predCtrXs[21];
			float predCtrYs[21];
			float predWs[21];
			float predHs[21];

			float predBoxs[21*4];
#endif

			for (uint32_t j = 0; j < numClasses; j++) {
				deltaElemIndex = i*numDeltaElems + j*4;

				//dx = deltaPtr[deltaElemIndex + 0]*TRAIN_BBOX_NORMALIZE_STDS[0]+
                //  TRAIN_BBOX_NORMALIZE_MEANS[0];
				//dy = deltaPtr[deltaElemIndex + 1]*TRAIN_BBOX_NORMALIZE_STDS[1]+
                //  TRAIN_BBOX_NORMALIZE_MEANS[1];
				//dw = deltaPtr[deltaElemIndex + 2]*TRAIN_BBOX_NORMALIZE_STDS[2]+
                //  TRAIN_BBOX_NORMALIZE_MEANS[2];
				//dh = deltaPtr[deltaElemIndex + 3]*TRAIN_BBOX_NORMALIZE_STDS[3]+
                //  TRAIN_BBOX_NORMALIZE_MEANS[3];

				dx = deltaPtr[deltaElemIndex + 0];
				dy = deltaPtr[deltaElemIndex + 1];
				dw = deltaPtr[deltaElemIndex + 2];
				dh = deltaPtr[deltaElemIndex + 3];

				predCtrX = dx * width + ctrX;
				predCtrY = dy * height + ctrY;
				predW = std::exp(dw) * width;
				predH = std::exp(dh) * height;

				predBox[j*4 + 0] = predCtrX - 0.5 * predW;
				predBox[j*4 + 1] = predCtrY - 0.5 * predH;
				//predBox[j*4 + 2] = predCtrX + 0.5 * predW;
				//predBox[j*4 + 3] = predCtrY + 0.5 * predH;
				predBox[j*4 + 2] = predCtrX + 0.5 * predW;
				predBox[j*4 + 3] = predCtrY + 0.5 * predH;

#if BBOXTRANSFORMUTIL_LOG
				dxs[j] = dx;
				dys[j] = dy;
				dws[j] = dw;
				dhs[j] = dh;

				predCtrXs[j] = predCtrX;
				predCtrYs[j] = predCtrY;
				predWs[j] = predW;
				predHs[j] = predH;

				predBoxs[j*4 + 0] = predBox[j*4+0];
				predBoxs[j*4 + 1] = predBox[j*4+1];
				predBoxs[j*4 + 2] = predBox[j*4+2];
				predBoxs[j*4 + 3] = predBox[j*4+3];
#endif

			}

#if BBOXTRANSFORMUTIL_LOG
			std::cout << "dxs: ";
			for (uint32_t j = 0; j < 21; j++)
				std::cout << dxs[j] << ", ";
			std::cout << std::endl;

			std::cout << "dys: ";
			for (uint32_t j = 0; j < 21; j++)
				std::cout << dys[j] << ", ";
			std::cout << std::endl;

			std::cout << "dws: ";
			for (uint32_t j = 0; j < 21; j++)
				std::cout << dws[j] << ", ";
			std::cout << std::endl;

			std::cout << "dhs: ";
			for (uint32_t j = 0; j < 21; j++)
				std::cout << dhs[j] << ", ";
			std::cout << std::endl;

			std::cout << "predCtrXs: ";
			for (uint32_t j = 0; j < 21; j++)
				std::cout << predCtrXs[j] << ", ";
			std::cout << std::endl;

			std::cout << "predCtrYs: ";
			for (uint32_t j = 0; j < 21; j++)
				std::cout << predCtrYs[j] << ", ";
			std::cout << std::endl;

			std::cout << "predWs: ";
			for (uint32_t j = 0; j < 21; j++)
				std::cout << predWs[j] << ", ";
			std::cout << std::endl;

			std::cout << "predHs: ";
			for (uint32_t j = 0; j < 21; j++)
				std::cout << predHs[j] << ", ";
			std::cout << std::endl;

			std::cout << "predBoxs: ";
			for (uint32_t j = 0; j < 21*4; j++)
				std::cout << predBoxs[j] << ", ";
			std::cout << std::endl;
#endif
		}
	}

	template <typename Dtype>
	static void clipBoxes(std::vector<std::vector<Dtype>>& boxes,
			const std::vector<Dtype>& imShape) {
		// clip boxes to image boundaries.
		if (boxes.size() < 1)
			return;

		const uint32_t numClasses = boxes[0].size() / 4;
		const uint32_t numBoxes = boxes.size();
		for (uint32_t i = 0; i < numBoxes; i++) {
			std::vector<Dtype>& box = boxes[i];

			for (uint32_t j = 0; j < numClasses; j++) {
				// x1 >= 0
				box[4*j+0] = std::max(std::min(box[4*j+0], imShape[1] - 1.0f), 0.0f);
				// y1 >= 0
				box[4*j+1] = std::max(std::min(box[4*j+1], imShape[0] - 1.0f), 0.0f);
				// x2 < imShape[1]
				box[4*j+2] = std::max(std::min(box[4*j+2], imShape[1] - 1.0f), 0.0f);
				// y2 < imShape[2]
				box[4*j+3] = std::max(std::min(box[4*j+3], imShape[0] - 1.0f), 0.0f);
			}
		}
	}
};

#endif /* BBOXTRANSFORMUTIL_H_ */
