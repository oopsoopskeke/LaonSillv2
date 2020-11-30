/*
 * RoIDB.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef ROIDB_H_
#define ROIDB_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "frcnn_common.h"

#define ROIDB_LOG 0

class RoIDB {
public:
	RoIDB() {};

	// 통상적으로 call by value등 다양한 곳에서 copy constructor가 호출되는데
	// override할 경우 상식적인 동작과 조금 달라 혼동이 생긴다.
	// 현재 override할 이유는 mat을 copy하는 것 뿐인데 이를 직접해주는 걸로 일단 work around
	/*
	RoIDB(const RoIDB& roidb) {
		std::cout << "roidb copy constructor called ... " << std::endl;

		this->image = roidb.image;
		this->flipped = roidb.flipped;
		this->width = roidb.width;
		this->height = roidb.height;

		this->boxes = roidb.boxes;
		this->gt_classes = roidb.gt_classes;
		this->gt_overlaps = roidb.gt_overlaps;
		this->max_classes = roidb.max_classes;
		this->max_overlaps = roidb.max_overlaps;
		this->bbox_targets = roidb.bbox_targets;

		roidb.mat.copyTo(this->mat);
	}
	*/

	void print() {
		std::cout << "---RoIDB---" << std::endl;
		printPrimitive("image", image);
		printPrimitive("flipped", flipped);
		printPrimitive("width", width);
		printPrimitive("height", height);
		print2dArray("boxes", boxes);
		printArray("gt_classes", gt_classes);
		print2dArray("gt_overlaps", gt_overlaps);
		printArray("max_classes", max_classes);
		printArray("max_overlaps", max_overlaps);
		print2dArray("bbox_targets", bbox_targets);
		std::cout << "--------------" << std::endl;
	}

	const cv::Mat getMat() const {
		return mat;
	}

	std::string image;
	bool flipped;							// Original, Flipped Image 여부
	uint32_t width;
	uint32_t height;
    std::vector<std::vector<uint32_t>> boxes;	// Annotation의 원본 bounding box 
                                                // (x1, y1, x2, y2)
	std::vector<uint32_t> gt_classes;	// 각 bounding box의 class, [9, 9, 9]의 형식
	std::vector<std::vector<float>> gt_overlaps;  // 각 bounding box의 gt_box와의 IoU Value
	std::vector<uint32_t> max_classes;
	std::vector<float> max_overlaps;
	std::vector<std::vector<float>> bbox_targets;

	cv::Mat mat;
};

#endif /* ROIDB_H_ */
