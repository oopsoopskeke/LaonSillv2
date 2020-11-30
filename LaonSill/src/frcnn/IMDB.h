/*
 * IMDB.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef IMDB_H_
#define IMDB_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "frcnn_common.h"
#include "RoIDB.h"

class IMDB {
public:
	IMDB(const std::string& name) {
		this->name = name;
		this->numClasses = 0;
	}
	virtual ~IMDB() {}

	void appendFlippedImages() {
		const uint32_t numImages = this->imageIndex.size();

		uint32_t oldx1, oldx2;
		for (uint32_t i = 0; i < numImages; i++) {
			// copy constructor 사용시 mat copy부분이 조금 모호해서 직접 copy
			RoIDB roidb;
			roidb.image = this->roidb[i].image;
			roidb.flipped = this->roidb[i].flipped;
			roidb.width = this->roidb[i].width;
			roidb.height = this->roidb[i].height;
			roidb.boxes = this->roidb[i].boxes;
			roidb.gt_classes = this->roidb[i].gt_classes;
			roidb.gt_overlaps = this->roidb[i].gt_overlaps;
			roidb.max_classes = this->roidb[i].max_classes;
			roidb.max_overlaps = this->roidb[i].max_overlaps;
			roidb.bbox_targets = this->roidb[i].bbox_targets;
			//this->roidb[i].mat.copyTo(roidb.mat);

			const uint32_t numObjects = this->roidb[i].boxes.size();
			for (uint32_t j = 0; j < numObjects; j++) {
				oldx1 = roidb.boxes[j][0];
				oldx2 = roidb.boxes[j][2];
				roidb.boxes[j][0] = roidb.width - oldx2 - 1;
				roidb.boxes[j][2] = roidb.width - oldx1 - 1;
			}
			roidb.flipped = true;
			//cv::flip(roidb.mat, roidb.mat, 1);
			cv::flip(this->roidb[i].mat, roidb.mat, 1);
			this->roidb.push_back(roidb);
			this->imageIndex.push_back(this->imageIndex[i]);
		}
	}

	virtual void getWidths(std::vector<uint32_t>& widths) {
		std::cout << "IMDB::getWidths() is not supported ... " << std::endl;
		exit(1);
	}

	virtual void loadGtRoidb() {
		std::cout << "IMDB::loadGtRoidb() is not supported ... " << std::endl;
		exit(1);
	}

	virtual std::string imagePathAt(const uint32_t index) {
		std::cout << "IMDB::imagePathAt() is not supported ... " << std::endl;
		exit(1);
	}

	virtual uint32_t convertClassToInd(const std::string& cls) {
		std::cout << "IMDB::convertClassToInd() is not supported ... " << std::endl;
		exit(1);
	}

	virtual std::string convertIndToClass(const uint32_t ind) {
		std::cout << "IMDB::convertIndToClass() is not supported ... " << std::endl;
		exit(1);
	}


	std::string name;
	uint32_t numClasses;
	std::vector<uint32_t> clasess;
	std::vector<std::string> imageIndex;
	std::vector<RoIDB> roidb;
};



#endif /* IMDB_H_ */
