/*
 * ssd_common.h
 *
 *  Created on: Apr 28, 2017
 *      Author: jkim
 */

#ifndef SSD_COMMON_H_
#define SSD_COMMON_H_

#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "EnumDef.h"
#include "SDF.h"


template <typename Dtype>
class BoundingBox {
public:
	void print();
public:
	std::string name;
	int label;
	// unnormalized coords
	int xmin;
	int ymin;
	int xmax;
	int ymax;

	int diff;

	// normalized coords, ready for use
	Dtype buf[8];
};

// Obejct Detection Raw Data
template <typename Dtype>
class ODRawData {
public:
	void print();
	void displayBoundingBoxes(const std::string& baseDataPath,
			std::vector<cv::Scalar>& colorList);
public:
	cv::Mat im;
	std::string imPath;
	std::string annoPath;

	int width;
	int height;
	int depth;

	std::vector<BoundingBox<Dtype>> boundingBoxes;
};

// Object Detection Meta Data
template <typename Dtype>
class ODMetaData {
public:
	int rawIdx;
	bool flip;
};

template <typename Dtype>
class LabelMap {
public:



public:
	LabelMap();
	LabelMap(const std::string& labelMapPath);
	void build();
	void build(std::vector<LabelItem>& labelItemList);

	void setLabelMapPath(const std::string& labelMapPath);

	int convertLabelToInd(const std::string& label);
	std::string convertIndToLabel(int ind);

	void mapLabelToName(std::map<int, std::string>& labelToName);

	int getCount();
	void printLabelItemList();
    bool isValid();

public:
	std::string labelMapPath;
	std::vector<LabelItem> labelItemList;
	std::map<std::string, int> labelToIndMap;
	std::map<int, std::string> indToLabelMap;
	std::vector<cv::Scalar> colorList;

private:
    bool valid;
};


/*
// The normalized bounding box [0, 1] w.r.t. the input image size
class NormalizedBBox {
public:
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	int label;
	bool difficult;
	float score;
	float size;

	NormalizedBBox() {
		this->xmin = 0.f;
		this->ymin = 0.f;
		this->xmax = 0.f;
		this->ymax = 0.f;
		this->label = 0;
		this->difficult = false;
		this->score = 0.f;
		this->size = 0.f;
	}

	void print() {
		std::cout << "\txmin: " 		<< this->xmin		<< std::endl;
		std::cout << "\tymin: " 		<< this->ymin		<< std::endl;
		std::cout << "\txmax: " 		<< this->xmax		<< std::endl;
		std::cout << "\tymax: " 		<< this->ymax		<< std::endl;
		std::cout << "\tlabel: " 		<< this->label		<< std::endl;
		std::cout << "\tdifficult: "	<< this->difficult	<< std::endl;
		std::cout << "\tscore: " 		<< this->score		<< std::endl;
		std::cout << "\tsize: "			<< this->size		<< std::endl;
	}
};
*/







#endif /* SSD_COMMON_H_ */
