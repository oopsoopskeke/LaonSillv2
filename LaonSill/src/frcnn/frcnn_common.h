/*
 * frcnn_common.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef FRCNN_COMMON_H_
#define FRCNN_COMMON_H_


#include <iostream>
#include <ostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <array>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



#include "common.h"
#include "Data.h"
#include "cnpy.h"
#include "gpu_nms.hpp"
#include "SysLog.h"
#include "ColdLog.h"
#define FRCNN_TRAIN 1


#define SOOOA_DEBUG	0


const uint32_t GT = 0;
const uint32_t GE = 1;
const uint32_t EQ = 2;
const uint32_t LE = 3;
const uint32_t LT = 4;
const uint32_t NE = 5;

// Minibatch size (number of regions of interest [ROIs])
const uint32_t TRAIN_BATCH_SIZE = 128;

const float TRAIN_FG_THRESH = 0.5f;
const float TRAIN_BG_THRESH_HI = 0.5f;
const float TRAIN_BG_THRESH_LO = 0.0f;
const float TRAIN_BBOX_THRESH = 0.5f;

const float TRAIN_FG_FRACTION = 0.25f;

const bool	TRAIN_BBOX_NORMALIZE_TARGETS_PRECOMPUTED = true;
const std::vector<float> TRAIN_BBOX_NORMALIZE_MEANS = {0.0f, 0.0f, 0.0f, 0.0f};
const std::vector<float> TRAIN_BBOX_NORMALIZE_STDS = {0.1f, 0.1f, 0.2f, 0.2f};

const std::vector<float> TRAIN_BBOX_INSIDE_WEIGHTS = {1.0f, 1.0f, 1.0f, 1.0f};


// Use RPN to detect objects
#if FRCNN_TRAIN
const bool TRAIN_HAS_RPN = true;
#else
const bool TRAIN_HAS_RPN = false;
#endif

// Images to use per minibatch
const uint32_t TRAIN_IMS_PER_BATCH = 1;

// Scales to use duing training (can list multiple scales)
// Each scale is the pixel isze of an image's shortest side
//const std::vector<uint32_t> TRAIN_SCALES = {500};
const std::vector<uint32_t> TRAIN_SCALES = {400, 500, 600, 700};

// Max pixel size of the longest side of a scaled input image
const uint32_t TRAIN_MAX_SIZE = 1000;



// IOU >= thresh: positive example
const float TRAIN_RPN_POSITIVE_OVERLAP = 0.7f;
// IOU < thresh: negative example
const float TRAIN_RPN_NEGATIVE_OVERLAP = 0.3f;
// If an anchor statisfied by positive and negative conditions set to negative
const bool TRAIN_RPN_CLOBBER_POSITIVES = false;
// Max number of oreground examples
const float TRAIN_RPN_FG_FRACTION = 0.5f;
// Total number of examples
const uint32_t TRAIN_RPN_BATCHSIZE = 256;
// NMS threshold used on RPN proposalse
const float TRAIN_RPN_NMS_THRESH = 0.7f;
// Number of top scoring boxes to keep before apply NMS to RPN proposals
const uint32_t TRAIN_RPN_PRE_NMS_TOP_N = 12000;
//const uint32_t TRAIN_RPN_PRE_NMS_TOP_N = 6000;
// Number of top scoring boxes to keep after applying NMS to RPN proposals
const uint32_t TRAIN_RPN_POST_NMS_TOP_N = 2000;
//const uint32_t TRAIN_RPN_POST_NMS_TOP_N = 300;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at org image scale)
const uint32_t TRAIN_RPN_MIN_SIZE = 16;
// Deprecated (outside weights)
const std::vector<float> TRAIN_RPN_BBOX_INSIDE_WEIGHTS = {1.0f, 1.0f, 1.0f, 1.0f};
// Give the positive RPN examples weight of p * 1 / (num positives)
// and give negatives a weight of (1 - p)
// Set to -1.0 to use uniform example weighting
const float TRAIN_RPN_POSITIVE_WEIGHT = -1.0f;








// Scales to use during testing (can list multiple scales)
// Each scale is the pixel size of an image's shortest side
//const std::vector<uint32_t> TEST_SCALES = {600};
const std::vector<uint32_t> TEST_SCALES = {600};

// Max pixel size of the longest side of a scaled input image
const uint32_t TEST_MAX_SIZE = 1000;

// Overlap threshold used for non-maximum suppression (suppress boxes with
// IoU >= this threshold)
const float TEST_NMS = 0.3f;

// Experimental: treat the (K+1) units in the cls_score layer as linear
// predictors (trained, eg, with one-vs-rest SVMs).
const bool TEST_SVM = false;

// Test using bounding-box regressors
const bool TEST_BBOX_REG = true;

// Propose boxes
const bool TEST_HAS_RPN = true;

// Test using these proposals
const std::string TEST_PROPOSAL_METHOD = "selective_search";

// NMS threshold used on RPN proposals
const float TEST_RPN_NMS_THRESH = 0.7f;
// Number of top scoring boxes to keep before apply NMS to RPN proposals
const uint32_t TEST_RPN_PRE_NMS_TOP_N = 6000;
// Number of top scoring boxes to keep after applying NMS to RPN proposals
const uint32_t TEST_RPN_POST_NMS_TOP_N = 300;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
const uint32_t TEST_RPN_MIN_SIZE = 16;


// The mapping from image coordinates to feature map coordinates might cause
// some boxes that are distinct in image space to become identical in feature
// coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
// for identifying duplicate boxes.
// 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
const float TEST_DEDUP_BOXES = 1.0f / 16.0f;






static void loadPredefinedOrder(const std::string& path, std::vector<uint32_t>& order);



/**
 * 전달되는 cv::Mat을 그대로 사용한다. (직접 조작을 가한다.)
 * 원본을 보존해야 하는 경우 주의해서 사용할 것.
 */
template <typename Dtype>
void _displayBoxesOnImage(
		const std::string& windowName,
		cv::Mat& im,
		const float scale,
		const std::vector<std::vector<Dtype>>& boxes,
		const std::vector<uint32_t>& boxLabels,
		const std::vector<std::string>& boxLabelsText,
		const std::vector<cv::Scalar>& boxColors,
		const int boxOffset=0,
		const int numMaxBoxes=-1,
		const bool pause=true) {

	if (scale != 1.0f)
		cv::resize(im, im, cv::Size(), scale, scale, CV_INTER_LINEAR);

	uint32_t numBoxes = (numMaxBoxes<0)?
			(uint32_t)boxes.size():
			(std::min((uint32_t)numMaxBoxes, (uint32_t)boxes.size()));

	bool hasLabelText = (boxLabelsText.size() > 0) ? true : false;

	for (uint32_t i = 0; i < numBoxes; i++) {
		cv::rectangle(im, cv::Point(boxes[i][boxOffset+0], boxes[i][boxOffset+1]),
			cv::Point(boxes[i][boxOffset+2], boxes[i][boxOffset+3]),
			boxColors[boxLabels[i]-1], 2);

		std::string boxLabel = std::to_string(boxLabels[i]) +
				(hasLabelText ? (": " + boxLabelsText[i]) : "");
		cv::putText(im, boxLabel , cv::Point(boxes[i][boxOffset+0],
				boxes[i][boxOffset+1]+15.0f), 2, 0.5f, boxColors[boxLabels[i]-1]);

	}

	cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName, im);

	if (pause) {
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}

template <typename Dtype>
void displayBoxesOnImage(
		const std::string& windowName,
		const std::string& imagePath,
		const float scale,
		const std::vector<std::vector<Dtype>>& boxes,
		const std::vector<uint32_t>& boxLabels,
		const std::vector<std::string>& boxLabelsText,
		const std::vector<cv::Scalar>& boxColors,
		const int boxOffset=0,
		const int numMaxBoxes=-1,
		const bool pause=true) {

	cv::Mat im = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	_displayBoxesOnImage(windowName, im, scale, boxes, boxLabels, boxLabelsText,
			boxColors, boxOffset, numMaxBoxes, pause);
}

template <typename Dtype>
void displayBoxesOnImage(
		const std::string& windowName,
		cv::Mat& im,
		const float scale,
		const std::vector<std::vector<Dtype>>& boxes,
		const std::vector<uint32_t>& boxLabels,
		const std::vector<std::string>& boxLabelsText,
		const std::vector<cv::Scalar>& boxColors,
		const int boxOffset=0,
		const int numMaxBoxes=-1,
		const bool pause=true) {

	cv::Mat tempIm;
	im.copyTo(tempIm);

	_displayBoxesOnImage(windowName, tempIm, scale, boxes, boxLabels, boxLabelsText,
			boxColors, boxOffset, numMaxBoxes, pause);
}





template <typename PtrType, typename PrtType>
void printMat(cv::Mat& im) {
	std::cout << "rows: " << im.rows << ", cols: " << im.cols <<
				", channels: " << im.channels() << std::endl;

	const size_t numImElems = im.rows*im.cols*im.channels();
	const int rowElemSize = im.cols*im.channels();
	const int colElemSize = im.channels();

	for (int i = 0; i < im.rows; i++) {
		for (int j = 0; j < im.cols; j++) {
			std::cout << "[";
			for (int k = 0; k < im.channels(); k++) {
				std::cout << 
                    (PrtType)((PtrType*)im.data)[i*rowElemSize+j*colElemSize+k] << ",";
			}
			std::cout << "],";
		}
		std::cout << std::endl;
	}
}

template <typename Dtype>
static void printArray(const std::string& name, std::vector<Dtype>& array,
		const bool printName=true, const bool landscape=false) {
	if (printName) {
		std::cout << name << ": " << array.size() << std::endl;
	}

	const uint32_t arraySize = array.size();
	//std::cout << "[ ";
	for (uint32_t i = 0; i < arraySize; i++) {
		if (!landscape) {
			std::cout << i << ",,";
		}

		std::cout << array[i];

		if (landscape)
			std::cout << ", ";
		else
			std::cout << std::endl;

	}
	//std::cout << "]" << std::endl;
	std::cout << std::endl;
}

template <typename Dtype>
static void print2dArray(const std::string& name, std::vector<std::vector<Dtype>>& array,
		const bool printName=true) {
	if (printName) {
		std::cout << name << ": " << array.size();
		if (array.size() > 0)
			std::cout << " x " << array[0].size() << std::endl;
		else
			std::cout << " x 0" << std::endl;
	}

	const uint32_t arraySize = array.size();
	//std::cout << "[ " << std::endl;
	std::cout << std::endl;
	for (uint32_t i = 0; i < arraySize; i++) {
		//std::cout << i << "\t\t: ";
		std::cout << i << ",,";
		printArray(name, array[i], false, true);
	}
	//std::cout << "]" << std::endl;
	std::cout << std::endl;
}


template <typename Dtype>
static void printPrimitive(const std::string& name, const Dtype data,
		const bool printName=true) {
	std::cout << name << ": " << data << std::endl;
}


/**
 * 각 row의 vector에서 최대값을 추출
 */
template <typename Dtype>
static void np_maxByAxis(const std::vector<std::vector<Dtype>>& array,
		std::vector<Dtype>& result) {
	const uint32_t numArrayElem = array.size();
	SASSERT0(numArrayElem > 0);
	const uint32_t numAxisElem = array[0].size();
	SASSERT0(numAxisElem > 0);

	result.clear();
	result.resize(numArrayElem);
	Dtype max;
	for (uint32_t i = 0; i < numArrayElem; i++) {
		for (uint32_t j = 0; j < numAxisElem; j++) {
			if (j == 0) max = array[i][0];
			else if (array[i][j] > max) {
				max = array[i][j];
			}
		}
		result[i] = max;
	}
}

/**
 * 해당 axis에서 index에 해당하는 값을 조회
 */
template <typename Dtype>
static void np_array_value_by_index_array(const std::vector<std::vector<Dtype>>& array,
		const uint32_t axis, const std::vector<uint32_t>& inAxisIndex,
		std::vector<Dtype>& result) {
	SASSERT0(axis >= uint32_t(0) && axis < uint32_t(2));

	const uint32_t numArrayElem = array.size();
	SASSERT0(numArrayElem > 0);
	const uint32_t numAxisElem = array[0].size();
	SASSERT0(numAxisElem > 0);

	result.clear();

	if (axis == 0) {
		result.resize(numAxisElem);
		for (uint32_t i = 0; i < numAxisElem; i++) {
			result[i] = array[inAxisIndex[i]][i];
		}
	} else if (axis == 1) {
		result.resize(numArrayElem);
		for (uint32_t i = 0; i < numArrayElem; i++) {
			result[i] = array[i][inAxisIndex[i]];
		}
	}
}



/**
 * 각 column의 최대값을 추출
 */
template <typename Dtype>
static void np_array_max(const std::vector<std::vector<Dtype>>& array,
		std::vector<Dtype>& result) {
	const uint32_t numArrayElem = array.size();
	SASSERT0(numArrayElem > 0);
	const uint32_t numAxisElem = array[0].size();
	SASSERT0(numAxisElem > 0);

	result.clear();
	result.resize(numAxisElem);
	Dtype max;
	for (uint32_t j = 0; j < numAxisElem; j++) {
		for (uint32_t i = 0; i < numArrayElem; i++) {
			if (i == 0) max = array[0][j];
			else if (array[i][j] > max) {
				max = array[i][j];
			}
		}
		result[j] = max;
	}
}


template <typename Dtype>
static void np_scalar_divided_by_array(const Dtype scalar,
		const std::vector<Dtype>& array, std::vector<Dtype>& result) {

	const uint32_t arraySize = array.size();
	result.resize(arraySize);

	for (uint32_t i = 0; i < arraySize; i++) {
		result[i] = scalar / array[i];
	}
}

template <typename Dtype>
static void np_scalar_multiplied_by_array(const Dtype scalar,
		const std::vector<Dtype>& array, std::vector<Dtype>& result) {

	const uint32_t arraySize = array.size();
	result.resize(arraySize);

	for (uint32_t i = 0; i < arraySize; i++) {
		result[i] = scalar * array[i];
	}
}


template <typename Dtype1, typename Dtype2>
static void np_array_elementwise_mul(const std::vector<Dtype1>& a,
		const std::vector<Dtype2>& b, std::vector<Dtype2>& result) {

	SASSERT0(a.size() == b.size());

	const uint32_t arraySize = a.size();
	result.resize(arraySize);
	for (uint32_t i = 0; i < arraySize; i++) {
		result[i] = a[i] * b[i];
	}
}



template <typename Dtype>
static void np_sqrt(const std::vector<Dtype>& a, std::vector<Dtype>& result) {

	const uint32_t arraySize = a.size();
	result.resize(arraySize);

	for (uint32_t i = 0; i < arraySize; i++) {
		result[i] = std::sqrt(a[i]);
	}
}



template <typename Dtype>
static Dtype np_min(const std::vector<Dtype>& array,
		uint32_t begin, uint32_t end) {
	Dtype min;
	const uint32_t arraySize = array.size();
	SASSERT0(end < arraySize);

	for (uint32_t i = begin; i < end; i++) {
		if (i == 0) min = array[0];
		else if (array[i] < min) min = array[i];
	}
	return min;
}

template <typename Dtype>
static Dtype np_max(const std::vector<Dtype>& array,
		uint32_t begin, uint32_t end) {
	Dtype max;
	const uint32_t arraySize = array.size();
	SASSERT0(end < arraySize);

	for (uint32_t i = begin; i < end; i++) {
		if (i == 0) max = array[0];
		else if (array[i] > max) max = array[i];
	}
	return max;
}



template <typename Dtype>
static void np_maximum(const Dtype value, const std::vector<Dtype>& array,
		const std::vector<uint32_t>& index,	const uint32_t offset,
		std::vector<Dtype>& result) {
	const uint32_t indexSize = index.size();
	result.resize(indexSize-offset);

	for (uint32_t i = offset; i < indexSize; i++) {
		result[i-offset] = (value >= array[index[i]]) ?
				value :
				array[index[i]];
	}
}

template <typename Dtype>
static void np_minimum(const Dtype value, const std::vector<Dtype>& array,
		const std::vector<uint32_t>& index,	const uint32_t offset,
		std::vector<Dtype>& result) {
	const uint32_t indexSize = index.size();
	result.resize(indexSize-offset);

	for (uint32_t i = offset; i < indexSize; i++) {
		result[i-offset] = (value <= array[index[i]]) ?
				value :
				array[index[i]];
	}
}


/*
template <typename Dtype>
static void np_maximum(const Dtype value, const std::vector<Dtype>& array,
		std::vector<Dtype>& result) {
	const uint32_t arraySize = array.size();
	result.resize(arraySize);

	for (uint32_t i = 0; i < arraySize; i++) {
		result[i] = std::max(value, array[i]);
	}
}
*/


template <typename Dtype>
static void np_argmax(const std::vector<std::vector<Dtype>>& array, const uint32_t axis,
		std::vector<uint32_t>& result) {
	SASSERT0(axis >= uint32_t(0) && axis < uint32_t(2));

	const uint32_t numArrayElem = array.size();
	SASSERT0(numArrayElem >= 1);
	const uint32_t numAxisElem = array[0].size();
	SASSERT0(numAxisElem >= 1);

	result.clear();

	if (axis == 0) {
		result.resize(numAxisElem);
		Dtype max;
		uint32_t maxIndex;
		for (uint32_t i = 0; i < numAxisElem; i++) {
			for (uint32_t j = 0; j < numArrayElem; j++) {
				if (j == 0) {
					max = array[0][i];
					maxIndex = 0;
				}
				else if (array[j][i] > max) {
					max = array[j][i];
					maxIndex = j;
				}
			}
			result[i] = maxIndex;
		}
	} else if (axis == 1) {
		result.resize(numArrayElem);
		Dtype max;
		uint32_t maxIndex;
		for (uint32_t i = 0; i < numArrayElem; i++) {
			for (uint32_t j = 0; j < numAxisElem; j++) {
				if (j == 0) {
					max = array[i][0];
					maxIndex = 0;
				}
				else if (array[i][j] > max) {
					max = array[i][j];
					maxIndex = j;
				}
			}
			result[i] = maxIndex;
		}
	}
}

/**
 * np_where 단수 조건 버전, compare 옵션 있음.
 */
template <typename Dtype, typename Dtype2>
static void np_where_s(const std::vector<Dtype>& array, const uint32_t comp,
    const Dtype2 criteria, std::vector<uint32_t>& result) {
	const uint32_t numArrayElem = array.size();
	result.clear();

	if (numArrayElem < 1)
		return;

	switch (comp) {
	case GT:
		for (uint32_t i = 0; i < numArrayElem; i++) 
            if (array[i] > criteria) result.push_back(i);
		break;
	case GE:
		for (uint32_t i = 0; i < numArrayElem; i++)
            if (array[i] >= criteria) result.push_back(i);
		break;
	case EQ:
		for (uint32_t i = 0; i < numArrayElem; i++)
            if (array[i] == criteria) result.push_back(i);
		break;
	case LE:
		for (uint32_t i = 0; i < numArrayElem; i++)
            if (array[i] <= criteria) result.push_back(i);
		break;
	case LT:
		for (uint32_t i = 0; i < numArrayElem; i++)
            if (array[i] < criteria) result.push_back(i);
		break;
	case NE:
		for (uint32_t i = 0; i < numArrayElem; i++)
            if (array[i] != criteria) result.push_back(i);
		break;
	default:
		std::cout << "invalid comp: " << comp << std::endl;
		exit(1);
	}
}

/**
 * np_where 단수 조건 버전, equality에 대해서만 테스트
 */
template <typename Dtype>
static void np_where_s(const std::vector<std::vector<Dtype>>& array, const Dtype criteria,
		const uint32_t loc, std::vector<uint32_t>& result) {
	const uint32_t numArrayElem = array.size();
	SASSERT0(numArrayElem > 0);
	SASSERT0(loc < array[0].size());

	result.clear();
	for (uint32_t i = 0; i < numArrayElem; i++) {
		if (array[i][loc] == criteria) {
			result.push_back(i);
		}
	}
}

/**
 * np_where 단수 조건 버전, equality에 대해서만 테스트
 * 한 row의 기준값과 array값에 대해 각각의 axis에 대해 하나라도 일치할 경우 찾은 것으로 판정
 */
template <typename Dtype>
static void np_where_s(const std::vector<std::vector<Dtype>>& array,
		const std::vector<Dtype>& criteria,
		std::vector<uint32_t>& result) {
	result.clear();
	if (array.size() < 1)
		return;

	SASSERT0(array[0].size() == criteria.size());

	const uint32_t numArrayElem = array.size();
	const uint32_t numAxisElem = criteria.size();

	for (uint32_t i = 0; i < numArrayElem; i++) {
		for (uint32_t j = 0; j < numAxisElem; j++) {
			if (array[i][j] == criteria[j]) {
				result.push_back(i);
				break;
			}
		}
	}
}

/**
 * np_where 복수 조건 버전
 */
static void np_where(std::vector<float>& array, const std::vector<uint32_t>& comp,
		const std::vector<float>& criteria, std::vector<uint32_t>& result) {

	if (comp.size() < 1 ||
			criteria.size() < 1 ||
			comp.size() != criteria.size() ||
			array.size() < 1) {

		std::cout << "invalid array dimension ... " << std::endl;
		exit(1);
	}

	result.clear();

	const uint32_t numComps = comp.size();
	const uint32_t numArrayElem = array.size();
	bool cond;
	for (uint32_t i = 0; i < numArrayElem; i++) {
		cond = true;

		for (uint32_t j = 0; j < numComps; j++) {
			switch (comp[j]) {
			case GT: if (array[i] <= criteria[j])	cond = false; break;
			case GE: if (array[i] <	criteria[j])	cond = false; break;
			case EQ: if (array[i] != criteria[j])	cond = false; break;
			case LE: if (array[i] > criteria[j])	cond = false; break;
			case LT: if (array[i] >= criteria[j])	cond = false; break;
			default:
				std::cout << "invalid comp: " << comp[j] << std::endl;
				exit(1);
			}
			if (!cond) break;
		}
		if (cond) result.push_back(i);
	}
}

template <typename Dtype>
static void np_tile(const std::vector<Dtype>& array, const uint32_t repeat,
		std::vector<std::vector<Dtype>>& result) {

	result.clear();
	for (uint32_t i = 0; i < repeat; i++) {
		result.push_back(array);
	}
}

static uint32_t np_round(float a) {
	uint32_t lower = static_cast<uint32_t>(floorf(a) + 0.1f);
	if (lower % 2 == 0) return lower;

	uint32_t upper = static_cast<uint32_t>(ceilf(a) + 0.1f);
	return upper;
}

static void np_round(const std::vector<float>& a, std::vector<uint32_t>& result) {
	const uint32_t arraySize = a.size();
	result.resize(arraySize);

	for (uint32_t i = 0; i < arraySize; i++) {
		//result[i] = np_round(a[i]);
		result[i] = roundf(a[i]);
	}
}

static void npr_randint(const uint32_t lb, const uint32_t ub, const uint32_t size,
		std::vector<uint32_t>& result) {

	//srand((uint32_t)time(NULL));
	result.resize(size);

	for (uint32_t i = 0; i < size; i++) {
		result[i] = lb + rand()%(ub-lb);
	}
}

template <typename Dtype>
static void npr_choice(const std::vector<Dtype>& array, const uint32_t size,
		std::vector<Dtype>& result) {

	const uint32_t arraySize = array.size();
	SASSERT0(size <= arraySize);

	std::vector<uint32_t> indexArray(arraySize);
	iota(indexArray.begin(), indexArray.end(), 0);
	random_shuffle(indexArray.begin(), indexArray.end());

	result.resize(size);
	for (uint32_t i = 0; i < size; i++) {
		result[i] = array[indexArray[i]];
	}
	//printArray("result-final", result);
}

static void np_arange(int start, int stop, std::vector<uint32_t>& result) {
	SASSERT0(start < stop);

	result.clear();
	for (int i = start; i < stop; i++) {
		result.push_back(i);
	}
}

template <typename Dtype>
static void py_arrayElemsWithArrayInds(const std::vector<Dtype>& array,
		const std::vector<uint32_t>& inds, std::vector<Dtype>& result) {

	const uint32_t arraySize = array.size();
	const uint32_t indsSize = inds.size();

	//result.clear();
	result.resize(indsSize);
	for (uint32_t i = 0; i < indsSize; i++) {
		SASSERT0(inds[i] < arraySize);
		result[i] = array[inds[i]];
	}
}

template <typename Dtype>
static Dtype vec_max(const std::vector<Dtype>& array) {
	Dtype max;
	const uint32_t arraySize = array.size();
	for (uint32_t i = 0; i < arraySize; i++) {
		if (i == 0) max = array[i];
		else if (max < array[i]) max = array[i];
	}
	return max;
}

template <typename Dtype, typename Dtype2>
static void fillDataWith2dVec(const std::vector<std::vector<Dtype>>& array,
		Data<Dtype2>* data) {
	if (array.size() < 1) {
		SASSERT0(array.size() > 0);
	}

	const uint32_t dim1 = array.size();
	const uint32_t dim2 = array[0].size();

	data->reshape({1, 1, dim1, dim2});
	Dtype2* dataPtr = data->mutable_host_data();

	for (uint32_t i = 0; i < dim1; i++) {
		memcpy(dataPtr + i*dim2, &array[i][0], sizeof(Dtype)*dim2);
	}
}

/*
template <typename Dtype, typename Dtype2>
static void fillDataWith2dVec(const std::vector<std::vector<Dtype>>& array,
		const std::vector<uint32_t>& transpose,	Data<Dtype2>* data) {
	SASSERT0(array.size() > 0);
	const uint32_t dim1 = array.size();
	const uint32_t dim2 = array[0].size();

	const std::vector<uint32_t>& shape = data->getShape();
	SASSERT0(shape[3]%dim2 == 0);

	const uint32_t tBatchSize = shape[transpose[1]]*shape[transpose[2]]*shape[transpose[3]];
	const uint32_t tHeightSize = shape[transpose[2]]*shape[transpose[3]];
	const uint32_t tWidthSize = shape[transpose[3]];

	Dtype2* dataPtr = data->mutable_host_data();
	const uint32_t shape3 = shape[3] / dim2;
	const uint32_t batchSize = shape[1]*shape[2]*shape3;
	const uint32_t heightSize = shape[2]*shape3;
	const uint32_t widthSize = shape3;

	uint32_t s[4];
	uint32_t& ts0 = s[transpose[0]];
	uint32_t& ts1 = s[transpose[1]];
	uint32_t& ts2 = s[transpose[2]];
	uint32_t& ts3 = s[transpose[3]];

	uint32_t q, r;
	// batch
	for (s[0] = 0; s[0] < shape[0]; s[0]++) {
		// height
		for (s[1] = 0; s[1] < shape[1]; s[1]++) {
			// width
			for (s[2] = 0; s[2] < shape[2]; s[2]++) {
				// Anchors
				for (s[3] = 0; s[3] < shape[3]; s[3]++) {
					q = s[3] / dim2;
					r = s[3] % dim2;
					dataPtr[ts0*tBatchSize+ts1*tHeightSize+ts2*tWidthSize+ts3] =
							array[s[0]*batchSize+s[1]*heightSize+s[2]*widthSize+q][r];
				}
			}
		}
	}
}

template <typename Dtype, typename Dtype2>
static void fillDataWith1dVec(const std::vector<Dtype>& array,
		const std::vector<uint32_t>& transpose,
		Data<Dtype2>* data) {
	SASSERT0(array.size() > 0);
	const uint32_t dim1 = array.size();

	const std::vector<uint32_t>& shape = data->getShape();

	Dtype2* dataPtr = data->mutable_host_data();
	const uint32_t batchSize = shape[1]*shape[2]*shape[3];
	const uint32_t heightSize = shape[2]*shape[3];
	const uint32_t widthSize = shape[3];

	const uint32_t tBatchSize = shape[transpose[1]]*shape[transpose[2]]*shape[transpose[3]];
	const uint32_t tHeightSize = shape[transpose[2]]*shape[transpose[3]];
	const uint32_t tWidthSize = shape[transpose[3]];

	uint32_t s[4];
	uint32_t& ts0 = s[transpose[0]];
	uint32_t& ts1 = s[transpose[1]];
	uint32_t& ts2 = s[transpose[2]];
	uint32_t& ts3 = s[transpose[3]];

	// batch
	for (s[0] = 0; s[0] < shape[0]; s[0]++) {
		// height
		for (s[1] = 0; s[1] < shape[1]; s[1]++) {
			// width
			for (s[2] = 0; s[2] < shape[2]; s[2]++) {
				// Anchors
				for (s[3] = 0; s[3] < shape[3]; s[3]++) {
					dataPtr[ts0*tBatchSize+ts1*tHeightSize+ts2*tWidthSize+ts3]
					        = Dtype2(array[s[0]*batchSize+s[1]*heightSize+s[2]*widthSize+s[3]]);
				}
			}
		}
	}
}
*/

template <typename Dtype, typename Dtype2>
static void fill1dVecWithData(Data<Dtype>* data,
		std::vector<Dtype2>& array) {

	const std::vector<uint32_t>& dataShape = data->getShape();
	SASSERT0(dataShape[0] == 1);
	SASSERT0(dataShape[1] == 1);
	SASSERT0(dataShape[2] == 1);

	const uint32_t dim1 = dataShape[3];
	array.resize(dim1);
	const Dtype* dataPtr = data->host_data();

	for (uint32_t i = 0; i < dim1; i++) {
		array[i] = Dtype2(dataPtr[i]);
	}
}

template <typename Dtype, typename Dtype2>
static void fill2dVecWithData(Data<Dtype>* data,
		std::vector<std::vector<Dtype2>>& array) {

	const std::vector<uint32_t>& dataShape = data->getShape();

	int dim1 = 0;
	int dim2 = 0;
	for (int i = 0; i < dataShape.size(); i++) {
		if (dataShape[i] > 1) {
			if (dim1 == 0) dim1 = dataShape[i];
			else if (dim2 == 0) dim2 = dataShape[i];
			else {
				std::cout << "fill2dVecWithData: invalid data shape ... " << std::endl;
			}
		}
	}

	if (dim1 == 0 || dim2 == 0) {
		dim1 = dataShape[2];
		dim2 = dataShape[3];
	}

	SASSERT0(dim1*dim2 == data->getCount());


	//SASSERT0(dataShape[0] == 1);
	//SASSERT0(dataShape[1] == 1);

	array.resize(dim1);
	const Dtype* dataPtr = data->host_data();

	for (uint32_t i = 0; i < dim1; i++) {
		array[i].resize(dim2);
		for (uint32_t j = 0; j < dim2; j++) {
			array[i][j] = Dtype2(dataPtr[i*dim2+j]);
		}
	}
}


template <typename Dtype>
static std::vector<Dtype> vec_keep_by_index(const std::vector<Dtype>& array,
		const std::vector<uint32_t>& index) {

	const uint32_t numKeep = index.size();
	std::vector<Dtype> result(numKeep);
	for (uint32_t i = 0; i < numKeep; i++) {
		result[i] = array[index[i]];
	}
	return result;
}

template <typename Dtype>
static void vec_argsort(const std::vector<Dtype>& array, std::vector<uint32_t>& arg,
    int order=0) {
	SASSERT0(order == 0 || order == 1);

	const uint32_t arraySize = array.size();
	//arg.resize(arraySize);
	//iota(arg.begin(), arg.end(), 0);

	if (order == 0) {
		sort(arg.begin(), arg.end(), [&array](size_t i1, size_t i2) {
			return array[i1] > array[i2];
		});
	} else if (order == 1) {
		sort(arg.begin(), arg.end(), [&array](size_t i1, size_t i2) {
			return array[i1] < array[i2];
		});
	}
}

template <typename Dtype>
static void vec_2d_pad(const uint32_t leftPad, std::vector<std::vector<Dtype>>& array) {
	if (array.size() < 1)
		return;

	const uint32_t outerSize = array.size();
	const uint32_t innerSize = array[0].size() + leftPad;

	for (uint32_t i = 0; i < outerSize; i++) {
		std::vector<Dtype>& item = array[i];
		item.resize(innerSize);

		for (uint32_t j = innerSize-1; j > leftPad-1; j--) {
			item[j] = item[j-leftPad];
		}
		for (uint32_t j = 0; j < leftPad; j++) {
			item[j] = 0;
		}
	}
}

/*
static std::string cv_type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}
*/


static void nms(std::vector<std::vector<float>>& proposals, std::vector<float>& scores,
		const float thresh, std::vector<uint32_t>& keep) {
	//SASSERT0(proposals.size() > 0 && proposals.size() == scores.size());
    if (proposals.size() == 0) {
        COLD_LOG(ColdLog::WARNING, true, "there is no proposal in proposal layer");
        keep.clear();
        return;
    }

	//const float thresh = 0.7f;
	const int device_id = 0;
	const int boxes_num = proposals.size();
	const int boxes_dim = proposals[0].size() + 1;

	int num_out;
	int _keep[boxes_num];
	//float scores[boxes_num];
	float sorted_dets[boxes_num][boxes_dim];
	//vector<int> order(boxes_num);

	//keep.resize(boxes_num);

	std::vector<uint32_t> order(scores.size());
#if !SOOOA_DEBUG
	iota(order.begin(), order.end(), 0);
	vec_argsort(scores, order);

	/*
	std::cout << "nms sort result: " << std::endl;
	for (int i = 0; i < std::min((int)scores.size(), (int)30); i++) {
		std::cout << "\tscore: " << scores[order[i]] << std::endl;
	}
	*/

#else
	//const std::string path = "/home/jkim/Dev/data/numpy_array/order.npz";
	//loadPredefinedOrder(path, order);
	// 이미 정렬된 score를 기준으로 정렬하므로 0부터 순차적인 배열이 만들어진다.
	iota(order.begin(), order.end(), 0);
	//printArray("order", order);
#endif

	for (int i = 0; i < boxes_num; i++) {
		sorted_dets[i][0] = proposals[order[i]][0];
		sorted_dets[i][1] = proposals[order[i]][1];
		sorted_dets[i][2] = proposals[order[i]][2];
		sorted_dets[i][3] = proposals[order[i]][3];
		sorted_dets[i][4] = scores[order[i]];
	}

	_nms(&_keep[0], &num_out, &sorted_dets[0][0], boxes_num, boxes_dim, thresh, device_id);

	keep.resize(num_out);
	for (int i = 0; i < num_out; i++)
		keep[i] = order[_keep[i]];

	//printArray("keep", keep);

	/*
	for (int i = 0; i < num_out; i++) {
		std::cout << i << ": ";
		for (int j = 0; j < boxes_dim; j++) {
			if (j < boxes_dim - 1)
				std::cout << proposals[order[keep[i]]][j] << ", ";
			else
				std::cout << scores[order[keep[i]]] << std::endl;
		}
	}
	*/
}


static void loadPredefinedOrder(const std::string& path, std::vector<uint32_t>& order) {
	cnpy::npz_t cnpy_order = cnpy::npz_load(path);
	for (cnpy::npz_t::iterator itr = cnpy_order.begin(); itr != cnpy_order.end(); itr++) {
		cnpy::NpyArray npyArray = itr->second;

		int n = itr->second.shape[0];
		order.resize(n);

		long* srcPtr = (long*)npyArray.data;
		for (int i = 0; i < n; i++) {
			order[i] = uint32_t(srcPtr[i]);
		}
	}
}






struct Size {
	uint32_t width;
	uint32_t height;
	uint32_t depth;

	void print() {
		std::cout << "Size:" << std::endl <<
				"\twidth: " << width << std::endl <<
				"\theight: " << height << std::endl <<
				"\tdepth: " << depth << std::endl;
	}
};

struct Object {
	std::string name;
	uint32_t label;
	uint32_t difficult;
	uint32_t xmin;
	uint32_t ymin;
	uint32_t xmax;
	uint32_t ymax;

	void print() {
		std::cout << "Object: " << std::endl <<
				"\tname: " << name << std::endl <<
				"\tlabel: " << label << std::endl <<
				"\tdifficult: " << difficult << std::endl <<
				"\txmin: " << xmin << std::endl <<
				"\tymin: " << ymin << std::endl <<
				"\txmax: " << xmax << std::endl <<
				"\tymax: " << ymax << std::endl;
	}
};

struct Annotation {
	std::string filename;
	Size size;
	std::vector<Object> objects;

	void print() {
		std::cout << "Annotation:" << std::endl <<
				"\tfilename: " << filename << std::endl;
		size.print();
		for (uint32_t i = 0; i < objects.size(); i++) {
			objects[i].print();
		}
	}
};

#endif /* FRCNN_COMMON_H_ */
