/*
 * IO.h
 *
 *  Created on: Jun 29, 2017
 *      Author: jkim
 */

#ifndef IO_H_
#define IO_H_

#include <string>
#include <opencv2/core/core.hpp>
#include <map>

#include "Datum.h"
#include "Data.h"

bool ReadImageToDatum(const std::string& filename, const std::vector<int>& label,
		const int height, const int width, const int min_dim, const int max_dim,
		const bool channel_separated, const bool is_color, const std::string& encoding,
		class Datum* datum);

bool ReadRichImageToAnnotatedDatum(const std::string& filename, const std::string& labelname,
		const int height, const int width, const int min_dim, const int max_dim,
		const bool is_color, const std::string& encoding, const AnnotationType type,
		const std::string& labeltype, const std::map<std::string, int>& name_to_label,
		AnnotatedDatum* anno_datum);

/*
bool ReadImageToDatum(const std::string& filename, const int label, const int height,
		const int width, const bool is_color, const std::string& encoding, Datum* datum) {
	//return ReadImageToDatum(filename, label, height, width, 0, 0, is_color, encoding, datum);
	return false;
}
*/

cv::Mat ReadImageToCVMat(const std::string& filename, const int height, const int width,
		const int min_dim, const int max_dim, const bool is_color,
		int* imgHeight = NULL, int* imgWidth = NULL);

cv::Mat ReadImageToCVMat(const std::string& filename, const int height, const int width);



/**
 * OpenCV cv::Mat을 Datum 객체에 fill
 * @param channel_separated
 * 		true: all B / all G / all R 구조로 channel을 나누어서 저장
 * 		false: channel분리하지 않고 opencv가 제공하는 포맷 그대로 저장
 */
void CVMatToDatum(const cv::Mat& cv_img, const bool channel_separated, Datum* datum);

bool ReadFileToDatum(const std::string& filename, const int label, Datum* datum);

/**
 * OpenCV cv::Mat의 데이터를 지정된 방법으로 encoding하여 Datum 객체에 저장
 */
void EncodeCVMatToDatum(const cv::Mat& cv_img, const std::string& encoding, Datum* datum);



/**
 * Datum 객체의 이미지 정보를 OpenCV cv::Mat로 변환
 */
cv::Mat DecodeDatumToCVMat(const Datum* datum, bool is_color, bool channel_separated);


bool ReadXMLToAnnotatedDatum(const std::string& labelname, const int img_height,
		const int img_width, const std::map<std::string, int>& name_to_label,
		AnnotatedDatum* anno_datum);



void GetImageSize(const std::string& filename, int* height, int* width);




template <typename Dtype>
void CheckCVMatDepthWithDtype(const cv::Mat& im);

template <typename Dtype>
void ConvertHWCToCHW(const int channels, const int height, const int width, const Dtype* src,
		Dtype* dst);


/**
 * HWC CVMat의 data를 Datum의 데이터 구조인 CHW로 변환
 * CVMat으로부터 Datum으로 변환할 수 있다.
 */
template <typename Dtype>
void ConvertHWCCVToCHW(const cv::Mat& im, Dtype* dst);

/**
 * HWC 구조의 데이터를 CHW 구조의 데이터로 변환
 */
template <typename Dtype>
void ConvertHWCToHWC(const int channels, const int height, const int width, const Dtype* src,
		Dtype* dst);

/**
 * HWC CVMat의 data를 HWC로 그대로 복사
 */
template <typename Dtype>
void ConvertHWCCVToHWC(const cv::Mat& im, Dtype* dst);

/**
 * @brief CHW 구조의 데이터를 HWC 구조의 데이터로 변환
 */
template <typename Dtype>
void ConvertCHWToHWC(const int channels, const int height, const int width, const Dtype* src,
		Dtype* dst);
/**
 * CHW Datum의 data를 CVMat의 데이터 구조인 HWC로 변환
 * 변환 후, cv::Mat 생성자를 통해 cv::Mat 객체를 생성할 수 있다.
 */
void ConvertCHWDatumToHWC(const Datum* datum, uchar* dst);


template <typename Dtype>
cv::Mat ConvertCHWDataToHWCCV(Data<Dtype>* data, const int batchIdx);






template <typename Dtype>
void PrintImageData(const int channels, const int height, const int width, const Dtype* ptr,
		bool hwc);
void PrintCVMatData(const cv::Mat& mat);
void PrintDatumData(const Datum* datum, bool hwc);





#endif /* IO_H_ */
