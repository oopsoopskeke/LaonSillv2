/*
 * DataTransformer.h
 *
 *  Created on: Jul 19, 2017
 *      Author: jkim
 */

#ifndef DATATRANSFORMER_H_
#define DATATRANSFORMER_H_

#include <opencv2/core/core.hpp>

#include "Datum.h"
#include "Data.h"
#include "LayerPropParam.h"




template <typename Dtype>
class DataTransformer {
public:
	DataTransformer(DataTransformParam* param = NULL);
	virtual ~DataTransformer();


	void transformWithMeanScale(Datum* datum, const std::vector<float>& mean,
			const float scale, Dtype* dataPtr);

	void transform(Datum* datum, Dtype* dataPtr);
	void transform(cv::Mat& im, Data<Dtype>* data, int batchIdx = 0);
	void transform(const Datum* datum, Data<Dtype>* data, int batchIdx);

	void transform(AnnotatedDatum* annoDatum, Data<Dtype>* data, int batchIdx,
			std::vector<AnnotationGroup>& transformedAnnoVec);
	void transform(AnnotatedDatum* annoDatum, Data<Dtype>* data, int batchIdx,
			std::vector<AnnotationGroup>& transformedAnnoVec, bool* doMirror);
	void transform(AnnotatedDatum* annoDatum, Data<Dtype>* data, int batchIdx,
			std::vector<AnnotationGroup>* transformedAnnoGroupAll, bool* doMirror);
	void transform(const Datum* datum, Data<Dtype>* data, int batchIdx,
			NormalizedBBox* cropBBox, bool* doMirror);
	void transform(const cv::Mat& cv_img, Data<Dtype>* data, int batchIdx,
			NormalizedBBox* cropBBox, bool* doMirror);
	void transform(const Datum* datum, Dtype* transformedData, NormalizedBBox* cropBBox,
			bool* doMirror);



	std::vector<uint32_t> inferDataShape(const Datum* datum);
	std::vector<uint32_t> inferDataShape(const cv::Mat& cv_img);
	std::vector<uint32_t> inferDataShape(const int channels, const int height, 
            const int width);

	void distortImage(const Datum* datum, Datum* distortDatum);

	void expandImage(const cv::Mat& img, const float expandRatio, NormalizedBBox* expandBBox,
			cv::Mat* expand_img);
	void expandImage(const Datum* datum, const float expandRatio, NormalizedBBox* expandBBox,
			Datum* expandDatum);
	void expandImage(const AnnotatedDatum* annoDatum, AnnotatedDatum* expandedAnnoDatum);

	void transformAnnotation(const AnnotatedDatum* annoDatum, const bool doResize,
			const NormalizedBBox& cropBBox, const bool doMirror,
			std::vector<AnnotationGroup>& transformedAnnoGroupAll);

	void cropImage(const Datum* datum, const NormalizedBBox& bbox,
			Datum* cropDatum);
	void cropImage(const AnnotatedDatum* annoDatum, const NormalizedBBox& bbox,
			AnnotatedDatum* croppedAnnoDatum);
	void cropImage(const cv::Mat& img, const NormalizedBBox& bbox, cv::Mat* crop_img);


private:
	int rand(int n);



public:
	DataTransformParam param;
	bool hasMean;
	bool hasCropSize;
	bool hasScale;
	bool hasMirror;

};




/**
 * from old DataTransformer.h
 */
template <typename Dtype>
void transformInv(const int num, const int singleImageSize,
		const int imageHeight, const int imageWidth,
		const int height, const int width, const std::vector<Dtype>& pixelMeans,
		const Dtype* dataData, Data<Dtype>& temp, cv::Mat& im);












#endif /* DATATRANSFORMER_H_ */
