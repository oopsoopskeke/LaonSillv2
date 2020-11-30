/*
 * ImTransforms.h
 *
 *  Created on: Sep 4, 2017
 *      Author: jkim
 */

#ifndef IMTRANSFORMS_H_
#define IMTRANSFORMS_H_

#include <opencv2/core/core.hpp>

#include "LayerPropParam.h"
#include "Datum.h"



// Generate random number given the probablities for each number.
int roll_weighted_die(const std::vector<float>& probabilities);


void UpdateBBoxByResizePolicy(const ResizeParam& param, const int oldWidth,
		const int oldHeight, NormalizedBBox* bbox);





void InferNewSize(const ResizeParam& resizeParam, const int oldWidth, const int oldHeight,
		int* newWidth, int* newHeight);








cv::Mat ApplyResize(const cv::Mat& in_img, const ResizeParam& param);

cv::Mat ApplyNoise(const cv::Mat& in_img, const NoiseParam& param);





void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img, const float brightness_prob,
		const float brightness_delta);

void AdjustBrightness(const cv::Mat& in_img, const float delta, cv::Mat* out_img);

void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img, const float contrast_prob,
		const float lower, const float upper);

void AdjustContrast(const cv::Mat& in_img, const float delta, cv::Mat* out_img);

void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img, const float saturation_prob,
		const float lower, const float upper);

void AdjustSaturation(const cv::Mat& in_img, const float delta, cv::Mat* out_img);

void RandomHue(const cv::Mat& in_img, cv::Mat* out_img, const float hue_prob,
		const float hue_delta);

void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img);

void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
		const float random_order_prob);

cv::Mat ApplyDistort(const cv::Mat& in_img, const DistortionParam& param);






cv::Mat AspectKeepingResizeAndPad(const cv::Mat& in_img, const int newWidth,
		const int newHeight, const int padType,  const cv::Scalar padVal,
		const int interpMode);

cv::Mat AspectKeepingResizeBySmall(const cv::Mat& in_img, const int newWidth,
		const int newHeight, const int interpMode);













#endif /* IMTRANSFORMS_H_ */
