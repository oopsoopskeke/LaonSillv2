#include <opencv2/imgproc/imgproc.hpp>

#include "ImTransforms.h"
#include "MathFunctions.h"
#include "SysLog.h"

using namespace std;




int roll_weighted_die(const vector<float>& probabilities) {
	vector<float> cumulative;
	std::partial_sum(&probabilities[0], &probabilities[0] + probabilities.size(),
			std::back_inserter(cumulative));
	float val;
	soooa_rng_uniform(1, static_cast<float>(0), cumulative.back(), &val);

	// Find the position within the sequence and add 1
	return (std::lower_bound(cumulative.begin(), cumulative.end(), val) - cumulative.begin());
}


void UpdateBBoxByResizePolicy(const ResizeParam& param, const int oldWidth,
		const int oldHeight, NormalizedBBox* bbox) {

	float newHeight = param.height;
	float newWidth = param.width;
	float origAspect = static_cast<float>(oldWidth) / oldHeight;
	float newAspect = newWidth / newHeight;

	float xmin = bbox->xmin * oldWidth;
	float ymin = bbox->ymin * oldHeight;
	float xmax = bbox->xmax * oldWidth;
	float ymax = bbox->ymax * oldHeight;
	float padding;
	switch (param.resizeMode) {
	case WARP:
		xmin = std::max(0.f, xmin * newWidth / oldWidth);
		xmax = std::min(newWidth, xmax * newWidth / oldWidth);
		ymin = std::max(0.f, ymin * newHeight / oldHeight);
		ymax = std::min(newHeight, ymax * newHeight / oldHeight);
		break;
	case FIT_LARGE_SIZE_AND_PAD:
		if (origAspect > newAspect) {
			padding = (newHeight - newWidth / origAspect) / 2;
			xmin = std::max(0.f, xmin * newWidth / oldWidth);
			xmax = std::min(newWidth, xmax * newWidth / oldWidth);
			ymin = ymin * (newHeight - 2 * padding) / oldHeight;
			ymin = padding + std::max(0.f, ymin);
			ymax = ymax * (newHeight - 2 * padding) / oldHeight;
			ymax = padding + std::min(newHeight, ymax);
		} else {
			padding = (newWidth - origAspect * newHeight) / 2;
			xmin = xmin * (newWidth - 2 * padding) / oldWidth;
			xmin = padding + std::max(0.f, xmin);
			xmax = xmax * (newWidth - 2 * padding) / oldWidth;
			xmax = padding + std::min(newWidth, xmax);
			ymin = std::max(0.f, ymin * newHeight / oldHeight);
			ymax = std::min(newHeight, ymax * newHeight / oldHeight);
		}
		break;
	case FIT_SMALL_SIZE:
		if (origAspect < newAspect) {
			newHeight = newWidth / origAspect;
		} else {
			newWidth = origAspect * newHeight;
		}
		xmin = std::max(0.f, xmin * newWidth / oldWidth);
		xmax = std::min(newWidth, xmax * newWidth / oldWidth);
		ymin = std::max(0.f, ymin * newHeight / oldHeight);
		ymax = std::min(newHeight, ymax * newHeight / oldHeight);
		break;
	default:
		SASSERT(false, "Unknown resize mode.");
		break;
	}
	bbox->xmin = xmin / newWidth;
	bbox->ymin = ymin / newHeight;
	bbox->xmax = xmax / newWidth;
	bbox->ymax = ymax / newHeight;
}


void InferNewSize(const ResizeParam& resizeParam, const int oldWidth, const int oldHeight,
		int* newWidth, int* newHeight) {
	int height = resizeParam.height;
	int width = resizeParam.width;
	float origAspect = static_cast<float>(oldWidth) / oldHeight;
	float aspect = static_cast<float>(width) / height;

	switch (resizeParam.resizeMode) {
	case WARP:
		break;
	case FIT_LARGE_SIZE_AND_PAD:
		break;
	case FIT_SMALL_SIZE:
		if (origAspect < aspect) {
			height = static_cast<int>(width / origAspect);
		} else {
			width = static_cast<int>(origAspect * height);
		}
		break;
	default:
		SASSERT(false, "Unknown resize mode.");
		break;
	}
	*newHeight = height;
	*newWidth = width;
}



cv::Mat ApplyResize(const cv::Mat& in_img, const ResizeParam& param) {
	cv::Mat out_img;

	// Reading parameters
	const int newHeight = param.height;
	const int newWidth = param.width;

	int padMode = cv::BORDER_CONSTANT;
	switch (param.padMode) {
	case CONSTANT:
		break;
	case MIRRORED:
		padMode = cv::BORDER_REFLECT101;
		break;
	case REPEAT_NEAREST:
		padMode = cv::BORDER_REPLICATE;
		break;
	default:
		SASSERT(false, "Unknown pad mode.");
		break;
	}

	int interpMode = cv::INTER_LINEAR;
	int numInterpMode = param.interpMode.size();
	if (numInterpMode > 0) {
		vector<float> probs(numInterpMode, 1.f / numInterpMode);
		int probNum = roll_weighted_die(probs);
		switch (param.interpMode[probNum]) {
		case AREA:
			interpMode = cv::INTER_AREA;
			break;
		case CUBIC:
			interpMode = cv::INTER_CUBIC;
			break;
		case LINEAR:
			interpMode = cv::INTER_LINEAR;
			break;
		case NEAREST:
			interpMode = cv::INTER_NEAREST;
			break;
		case LANCZOS4:
			interpMode = cv::INTER_LANCZOS4;
			break;
		default:
			SASSERT(false, "Unknown interp mode.");
			break;
		}
	}

	cv::Scalar padVal = cv::Scalar(0, 0, 0);
	const int imgChannels = in_img.channels();
	if (param.padValue.size() > 0) {
		SASSERT(param.padValue.size() == 1 ||
				param.padValue.size() == imgChannels,
				"Specify either 1 pad value or as many as channels: %d", imgChannels);
		vector<float> padValues;
		for (int i = 0; i < param.padValue.size(); i++) {
			padValues.push_back(param.padValue[i]);
		}
		if (imgChannels > 1 && param.padValue.size() == 1) {
			// Replicate the pad value for simplicity
			for (int c = 1; c < imgChannels; c++) {
				padValues.push_back(padValues[0]);
			}
		}
		padVal = cv::Scalar(padValues[0], padValues[1], padValues[2]);
	}

	switch (param.resizeMode) {
	case WARP:
		cv::resize(in_img, out_img, cv::Size(newWidth, newHeight), 0, 0, interpMode);
		break;
	case FIT_LARGE_SIZE_AND_PAD:
		out_img = AspectKeepingResizeAndPad(in_img, newWidth, newHeight, padMode, padVal,
				interpMode);
		break;
	case FIT_SMALL_SIZE:
		out_img = AspectKeepingResizeBySmall(in_img, newWidth, newHeight, interpMode);
		break;
	default:
		SASSERT(false, "Unknown resize mode.");
		break;
	}
	return out_img;
}



cv::Mat AspectKeepingResizeAndPad(const cv::Mat& in_img,
                                  const int newWidth, const int newHeight,
                                  const int padType,  const cv::Scalar padVal,
                                  const int interpMode) {
	cv::Mat img_resized;
	float origAspect = static_cast<float>(in_img.cols) / in_img.rows;
	float newAspect = static_cast<float>(newWidth) / newHeight;

	if (origAspect > newAspect) {
		int height = floor(static_cast<float>(newWidth) / origAspect);
		cv::resize(in_img, img_resized, cv::Size(newWidth, height), 0, 0, interpMode);
		cv::Size resSize = img_resized.size();
		int padding = floor((newHeight - resSize.height) / 2.0);
		cv::copyMakeBorder(img_resized, img_resized, padding,
				newHeight - resSize.height - padding, 0, 0,
				padType, padVal);
	} else {
		int width = floor(origAspect * newHeight);
		cv::resize(in_img, img_resized, cv::Size(width, newHeight), 0, 0, interpMode);
		cv::Size resSize = img_resized.size();
		int padding = floor((newWidth - resSize.width) / 2.0);
		cv::copyMakeBorder(img_resized, img_resized, 0, 0, padding,
				newWidth - resSize.width - padding,
				padType, padVal);
	}
	return img_resized;
}

cv::Mat AspectKeepingResizeBySmall(const cv::Mat& in_img,
                                   const int newWidth,
                                   const int newHeight,
                                   const int interpMode) {
	cv::Mat img_resized;
	float origAspect = static_cast<float>(in_img.cols) / in_img.rows;
	float newAspect = static_cast<float> (newWidth) / newHeight;

	if (origAspect < newAspect) {
		int height = floor(static_cast<float>(newWidth) / origAspect);
		cv::resize(in_img, img_resized, cv::Size(newWidth, height), 0, 0, interpMode);
	} else {
		int width = floor(origAspect * newHeight);
		cv::resize(in_img, img_resized, cv::Size(width, newHeight), 0, 0, interpMode);
	}
	return img_resized;
}



















void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img, const float brightness_prob,
		const float brightness_delta) {
	float prob;
	soooa_rng_uniform(1, 0.f, 1.f, &prob);
	if (prob < brightness_prob) {
		SASSERT(brightness_delta >= 0, "brightness_delta must be non-negative.");
		float delta;
		soooa_rng_uniform(1, -brightness_delta, brightness_delta, &delta);
		AdjustBrightness(in_img, delta, out_img);
	} else {
		*out_img = in_img;
	}
}

void AdjustBrightness(const cv::Mat& in_img, const float delta, cv::Mat* out_img) {
	//cout << "apply brightness " << delta << endl;
	if (fabs(delta) > 0) {
		in_img.convertTo(*out_img, -1, 1, delta);
	} else {
		*out_img = in_img;
	}
}

void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img, const float contrast_prob,
		const float lower, const float upper) {
	float prob;
	soooa_rng_uniform(1, 0.f, 1.f, &prob);
	if (prob < contrast_prob) {
		SASSERT(upper >= lower, "contrast upper must be >= lower.");
		SASSERT(lower >= 0, "contrast lower must be non-negative.");
		float delta;
		soooa_rng_uniform(1, lower, upper, &delta);
		AdjustContrast(in_img, delta, out_img);
	} else {
		*out_img = in_img;
	}
}

void AdjustContrast(const cv::Mat& in_img, const float delta, cv::Mat* out_img) {
	//cout << "apply contrast " << delta << endl;

	if (fabs(delta - 1.f) > 1e-3) {
		in_img.convertTo(*out_img, -1, delta, 0);
	} else {
		*out_img = in_img;
	}
}


void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img, const float saturation_prob,
		const float lower, const float upper) {
	float prob;
	soooa_rng_uniform(1, 0.f, 1.f, &prob);
	if (prob < saturation_prob) {
		SASSERT(upper >= lower, "saturation upper must be >= lower.");
		SASSERT(lower >= 0, "saturation lower must be non-negative.");
		float delta;
		soooa_rng_uniform(1, lower, upper, &delta);
		AdjustSaturation(in_img, delta, out_img);
	} else {
		*out_img = in_img;
	}
}

void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img) {
	//cout << "apply hue " << delta << endl;

	if (fabs(delta) > 0) {
		// Convert to HSV colorspae.
		cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

		// Split the image to 3 channels.
		vector<cv::Mat> channels;
		cv::split(*out_img, channels);

		// Adjust the hue.
		channels[0].convertTo(channels[0], -1, 1, delta);
		cv::merge(channels, *out_img);

		// Back to BGR colorspace.
		cvtColor(*out_img, *out_img, CV_HSV2BGR);
	} else {
		*out_img = in_img;
	}
}

void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
		const float random_order_prob) {
	float prob;
	soooa_rng_uniform(1, 0.f, 1.f, &prob);
	if (prob < random_order_prob) {
		//cout << "apply randomOrderChannels " << endl;

		// Split the image to 3 channels.
		vector<cv::Mat> channels;
		cv::split(*out_img, channels);
		SASSERT0(channels.size() == 3);

		// Shuffle the channels.
		std::random_shuffle(channels.begin(), channels.end());
		cv::merge(channels, *out_img);
	} else {
		*out_img = in_img;
	}
}

void AdjustSaturation(const cv::Mat& in_img, const float delta, cv::Mat* out_img) {
	//cout << "apply saturation " << delta << endl;

	if (fabs(delta - 1.f) != 1e-3) {
		// Convert to HSV colorspae.
		cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

		// Split the image to 3 channels.
		vector<cv::Mat> channels;
		cv::split(*out_img, channels);

		// Adjust the saturation.
		channels[1].convertTo(channels[1], -1, delta, 0);
		cv::merge(channels, *out_img);

		// Back to BGR colorspace.
		cvtColor(*out_img, *out_img, CV_HSV2BGR);
	} else {
		*out_img = in_img;
	}
}

void RandomHue(const cv::Mat& in_img, cv::Mat* out_img, const float hue_prob,
		const float hue_delta) {
	float prob;
	soooa_rng_uniform(1, 0.f, 1.f, &prob);
	if (prob < hue_prob) {
		SASSERT(hue_delta >= 0, "hue_delta must be non-negative.");
		float delta;
		soooa_rng_uniform(1, -hue_delta, hue_delta, &delta);
		AdjustHue(in_img, delta, out_img);
	} else {
		*out_img = in_img;
	}
}


cv::Mat ApplyDistort(const cv::Mat& in_img, const DistortionParam& param) {
	cv::Mat out_img = in_img;
	float prob;
	soooa_rng_uniform(1, 0.f, 1.f, &prob);

	if (prob > 0.5f) {
		// Do random brightness distortion.
		RandomBrightness(out_img, &out_img, param.brightnessProb, param.brightnessDelta);

		// Do random contrast distortion.
		RandomContrast(out_img, &out_img, param.contrastProb, param.contrastLower,
				param.contrastUpper);

		// Do random saturation distortion.
		RandomSaturation(out_img, &out_img, param.saturationProb, param.saturationLower,
				param.saturationUpper);

		// Do random hue distortion.
		RandomHue(out_img, &out_img, param.hueProb, param.hueDelta);

		// Do random reordering of the channels.
		RandomOrderChannels(out_img, &out_img, param.randomOrderProb);
	} else {
		// Do random brightness distortion.
		RandomBrightness(out_img, &out_img, param.brightnessProb, param.brightnessDelta);

		// Do random saturation distortion.
		RandomSaturation(out_img, &out_img, param.saturationProb, param.saturationLower,
				param.saturationUpper);

		// Do random hue distortion.
		RandomHue(out_img, &out_img, param.hueProb, param.hueDelta);

		// Do random contrast distortion.
		RandomContrast(out_img, &out_img, param.contrastProb, param.contrastLower,
				param.contrastUpper);

		// Do random reordering of the channels.
		RandomOrderChannels(out_img, &out_img, param.randomOrderProb);
	}

	return out_img;
}

