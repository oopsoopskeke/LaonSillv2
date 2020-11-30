/*
 * LayerPropParam.h
 *
 *  Created on: Sep 1, 2017
 *      Author: jkim
 */

#ifndef LAYERPROPPARAM_H_
#define LAYERPROPPARAM_H_


#include "EnumDef.h"
#include "frcnn_common.h"


class ResizeParam {
public:
	ResizeParam() {
		this->prob = -1.0f;			// has resize param 결정
		this->resizeMode = ResizeMode::WARP;
		this->height = 0;
		this->width = 0;
		this->heightScale = 0;
		this->widthScale = 0;
		this->padMode = PadMode::CONSTANT;
		this->interpMode0 = InterpMode::INTERP_NONE;
		this->interpMode1 = InterpMode::INTERP_NONE;
		this->interpMode2 = InterpMode::INTERP_NONE;
		this->interpMode3 = InterpMode::INTERP_NONE;
		this->interpMode4 = InterpMode::INTERP_NONE;
	}
	ResizeParam(const float prob, const ResizeMode resizeMode,
			const uint32_t height, const uint32_t width,
			const uint32_t heightScale, const uint32_t widthScale,
			const PadMode padMode, const std::vector<float>& padValue,
			const InterpMode interpMode0, const InterpMode interpMode1,
			const InterpMode interpMode2, const InterpMode interpMode3,
			const InterpMode interpMode4) {
		this->prob = prob;
		this->height = height;
		this->width = width;
		this->heightScale = heightScale;
		this->widthScale = widthScale;
		this->padMode = padMode;
		this->interpMode0 = interpMode0;
		this->interpMode1 = interpMode1;
		this->interpMode2 = interpMode2;
		this->interpMode3 = interpMode3;
		this->interpMode4 = interpMode4;

		updateInterpMode();
	}

	void updateInterpMode() {
		this->interpMode.clear();
		if (interpMode0 != InterpMode::INTERP_NONE) this->interpMode.push_back(interpMode0);
		if (interpMode1 != InterpMode::INTERP_NONE) this->interpMode.push_back(interpMode1);
		if (interpMode2 != InterpMode::INTERP_NONE) this->interpMode.push_back(interpMode2);
		if (interpMode3 != InterpMode::INTERP_NONE) this->interpMode.push_back(interpMode3);
		if (interpMode4 != InterpMode::INTERP_NONE) this->interpMode.push_back(interpMode4);
	}

	void print() {
		std::cout << "::: ResizeParam :::" 			<< std::endl;
		std::cout << "prob="		<< prob			<< std::endl;
		std::cout << "resizeMode="	<< resizeMode 	<< std::endl;
		std::cout << "height="		<< height 		<< std::endl;
		std::cout << "width="		<< width 		<< std::endl;
		std::cout << "heightScale="	<< heightScale	<< std::endl;
		std::cout << "widthScale="	<< widthScale 	<< std::endl;
		std::cout << "padMode="		<< padMode 		<< std::endl;
		printArray("padValue", padValue);
		printArray("interpMode", interpMode);
	}
public:
	float prob;
	ResizeMode resizeMode;
	uint32_t height;
	uint32_t width;
	uint32_t heightScale;
	uint32_t widthScale;
	PadMode padMode;
	std::vector<float> padValue;
	InterpMode interpMode0;
	InterpMode interpMode1;
	InterpMode interpMode2;
	InterpMode interpMode3;
	InterpMode interpMode4;
	std::vector<InterpMode> interpMode;

};

class EmitConstraint {
public:
	EmitConstraint() {
		this->emitType = EmitType::CENTER;
		this->emitOverlap = 0.0f;
	}
	EmitConstraint(const EmitType emitType, const float emitOverlap) {
		this->emitType = emitType;
		this->emitOverlap = emitOverlap;
	}
	void print() {
		std::cout << "::: EmitConstraint :::" 		<< std::endl;
		std::cout << "emitType="	<< emitType		<< std::endl;
		std::cout << "emitOverlap="	<< emitOverlap	<< std::endl;
	}
public:
	EmitType emitType;
	float emitOverlap;
};

class DistortionParam {
public:
	DistortionParam() {
		this->brightnessProb = 0.f;
		this->brightnessDelta = 0.f;
		this->contrastProb = 0.f;
		this->contrastLower = 0.f;
		this->contrastUpper = 0.f;
		this->hueProb = 0.f;
		this->hueDelta = 0.f;
		this->saturationProb = 0.f;
		this->saturationLower = 0.f;
		this->saturationUpper = 0.f;
		this->randomOrderProb = 0.f;
	}
	DistortionParam(const float brightnessProb,
		const float brightnessDelta,
		const float contrastProb,
		const float contrastLower,
		const float contrastUpper,
		const float hueProb,
		const float hueDelta,
		const float saturationProb,
		const float saturationLower,
		const float saturationUpper,
		const float randomOrderProb) {

		this->brightnessProb = brightnessProb;
		this->brightnessDelta = brightnessDelta;
		this->contrastProb = contrastProb;
		this->contrastLower = contrastLower;
		this->contrastUpper = contrastUpper;
		this->hueProb = hueProb;
		this->hueDelta = hueDelta;
		this->saturationProb = saturationProb;
		this->saturationLower = saturationLower;
		this->saturationUpper = saturationUpper;
		this->randomOrderProb = randomOrderProb;
	}

	void print() {
		std::cout << "::: DistortionParam :::" 				<< std::endl;
		std::cout << "brightnessProb="	<< brightnessProb	<< std::endl;
		std::cout << "brightnessDelta="	<< brightnessDelta	<< std::endl;
		std::cout << "contrastProb=" 	<< contrastProb		<< std::endl;
		std::cout << "contrastLower=" 	<< contrastLower	<< std::endl;
		std::cout << "contrastUpper=" 	<< contrastUpper	<< std::endl;
		std::cout << "hueProb=" 		<< hueProb			<< std::endl;
		std::cout << "hueDelta="		<< hueDelta 		<< std::endl;
		std::cout << "saturationProb="	<< saturationProb	<< std::endl;
		std::cout << "saturationLower="	<< saturationLower	<< std::endl;
		std::cout << "saturationUpper="	<< saturationUpper	<< std::endl;
		std::cout << "randomOrderProb="	<< randomOrderProb	<< std::endl;
	}

public:
	float brightnessProb;
	float brightnessDelta;
	float contrastProb;
	float contrastLower;
	float contrastUpper;
	float hueProb;
	float hueDelta;
	float saturationProb;
	float saturationLower;
	float saturationUpper;
	float randomOrderProb;
};


class ExpansionParam {
public:
	ExpansionParam() {
		this->prob = 0.f;
		this->maxExpandRatio = 1.f;
	}
	ExpansionParam(const float prob, const float maxExpandRatio) {
		this->prob = prob;
		this->maxExpandRatio = maxExpandRatio;
	}
	void print() {
		std::cout << "::: ExpansionParam :::" 				<< std::endl;
		std::cout << "prob=" 			<< prob 			<< std::endl;
		std::cout << "maxExpandRatio="	<< maxExpandRatio 	<< std::endl;
	}
public:
	float prob;
	float maxExpandRatio;
};


class SaltPepperParam {
public:
	SaltPepperParam() {
		this->fraction = 0.f;
	}
	SaltPepperParam(const float fraction, const std::vector<float>& value) {
		this->fraction = fraction;
		this->value = value;
	}
public:
	// Percentage of pixels
	float fraction;
	std::vector<float> value;
};

class NoiseParam {
public:
	NoiseParam() {
		this->prob = 0.f;
		this->histEq = false;
		this->inverse = false;
		this->decolorize = false;
		this->gaussBlur = false;
		this->jpeg = -1;
		this->posterize = false;
		this->erode = false;
		//this->saltpepper = false;
		this->clahe = false;
		this->convertToHSV = false;
		this->convertToLAB = false;
	}
	NoiseParam(const float prob,
			const bool histEq,
			const bool inverse,
			const bool decolorize,
			const bool gaussBlur,
			const float jpeg,
			const bool posterize,
			const bool erode,
			//const bool saltpepper,
			//const SaltPepperParam& saltpepperParam,
			const bool clahe,
			const bool convertToHSV,
			const bool convertToLAB) {
		this->prob = prob;
		this->histEq = histEq;
		this->inverse = inverse;
		this->decolorize = decolorize;
		this->gaussBlur = gaussBlur;
		this->jpeg = jpeg;
		this->posterize = posterize;
		this->erode = erode;
		//this->saltpepper = saltpepper;
		//this->saltpepperParam = saltpepperParam;
		this->clahe = clahe;
		this->convertToHSV = convertToHSV;
		this->convertToLAB = convertToLAB;
	}

	void print() {
		std::cout << "::: NoiseParam :::" 				<< std::endl;
		std::cout << "prob=" 			<< prob 		<< std::endl;
		std::cout << "histEq=" 			<< histEq 		<< std::endl;
		std::cout << "inverse=" 		<< inverse 		<< std::endl;
		std::cout << "decolorize=" 		<< decolorize 	<< std::endl;
		std::cout << "gaussBlur=" 		<< gaussBlur 	<< std::endl;
		std::cout << "jpeg=" 			<< jpeg 		<< std::endl;
		std::cout << "posterize=" 		<< posterize 	<< std::endl;
		std::cout << "erode=" 			<< erode 		<< std::endl;
		std::cout << "clahe=" 			<< clahe 		<< std::endl;
		std::cout << "convertToHSV="	<< convertToHSV << std::endl;
		std::cout << "convertToLAB="	<< convertToLAB	<< std::endl;
	}


public:
	float prob;
	bool histEq;
	bool inverse;
	bool decolorize;
	bool gaussBlur;
	float jpeg;
	bool posterize;
	bool erode;
	//bool saltpepper;
	//SaltPepperParam saltpepperParam;
	bool clahe;
	bool convertToHSV;
	bool convertToLAB;
};



class DataTransformParam {
public:
	DataTransformParam() {
		this->scale = 1.0f;
		this->mirror = false;
		this->cropSize = 0;
		this->cropH = 0;
		this->cropW = 0;
		this->mean.clear();
		this->forceColor = false;
		this->forceGray = false;
	}
	DataTransformParam(const float scale, const bool mirror, const std::vector<float> mean,
			const int cropSize, const ResizeParam resizeParam, const DistortionParam distortParam,
			const ExpansionParam expandParam, const NoiseParam noiseParam,
			const EmitConstraint emitConstraint) {
		this->scale = scale;
		this->mirror = mirror;
		this->mean = mean;
		this->cropSize = cropSize;
		this->resizeParam = resizeParam;
		this->distortParam = distortParam;
		this->expandParam = expandParam;
		this->noiseParam = noiseParam;
		this->emitConstraint = emitConstraint;
	}

	bool hasResizeParam() {
		return this->resizeParam.prob > 0.f;
	}
	bool hasDistortParam() {
		return (this->distortParam.brightnessProb > 0.f ||
				this->distortParam.contrastProb > 0.f ||
				this->distortParam.hueProb > 0.f ||
				this->distortParam.saturationProb > 0.f ||
				this->distortParam.randomOrderProb > 0.f);
	}
	bool hasExpandParam() {
		return (this->expandParam.prob > 0.f);
	}
	bool hasNoiseParam() {
		return (this->noiseParam.prob > 0.f);
	}
	bool hasEmitConstraint() {
		return !(this->emitConstraint.emitType == EmitType::EMIT_NONE);
	}

	void print() {
		std::cout << "::: DataTransformParam :::" 	<< std::endl;
		std::cout << "scale="		<< scale		<< std::endl;
		std::cout << "mirror="		<< mirror		<< std::endl;
		printArray("mean", mean);
		std::cout << "cropSize="	<< cropSize		<< std::endl;
		std::cout << "cropH="		<< cropH		<< std::endl;
		std::cout << "cropW="		<< cropW		<< std::endl;
		std::cout << "forceColor="	<< forceColor	<< std::endl;
		std::cout << "forceGray="	<< forceGray	<< std::endl;
		resizeParam.print();
		distortParam.print();
		expandParam.print();
		noiseParam.print();
		emitConstraint.print();
	}
public:
	float scale;
	bool mirror;
	std::vector<float> mean;
	int cropSize;
	int cropH;
	int cropW;
	bool forceColor;
	bool forceGray;
	ResizeParam resizeParam;
	DistortionParam distortParam;
	ExpansionParam expandParam;
	NoiseParam noiseParam;
	EmitConstraint emitConstraint;
};





class NonMaximumSuppressionParam {
public:
	NonMaximumSuppressionParam() {
		this->nmsThreshold = 0.3f;
		this->topK = 0;
		this->eta = 1.0f;
	}
	NonMaximumSuppressionParam(const float nmsThreshold, const int topK, const float eta) {
		this->nmsThreshold = nmsThreshold;
		this->topK = topK;
		this->eta = eta;
	}

public:
	float nmsThreshold;
	int topK;
	float eta;
};





class SaveOutputParam {
public:
	SaveOutputParam() {}
	SaveOutputParam(const std::string& outputDirectory, const std::string& outputNamePrefix,
			const std::string& outputFormat, const std::string& labelMapFile,
			const std::string& nameSizeFile, const int numTestImage) {
			//const ResizeParam& resizeParam) {
		this->outputDirectory = outputDirectory;
		this->outputNamePrefix = outputNamePrefix;
		this->outputFormat = outputFormat;
		this->labelMapFile = labelMapFile;
		this->nameSizeFile = nameSizeFile;
		this->numTestImage = numTestImage;
		//this->resizeParam = resizeParam;
	}

public:
	std::string outputDirectory;
	std::string outputNamePrefix;
	std::string outputFormat;
	std::string labelMapFile;
	std::string nameSizeFile;
	int numTestImage;
	ResizeParam resizeParam;
};


class Sampler {
public:
	/*
	Sampler(float& minScale) : minScale(minScale) {

	}
	*/
	Sampler() {
		this->minScale = 1.f;
		this->maxScale = 1.f;
		this->minAspectRatio = 1.f;
		this->maxAspectRatio = 1.f;
	}
	Sampler(const float minScale, const float maxScale, const float minAspectRatio,
			const float maxAspectRatio) {
		this->minScale = minScale;
		this->maxScale = maxScale;
		this->minAspectRatio = minAspectRatio;
		this->maxAspectRatio = maxAspectRatio;
	}
	void print() {
		std::cout << "::: Sampler :::"						<< std::endl;
		std::cout << "minScale="		<< minScale			<< std::endl;
		std::cout << "maxScale=" 		<< maxScale			<< std::endl;
		std::cout << "minAspectRatio="	<< minAspectRatio	<< std::endl;
		std::cout << "maxAspectRatio="	<< maxAspectRatio	<< std::endl;
	}
public:
	float minScale;
	float maxScale;
	float minAspectRatio;
	float maxAspectRatio;
};

class SampleConstraint {
public:
	SampleConstraint() {
		this->minJaccardOverlap = 0.f;
		this->maxJaccardOverlap = 0.f;
		this->minSampleCoverage = 0.f;
		this->maxSampleCoverage = 0.f;
		this->minObjectCoverage = 0.f;
		this->maxObjectCoverage = 0.f;
	}
	SampleConstraint(const float minJaccardOverlap, const float maxJaccardOverlap,
			const float minSampleCoverage, const float maxSampleCoverage,
			const float minObjectCoverage, const float maxObjectCoverage) {
		this->minJaccardOverlap = minJaccardOverlap;
		this->maxJaccardOverlap = maxJaccardOverlap;
		this->minSampleCoverage = minSampleCoverage;
		this->maxSampleCoverage = maxSampleCoverage;
		this->minObjectCoverage = minObjectCoverage;
		this->maxObjectCoverage = maxObjectCoverage;
	}
	bool hasMinJaccardOverlap() const {
		return (this->minJaccardOverlap > 0.f);
	}
	bool hasMaxJaccardOverlap() const {
		return (this->maxJaccardOverlap > 0.f);
	}
	bool hasMinSampleCoverage() const {
		return (this->minSampleCoverage > 0.f);
	}
	bool hasMaxSampleCoverage() const {
		return (this->maxSampleCoverage > 0.f);
	}
	bool hasMinObjectCoverage() const {
		return (this->minObjectCoverage > 0.f);
	}
	bool hasMaxObjectCoverage() const {
		return (this->maxObjectCoverage > 0.f);
	}
	void print() {
		std::cout << "::: SampleConstraint :::" 					<< std::endl;
		std::cout << "minJaccardOverlap="	<< minJaccardOverlap	<< std::endl;
		std::cout << "maxJaccardOverlap="	<< maxJaccardOverlap	<< std::endl;
		std::cout << "minSampleCoverage="	<< minSampleCoverage	<< std::endl;
		std::cout << "maxSampleCoverage="	<< maxSampleCoverage	<< std::endl;
		std::cout << "minObjectCoverage="	<< minObjectCoverage	<< std::endl;
		std::cout << "maxObjectCoverage="	<< maxObjectCoverage	<< std::endl;
	}
public:
	float minJaccardOverlap;
	float maxJaccardOverlap;
	float minSampleCoverage;
	float maxSampleCoverage;
	float minObjectCoverage;
	float maxObjectCoverage;
};

class BatchSampler {
public:
	BatchSampler()
	: BatchSampler(true, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0, 100) {}

	BatchSampler(const bool useOriginalImage, const float minScale, const float maxScale,
			const float minAspectRatio, const float maxAspectRatio,
			const float minJaccardOverlap, const float maxJaccardOverlap,
			const float minSampleCoverage, const float maxSampleCoverage,
			const float minObjectCoverage, const float maxObjectCoverage,
			const uint32_t maxSample, const uint32_t maxTrials)
	: minScale(sampler.minScale),
	  maxScale(sampler.maxScale),
	  minAspectRatio(sampler.minAspectRatio),
	  maxAspectRatio(sampler.maxAspectRatio),
	  minJaccardOverlap(sampleConstraint.minJaccardOverlap),
	  maxJaccardOverlap(sampleConstraint.maxJaccardOverlap),
	  minSampleCoverage(sampleConstraint.minSampleCoverage),
	  maxSampleCoverage(sampleConstraint.maxSampleCoverage),
	  minObjectCoverage(sampleConstraint.minObjectCoverage),
	  maxObjectCoverage(sampleConstraint.maxObjectCoverage) {

		this->useOriginalImage = useOriginalImage;

		this->minScale = minScale;
		this->maxScale = maxScale;
		this->minAspectRatio = minAspectRatio;
		this->maxAspectRatio = maxAspectRatio;

		this->minJaccardOverlap = minJaccardOverlap;
		this->maxJaccardOverlap = maxJaccardOverlap;
		this->minSampleCoverage = minSampleCoverage;
		this->maxSampleCoverage = maxSampleCoverage;
		this->minObjectCoverage = minObjectCoverage;
		this->maxObjectCoverage = maxObjectCoverage;

		this->maxSample = maxSample;
		this->maxTrials = maxTrials;
	}

	BatchSampler& operator=(const BatchSampler& other) {
		this->useOriginalImage = other.useOriginalImage;
		this->minScale = other.minScale;
		this->maxScale = other.maxScale;
		this->minAspectRatio = other.minAspectRatio;
		this->maxAspectRatio = other.maxAspectRatio;
		this->minJaccardOverlap = other.minJaccardOverlap;
		this->maxJaccardOverlap = other.maxJaccardOverlap;
		this->minSampleCoverage = other.minSampleCoverage;
		this->maxSampleCoverage = other.maxSampleCoverage;
		this->minObjectCoverage = other.minObjectCoverage;
		this->maxObjectCoverage = other.maxObjectCoverage;
		this->maxSample = other.maxSample;
		this->maxTrials = other.maxTrials;
		return *this;
	}

	bool hasMaxSample() const {
		return (this->maxSample > 0);
	}

	void print() {
		std::cout << "::: BatchSampler :::" 					<< std::endl;
		std::cout << "useOriginalImage="	<< useOriginalImage	<< std::endl;
		sampler.print();
		sampleConstraint.print();
		std::cout << "maxSample="			<< maxSample		<< std::endl;
		std::cout << "maxTrials=" 			<< maxTrials		<< std::endl;
	}
public:
	bool useOriginalImage;

	Sampler sampler;
	float& minScale;
	float& maxScale;
	float& minAspectRatio;
	float& maxAspectRatio;

	SampleConstraint sampleConstraint;
	float& minJaccardOverlap;
	float& maxJaccardOverlap;
	float& minSampleCoverage;
	float& maxSampleCoverage;
	float& minObjectCoverage;
	float& maxObjectCoverage;

	uint32_t maxSample;
	uint32_t maxTrials;
};

class AugmentationParam {
public:
	/*
	
	*/
	AugmentationParam() { }

	AugmentationParam(const float probability) { 
		this->probability = probability;
	}

	void print() {
		std::cout << "::: AugmentationParam :::"						<< std::endl;
		std::cout << "probability="	<< probability	<< std::endl;
	}

public:
	// 기능 별 안쓰는 멤버가 있음. 공용체로 관리??
	float probability;

	std::vector<float> x;
	std::vector<float> y;

	std::vector<float> angle; // rotation angle degree 범위. 

	std::string noiseType; // gaussian, snp(salt and pepper), poisson(보류), speckle(보류)
	float std;	// gaussian stddev. input data의 range에 따라 결정
	float mean;
	// float s_vs_p;	// salt 대 pepper 비율
	float amount;	// image 대비 salt pepper 비율
	std::vector<float> min_max;	// salt와 pepper의 값
	
	std::string filterType; // normal, gaussian, median, bilateral, sharpen
	std::vector<int> kernelSize;
	std::vector<float> sharpen; // sharpen 강도. 0.0 ~
};

#endif /* LAYERPROPPARAM_H_ */
