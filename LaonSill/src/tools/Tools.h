/*
 * Tools.h
 *
 *  Created on: Jul 28, 2017
 *      Author: jkim
 */

#ifndef TOOLS_H_
#define TOOLS_H_


#include "Datum.h"
#include "SDF.h"

class BaseConvertParam {
public:
	BaseConvertParam() {
		this->labelMapFilePath = "";
		this->outFilePath = "";
		this->resultCode = 0;
		this->resultMsg = "";
	}
	virtual void validityCheck();
	void print() {
		std::cout << "labelMapFilePath: " << this->labelMapFilePath << std::endl;
		std::cout << "outFilePath: " << this->outFilePath << std::endl;
	}
	std::string labelMapFilePath;
	std::string outFilePath;

	int resultCode;
	std::string resultMsg;
};






class MnistDataSet {
public:
	void validityCheck(int& resultCode, std::string& resultMsg);
	void print() {
		std::cout << "name: " << this->name << std::endl;
		std::cout << "imageFilePath: " << this->imageFilePath << std::endl;
		std::cout << "labelFilePath: " << this->labelFilePath << std::endl;
	}

	bool operator==(const MnistDataSet& other) {
		return (this->name == other.name &&
				this->imageFilePath == other.imageFilePath &&
				this->labelFilePath == other.labelFilePath);
	}
	std::string name;
	std::string imageFilePath;
	std::string labelFilePath;
};

class ConvertMnistDataParam : public BaseConvertParam {
public:
	ConvertMnistDataParam() {}
	ConvertMnistDataParam(MnistDataSet& dataSet) {
		addDataSet(dataSet);
	}
	ConvertMnistDataParam(const std::vector<MnistDataSet>& dataSetList)
	: dataSetList(dataSetList) {}

	virtual void validityCheck();
	void addDataSet(MnistDataSet& dataSet) {
		this->dataSetList.push_back(dataSet);
	}

	void print() {
		BaseConvertParam::print();
		std::cout << "dataSetList size: " << this->dataSetList.size() << std::endl;
		for (int i = 0; i < this->dataSetList.size(); i++) {
			this->dataSetList[i].print();
		}
	}
	std::vector<MnistDataSet> dataSetList;
};


class ImageSet {
public:
	void validityCheck(int& resultCode, std::string& resultMsg);
	void print() {
		std::cout << "name: " << this->name << std::endl;
		std::cout << "dataSetPath: " << this->dataSetPath << std::endl;
	}
	bool operator==(const ImageSet& other) {
		return (this->name == other.name &&
				this->dataSetPath == other.dataSetPath);
	}

	std::string name;
	std::string dataSetPath;
};

class ConvertImageSetParam : public BaseConvertParam {
public:
	ConvertImageSetParam() {
		this->gray = false;
		this->shuffle = false;
		this->multiLabel = false;
		this->channelSeparated = true;
		this->resizeWidth = 0;
		this->resizeHeight = 0;
		this->checkSize = false;
		this->encoded = true;
		this->encodeType = "jpg";
		this->basePath = "";
		//this->datasetPath = "";
		this->numClasses = 0;
	}

	virtual void validityCheck();

	void addImageSet(ImageSet& imageSet) {
		this->imageSetList.push_back(imageSet);
	}

	void print() {
		BaseConvertParam::print();

		std::cout << "gray: " 				<< this->gray << std::endl;
		std::cout << "shuffle: " 			<< this->shuffle << std::endl;
		std::cout << "multiLabel: " 		<< this->multiLabel << std::endl;
		std::cout << "channelSeparated: " 	<< this->channelSeparated << std::endl;
		std::cout << "resizeWidth: " 		<< this->resizeWidth << std::endl;
		std::cout << "resizeHeight: " 		<< this->resizeHeight << std::endl;
		std::cout << "checkSize: " 			<< this->checkSize << std::endl;
		std::cout << "encoded: " 			<< this->encoded << std::endl;
		std::cout << "encodeType: " 		<< this->encodeType << std::endl;
		std::cout << "basePath: " 			<< this->basePath << std::endl;
		//std::cout << "datasetPath: " 		<< this->datasetPath << std::endl;
		std::cout << "numClasses: "			<< this->numClasses << std::endl;
		std::cout << "imageSetList size: "	<< this->imageSetList.size() << std::endl;
		for (int i = 0; i < this->imageSetList.size(); i++) {
			this->imageSetList[i].print();
		}
	}

	bool gray;
	bool shuffle;
	bool multiLabel;
	bool channelSeparated;
	int resizeWidth;
	int resizeHeight;
	bool checkSize;
	bool encoded;
	std::string encodeType;
	std::string basePath;
	//std::string datasetPath;
	std::vector<ImageSet> imageSetList;
	int numClasses;
};

class ConvertAnnoSetParam : public ConvertImageSetParam {
public:
	ConvertAnnoSetParam() {
		this->annoType = "detection";
		this->labelType = "xml";
		//this->labelMapFile = "";
		this->checkLabel = true;
		this->minDim = 0;
		this->maxDim = 0;
	}

	virtual void validityCheck();

	void print() {
		ConvertImageSetParam::print();

		std::cout << "annoType: " 		<< this->annoType << std::endl;
		std::cout << "labelType: "		<< this->labelType << std::endl;
		std::cout << "checkLabel: "		<< this->checkLabel << std::endl;
		std::cout << "minDim: "			<< this->minDim << std::endl;
		std::cout << "maxDim: " 		<< this->maxDim << std::endl;
	}
	std::string annoType;
	std::string labelType;
	bool checkLabel;
	int minDim;
	int maxDim;
};





void parse_label_map(const std::string& labelMapPath, std::vector<LabelItem>& labelItemList);


void denormalizeTest(int argc, char** argv);
void denormalize(const std::string& oldParamPath, const std::string& newParamPath);

void convertMnistDataTest(int argc, char** argv);
void convertMnistDataTemp(ConvertMnistDataParam& param);
void convertMnistData(ConvertMnistDataParam& param);
//void convertMnistData(const std::string& imageFilePath, const std::string& labelFilePath,
//		const std::string& outFilePath);

void convertImageSetTest(int argc, char** argv);
void convertImageSet(ConvertImageSetParam& param);
/*
void convertImageSet(bool gray, bool shuffle, bool multiLabel, bool channelSeparated,
		int resizeWidth, int resizeHeight, bool checkSize, bool encoded,
		const std::string& encodeType, const std::string& imagePath,
		const std::string& datasetPath, const std::string& outPath);
		*/

void convertAnnoSetTest(int argc, char** argv);
void convertAnnoSet(ConvertAnnoSetParam& param);
/*
void convertAnnoSet(bool gray, bool shuffle, bool multiLabel, bool channelSeparated,
		int resizeWidth, int resizeHeight, bool checkSize, bool encoded,
		const std::string& encodeType, const std::string& annoType, const std::string& labelType,
		const std::string& labelMapFile, bool checkLabel, int minDim, int maxDim,
		const std::string& basePath, const std::string& datasetPath,
		const std::string& outPath);*/

void computeImageMeanTest(int argc, char** argv);
void computeImageMean(const std::string& sdfPath);



#endif /* TOOLS_H_ */
