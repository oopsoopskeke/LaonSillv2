/*
 * ImagePackDataSet.cpp
 *
 *  Created on: 2016. 7. 13.
 *      Author: jhkim
 */

#include <stddef.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "UByteImage.h"
#include "Util.h"
#include "ImagePackDataSet.h"
#include "SysLog.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
ImagePackDataSet<Dtype>::ImagePackDataSet(
		string trainImage,
		string trainLabel,
		uint32_t numTrainFile,
		string testImage,
		string testLabel,
		uint32_t numTestFile)
	: trainImage(trainImage),
	  trainLabel(trainLabel),
	  numTrainFile(numTrainFile),
	  testImage(testImage),
	  testLabel(testLabel),
	  numTestFile(numTestFile) {

	this->trainFileIndex = 0;
	this->testFileIndex = 0;
	this->bufDataSet = NULL;
	this->bufLabelSet = NULL;

	this->frontTrainDataSet = NULL;
	this->frontTrainLabelSet = NULL;
	this->backTrainDataSet = NULL;
	this->backTrainLabelSet = NULL;

	this->secondTrainDataSet = NULL;
	this->secondTrainLabelSet = NULL;

	this->threadArg.context = this;
	this->loading = false;

	this->trainFileIndices = NULL;
	SNEW(this->trainFileIndices, vector<uint32_t>, numTrainFile);
	SASSUME0(this->trainFileIndices != NULL);
	iota(this->trainFileIndices->begin(), this->trainFileIndices->end(), 0);
}

template <typename Dtype>
ImagePackDataSet<Dtype>::~ImagePackDataSet() {
	if(bufDataSet) SDELETE(bufDataSet);
	if(bufLabelSet) SDELETE(bufLabelSet);
	if(frontTrainDataSet) SDELETE(frontTrainDataSet);
	if(frontTrainLabelSet) SDELETE(frontTrainLabelSet);
	if(backTrainDataSet) SDELETE(backTrainDataSet);
	if(backTrainLabelSet) SDELETE(backTrainLabelSet);
	if(secondTrainDataSet) SDELETE(secondTrainDataSet);
	if(secondTrainLabelSet) SDELETE(secondTrainLabelSet);
	if(trainFileIndices) SDELETE(trainFileIndices);
}


template <typename Dtype>
const Dtype* ImagePackDataSet<Dtype>::getTrainDataAt(int index) {
	if(index >= this->numTrainData || index < 0) {
		cout << "invalid index for train data: numTrainData->" << this->numTrainData
            << ", index: " << index << endl;
		exit(1);
	}
	int reqPage = index / numImagesInTrainFile;
	//if(reqPage != trainFileIndex) {
	//	load(DataSet<Dtype>::Train, reqPage);
	//	trainFileIndex = reqPage;
	//}

	// 현재 페이지에 대한 요청,
	// 다음 페이지에 대한 요청 이외의 요청은 무효한 것으로 간주

	const int nextPage = (trainFileIndex+1)%numTrainFile;
	if(reqPage != trainFileIndex && reqPage != nextPage) {
		cout << "DataSet, only sequencial access allowed ... " << endl;
		exit(1);
	}

	// 다음 페이지를 요청하는 경우,
	// 이미로 로드된 다음페이지를 현재 페이지로 스왑하고
	// 그 다음의 페이지를 back page에 로드한다.
	if(numTrainFile > 1 && reqPage == nextPage) {
		while(loading);

		swap();
		//load(DataSet<Dtype>::Train, reqPage+1);

		threadArg.page = (reqPage+1)%numTrainFile;
		pthread_create(&bufThread, NULL, ImagePackDataSet<Dtype>::load_helper, &threadArg);

		trainFileIndex = reqPage;
	}

	const Dtype* ptr = &(*this->frontTrainDataSet)[
        this->dataSize*(*this->trainSetIndices)[(index-reqPage*numImagesInTrainFile)]];

	return ptr;
}

template <typename Dtype>
const Dtype* ImagePackDataSet<Dtype>::getTrainLabelAt(int index) {
	if(index >= this->numTrainData || index < 0) {
		cout << "invalid index for train label: numTrainData->" << this->numTrainData
            << ", index: " << index << endl;
		exit(1);
	}
	int reqPage = index / numImagesInTrainFile;
	//if(reqPage != trainFileIndex) {
	//	load(DataSet<Dtype>::Train, reqPage);
	//	trainFileIndex = reqPage;
	//}

	const int nextPage = (trainFileIndex+1)%numTrainFile;
	if(reqPage != trainFileIndex && reqPage != nextPage) {
		cout << "DataSet, only sequencial access allowed ... " << endl;
		exit(1);
	}
	if(numTrainFile > 1 && reqPage == nextPage) {
		while(loading);

		swap();

		threadArg.page = (reqPage+1)%numTrainFile;
		pthread_create(&bufThread, NULL, ImagePackDataSet<Dtype>::load_helper, &threadArg);

		trainFileIndex = reqPage;
	}
	return &(*this->frontTrainLabelSet)[
        (*this->trainSetIndices)[index-reqPage*numImagesInTrainFile]];
}

template <typename Dtype>
const Dtype* ImagePackDataSet<Dtype>::getValidationDataAt(int index) {
	if(index >= this->numValidationData || index < 0) {
		cout << "invalid index for validation data: numValidationData->"
            << this->numValidationData << ", index: " << index << endl;
		exit(1);
	}
	return &(*this->validationDataSet)[this->dataSize*index];
}

template <typename Dtype>
const Dtype* ImagePackDataSet<Dtype>::getValidationLabelAt(int index) {
	if(index >= this->numValidationData || index < 0) {
		cout << "invalid index for validation label: numValidationData->"
            << this->numValidationData << ", index: " << index << endl;
		exit(1);
	}
	return &(*this->validationLabelSet)[index];
}

template <typename Dtype>
const Dtype* ImagePackDataSet<Dtype>::getTestDataAt(int index) {
	if(index >= this->numTestData || index < 0) {
		cout << "invalid index for test data: numTestData->" << this->numTestData 
            << ", index: " << index << endl;
		exit(1);
	}
	int reqPage = index / numImagesInTestFile;
	if(reqPage != testFileIndex) {
		load(DataSet<Dtype>::Test, reqPage);
		testFileIndex = reqPage;
	}
	return &(*this->testDataSet)[
        this->dataSize*(*this->testSetIndices)[(index-reqPage*numImagesInTestFile)]];
}

template <typename Dtype>
const Dtype* ImagePackDataSet<Dtype>::getTestLabelAt(int index) {
	if(index >= this->numTestData || index < 0) {
		cout << "invalid index for test label: numTestData->" << this->numTestData
            << ", index: " << index << endl;
		exit(1);
	}
	int reqPage = index / numImagesInTestFile;
	if(reqPage != testFileIndex) {
		load(DataSet<Dtype>::Test, reqPage);
		testFileIndex = reqPage;
	}
	return &(*this->testLabelSet)[(*this->testSetIndices)[index-reqPage*numImagesInTestFile]];
}

template <typename Dtype>
void ImagePackDataSet<Dtype>::load() {

#ifndef GPU_MODE
	numTrainData = loadDataSetFromResource(filenames[0], trainDataSet, 0, 10000);
	numTestData = loadDataSetFromResource(filenames[1], testDataSet, 0, 0);
#else
	int numTrainDataInFile = load(DataSet<Dtype>::Train, 0);

	// train에 대해 load()할 경우 항상 back에 대해서 수행,
	// 최초의 0page에 대한 load()의 경우 primary, front에 대한 것, 이를 조정해 준다.
	this->trainDataSet = backTrainDataSet;
	frontTrainDataSet = this->trainDataSet;
	this->trainLabelSet = backTrainLabelSet;
	frontTrainLabelSet = this->trainLabelSet;
	backTrainDataSet = NULL;
	backTrainLabelSet = NULL;


	int numTestDataInFile = load(DataSet<Dtype>::Test, 0);
	if(numTrainDataInFile <= 0 || numTestDataInFile <= 0) {
		cout << "could not load resources ... " << endl;
		exit(1);
	}

	this->numTrainData = numTrainDataInFile*numTrainFile;
	this->numTestData = numTestDataInFile*numTestFile;
	this->numImagesInTrainFile = numTrainDataInFile;
	this->numImagesInTestFile = numTestDataInFile;

	// back에 1페이지 (두번째 페이지)를 로드
	// 별도의 thread에서 수행하는 것이 바람직하나 최초에 second에 대한 설정이 필요해서
	// 편의에 따라 main thread에서 load 수행함.
	if(numTrainFile > 1) {
		load(DataSet<Dtype>::Train, 1);

		this->secondTrainDataSet = backTrainDataSet;
		this->secondTrainLabelSet = backTrainLabelSet;
	}

#endif
}

template <typename Dtype>
void* ImagePackDataSet<Dtype>::load_helper(void* arg) {
	thread_arg_t* arg_ = (thread_arg_t*)arg;
	cout << "load train dataset page " << arg_->page << endl;

	((ImagePackDataSet<Dtype>*)(arg_->context))->load(DataSet<Dtype>::Train, arg_->page);
}

template <typename Dtype>
int ImagePackDataSet<Dtype>::load(typename DataSet<Dtype>::Type type, int page) {
	this->loading = true;

	int shuffledPage = (*this->trainFileIndices)[page];
	cout << "load train dataset shuffled page " << shuffledPage << endl;
	string pageSuffix = to_string(shuffledPage);
	int numData = 0;
	// train
	switch(type) {
	case DataSet<Dtype>::Train: {
		/*
		numData = loadDataSetFromResource(
				trainImage+pageSuffix,
				trainLabel+pageSuffix,
				this->trainDataSet,
				this->trainLabelSet,
				this->trainSetIndices);
				*/
		numData = loadDataSetFromResource(
				trainImage+pageSuffix,
				trainLabel+pageSuffix,
				this->backTrainDataSet,
				this->backTrainLabelSet,
				this->trainSetIndices);
		numImagesInTrainFile = numData;
		//zeroMean(true, true);

		break;
	}
	case DataSet<Dtype>::Test: {
		numData = loadDataSetFromResource(
				testImage+pageSuffix,
				testLabel+pageSuffix,
				this->testDataSet,
				this->testLabelSet,
				this->testSetIndices);
		numImagesInTestFile = numData;
		//zeroMean(true, false);
		break;
	}
	}

	loading = false;
	return numData;
}

#ifndef GPU_MODE
template <typename Dtype>
int ImagePackDataSet<Dtype>::loadDataSetFromResource(string resources[2],
    DataSample* &dataSet, int offset, int size) {
	// LOAD IMAGE DATA
	ImageInfo dataInfo(resources[0]);
	dataInfo.load();

	// READ IMAGE DATA META DATA
	unsigned char* dataPtr = dataInfo.getBufferPtrAt(0);
	int dataMagicNumber = Util::pack4BytesToInt(dataPtr);
	if(dataMagicNumber != 0x00000803) return -1;
	dataPtr += 4;
	int dataSize = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumRows = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumCols = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;

	// LOAD LABEL DATA
	ImageInfo targetInfo(resources[1]);
	targetInfo.load();

	// READ LABEL DATA META DATA
	unsigned char* targetPtr = targetInfo.getBufferPtrAt(0);
	int targetMagicNumber = Util::pack4BytesToInt(targetPtr);
	if(targetMagicNumber != 0x00000801) return -1;
	targetPtr += 4;
	//int labelSize = Util::pack4BytesToInt(targetPtr);
	targetPtr += 4;

	if(offset >= dataSize) return 0;

	dataPtr += offset;
	targetPtr += offset;

	int stop = dataSize;
	if(size > 0) stop = min(dataSize, offset+size);

	//int dataArea = dataNumRows*  dataNumCols;
	if(dataSet) SDELETE(dataSet);
	dataSet = new DataSample[stop-offset];

	for(int i = offset; i < stop; i++) {
		//const DataSample* dataSample = new DataSample(dataPtr, dataArea, targetPtr, 10);
		//dataSet.push_back(dataSample);
		//dataSet[i-offset].readData(dataPtr, dataNumRows, dataNumCols, 1, targetPtr, 10);
		dataSet[i-offset].readData(dataPtr, dataNumRows, dataNumCols, 1, targetPtr, 10);
	}
	return stop-offset;
}

#else
template <typename Dtype>
int ImagePackDataSet<Dtype>::loadDataSetFromResource(
		string data_path,
		string label_path,
		vector<Dtype>*& dataSet,
		vector<Dtype>*& labelSet,
		vector<uint32_t>*& setIndices) {

	FILE* imfp = fopen(data_path.c_str(), "rb");
	if(!imfp) {
		cout << "ERROR: Cannot open image dataset " << data_path << endl;
		return 0;
	}
	FILE* lbfp = fopen(label_path.c_str(), "rb");
	if(!lbfp) {
		fclose(imfp);
		cout << "ERROR: Cannot open label dataset " << label_path << endl;
		return 0;
	}

	UByteImageDataset image_header;
	UByteLabelDataset label_header;

	// Read and verify file headers
	if(fread(&image_header, sizeof(UByteImageDataset), 1, imfp) != 1) {
		cout << "ERROR: Invalid dataset file (image file header)" << endl;
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}
	if(fread(&label_header, sizeof(UByteLabelDataset), 1, lbfp) != 1) {
		cout << "ERROR: Invalid dataset file (label file header)" << endl;
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}

	// Byte-swap data structure values (change endianness)
	image_header.Swap();
	label_header.Swap();

	// Verify datasets
	if(image_header.magic != UBYTE_IMAGE_MAGIC) {
		printf("ERROR: Invalid dataset file (image file magic number)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}
	if (label_header.magic != UBYTE_LABEL_MAGIC) 	{
		printf("ERROR: Invalid dataset file (label file magic number)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}
	if (image_header.length != label_header.length) {
		printf("ERROR: Dataset file mismatch "
                "(number of images do not match the number of labels)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}

	// Output dimensions
	size_t width = image_header.width;
	size_t height = image_header.height;
	size_t channel = image_header.channel;
	this->cols = width;
	this->rows = height;
	this->channels = channel;
	this->dataSize = this->rows*this->cols*this->channels;

	// Read images and labels (if requested)
	size_t dataSetSize = ((size_t)image_header.length)*this->dataSize;
	if(!dataSet) {
		dataSet = NULL;
		SNEW(dataSet, vector<Dtype>, dataSetSize);
		SASSUME0(dataSet != NULL);
	}
	if(!bufDataSet) {
		this->bufDataSet = NULL;
		SNEW(this->bufDataSet, vector<uint8_t>, dataSetSize);
		SASSUME0(this->bufDataSet != NULL);
	}
	if(!labelSet) {
		labelSet = NULL;
		SNEW(labelSet, vector<Dtype>, label_header.length);
		SASSUME0(labelSet != NULL);
	}
	if(!this->bufLabelSet) {
		this->bufLabelSet = NULL;
		SNEW(this->bufLabelSet, vector<uint32_t>, label_header.length);
		SASSUME0(this->bufLabelSet != NULL);
	}

	if(!setIndices) {
		setIndices = NULL;
		SNEW(setIndices, vector<uint32_t>, label_header.length);
		SASSUME0(setIndices != NULL);
		iota(setIndices->begin(), setIndices->end(), 0);
	}

	if(fread(&(*bufDataSet)[0], sizeof(uint8_t), dataSetSize, imfp) != dataSetSize) {
		printf("ERROR: Invalid dataset file (partial image dataset)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}

	for (size_t i = 0; i < dataSetSize; i++) {
		//(*dataSet)[i] = (*this->bufDataSet)[i] / this->scale;
		(*dataSet)[i] = (*this->bufDataSet)[i];
	}

	if (fread(&(*this->bufLabelSet)[0], sizeof(uint32_t), label_header.length, lbfp)
        != label_header.length) {
		printf("ERROR: Invalid dataset file (partial label dataset)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}

	for (size_t i = 0; i < label_header.length; i++) {
		(*labelSet)[i] = Dtype((*bufLabelSet)[i]);
	}

	fclose(imfp);
	fclose(lbfp);

	return image_header.length;
}
#endif

template <typename Dtype>
void ImagePackDataSet<Dtype>::swap() {
	if(frontTrainDataSet == NULL ||
			frontTrainLabelSet == NULL ||
			backTrainDataSet == NULL ||
			backTrainLabelSet == NULL) {
		cout << "swap error, one of buffer is NULL ... " << endl;
		exit(1);
	}

	vector<Dtype>* tempDataSet = frontTrainDataSet;
	vector<Dtype>* tempLabelSet = frontTrainLabelSet;

	frontTrainDataSet = backTrainDataSet;
	frontTrainLabelSet = backTrainLabelSet;

	backTrainDataSet = tempDataSet;
	backTrainLabelSet = tempLabelSet;
}

/*
template <typename Dtype>
void ImagePackDataSet<Dtype>::zeroMean(bool hasMean, bool isTrain) {
	uint32_t di, ci, hi, wi;
	double sum[3] = {0.0, 0.0, 0.0};

	const uint32_t perImageSize = this->cols*this->rows*this->channels;
	const uint32_t perChannelSize = this->cols*this->rows;

	if(!hasMean) {
		for(di = 0; di < this->numTrainData; di++) {
			for(ci = 0; ci < this->channels; ci++) {
				for(hi = 0; hi < this->rows; hi++) {
					for(wi = 0; wi < this->cols; wi++) {
						sum[ci] += (*backTrainDataSet)[
                            wi+hi*this->cols+ci*perChannelSize+di*perImageSize];
					}
				}
			}
		}
		for(ci = 0; ci < this->channels; ci++) {
			this->mean[ci] = (Dtype)(sum[ci] / (perChannelSize*this->numTrainData));
		}
	}

	if(isTrain) {
		for(di = 0; di < this->numImagesInTrainFile; di++) {
			for(ci = 0; ci < this->channels; ci++) {
				for(hi = 0; hi < this->rows; hi++) {
					for(wi = 0; wi < this->cols; wi++) {
                        int index = wi+hi*this->cols+ci*perChannelSize+di*perImageSize;
						(*backTrainDataSet)[index] -= this->mean[ci];
					}
				}
			}
		}
	} else {
		for(di = 0; di < this->numImagesInTestFile; di++) {
			for(ci = 0; ci < this->channels; ci++) {
				for(hi = 0; hi < this->rows; hi++) {
					for(wi = 0; wi < this->cols; wi++) {
                        int index = wi+hi*this->cols+ci*perChannelSize+di*perImageSize;
						(*this->testDataSet)[index] -= this->mean[ci];
					}
				}
			}
		}
	}
}
*/

template <typename Dtype>
void ImagePackDataSet<Dtype>::shuffleTrainDataSet() {
	random_shuffle(&(*this->trainSetIndices)[0],
                    &(*this->trainSetIndices)[numImagesInTrainFile]);
	random_shuffle(&(*this->trainFileIndices)[0],
                    &(*this->trainFileIndices)[numTrainFile]);
}

template <typename Dtype>
void ImagePackDataSet<Dtype>::shuffleValidationDataSet() {
	cout << "shuffleValidationDataSet() is not supported yet ... " << endl;
	exit(1);
}

template <typename Dtype>
void ImagePackDataSet<Dtype>::shuffleTestDataSet() {
	random_shuffle(&(*this->testSetIndices)[0],
                   &(*this->testSetIndices)[numImagesInTestFile]);
}

template class ImagePackDataSet<float>;
