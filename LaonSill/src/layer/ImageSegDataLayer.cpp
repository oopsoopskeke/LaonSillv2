/*
 * ImageSegDataLayer.cpp
 *
 *  Created on: Aug 3, 2017
 *      Author: jkim
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ImageSegDataLayer.h"
#include "PropMgmt.h"
#include "EnumDef.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "IO.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
ImageSegDataLayer<Dtype>::ImageSegDataLayer()
: InputLayer<Dtype>(),
  imgDataTransformer(&SLPROP(ImageSegData, dataTransformParam)) {
	this->type = Layer<Dtype>::ImageSegData;

	LabelType labelType = SLPROP(ImageSegData, labelType);
	SASSERT(labelType == LabelType::PIXEL, "only LabelType::PIXEL is supported.");

	const string imageSource = SLPROP(ImageSegData, imageSource);
	SASSERT(!imageSource.empty(), "must provide imageSource.");
	const string listSource = SLPROP(ImageSegData, listSource);
	SASSERT(!listSource.empty(), "must provide listSource.");

	STDOUT_LOG("Opening file %s.", listSource.c_str());
	ifstream infile(listSource.c_str());

	string linestr;
	while (std::getline(infile, linestr)) {
		istringstream iss(linestr);
		string imgfn;
		string segfn;
		iss >> imgfn >> segfn;
		this->lines_.push_back(std::make_pair(imgfn, segfn));
	}

	if (SLPROP(ImageSegData, shuffle)) {
		STDOUT_LOG("Shuffling data");
		shuffleTrainDataSet();
	}
	STDOUT_LOG("A total of %d images.", this->lines_.size());

	this->lines_id_ = 0;
}

template <typename Dtype>
ImageSegDataLayer<Dtype>::~ImageSegDataLayer() {}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}

	Layer<Dtype>::_adjustInputShape();

	const string imageSource = SLPROP(ImageSegData, imageSource);
	const int inputSize = this->_inputData.size();
	for (int i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		// Read an iage, and use it to initialize the output data
		cv::Mat cv_img = ReadImageToCVMat(imageSource + this->lines_[lines_id_].first, 0, 0);
		SASSERT(cv_img.data, "Could not load %s", this->lines_[lines_id_].first.c_str());

		const int channels = cv_img.channels();
		const int height = cv_img.rows;
		const int width = cv_img.cols;

		int cropWidth = SLPROP(ImageSegData, cropSize);
		int cropHeight = SLPROP(ImageSegData, cropSize);

		const int batchSize = SNPROP(batchSize);
		if (cropWidth > 0 && cropHeight > 0) {
			// data
			if (i == 0) {
				this->_outputData[0]->reshape({(uint32_t)batchSize, (uint32_t)channels,
					(uint32_t)cropHeight, (uint32_t)cropWidth});
			}
			// label
			else if (i == 1) {
				this->_outputData[1]->reshape({(uint32_t)batchSize, 1, (uint32_t)cropHeight,
					(uint32_t)cropWidth});
			}
		} else {
			// data
			if (i == 0) {
				this->_outputData[0]->reshape({(uint32_t)batchSize, (uint32_t)channels,
					(uint32_t)height, (uint32_t)width});
			}
			// label
			else if (i == 1) {
				this->_outputData[1]->reshape({(uint32_t)batchSize, 1, (uint32_t)height,
					(uint32_t)width});
			}
		}

		if (i == 2) {
			this->_outputData[2]->reshape({(uint32_t)batchSize, 1, 1, 2});
		}

		this->_inputShape[i] = this->_outputData[i]->getShape();
		if (i == 0) {
			STDOUT_LOG("Output data size: %d, %d, %d, %d", this->_outputData[0]->batches(),
					this->_outputData[0]->channels(), this->_outputData[0]->height(),
					this->_outputData[0]->width());
		} else if (i == 1) {
			STDOUT_LOG("Output label size: %d, %d, %d, %d", this->_outputData[1]->batches(),
					this->_outputData[1]->channels(), this->_outputData[1]->height(),
					this->_outputData[1]->width());
		} else if (i == 2) {
			STDOUT_LOG("Output data_dim size: %d, %d, %d, %d", this->_outputData[2]->batches(),
					this->_outputData[2]->channels(), this->_outputData[2]->height(),
					this->_outputData[2]->width());
		}
	}
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::feedforward() {
	reshape();
	load_batch();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
	load_batch();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::load_batch() {
	const int maxHeight = this->_outputData[0]->height();
	const int maxWidth = this->_outputData[0]->width();

	const int batchSize = SNPROP(batchSize);
	const LabelType labelType = SLPROP(ImageSegData, labelType);
	const int ignoreLabel = SLPROP(ImageSegData, ignoreLabel);
	const string imageSource = SLPROP(ImageSegData, imageSource);

	const int linesSize = this->lines_.size();
	int offset;
	for (int itemId = 0; itemId < batchSize; itemId++) {
		SASSERT0(linesSize > this->lines_id_);

		Dtype* outputData = this->_outputData[0]->mutable_host_data();
		Dtype* outputLabel = this->_outputData[1]->mutable_host_data();
		Dtype* outputDataDim = this->_outputData[2]->mutable_host_data();

		cv::Mat cvImg;
		cv::Mat cvSeg;
		int imgRow;
		int imgCol;
		cvImg = ReadImageToCVMat(imageSource + this->lines_[this->lines_id_].first,
				0, 0, 0, 0, true, &imgRow, &imgCol);
		SASSERT(cvImg.data, "Fail to load img: %s%s", imageSource.c_str(),
				this->lines_[this->lines_id_].first.c_str());
		if (labelType == LabelType::PIXEL) {
			cvSeg = ReadImageToCVMat(imageSource + this->lines_[this->lines_id_].second,
					0, 0, 0, 0, false);
			SASSERT(cvSeg.data, "Fail to load seg: %s%s", imageSource.c_str(),
					this->lines_[this->lines_id_].second.c_str());
		}

		// 1. set data_dim ///////////////////////////////////////////////////////////////////
		offset = this->_outputData[2]->offset(itemId);
		outputDataDim += offset;
		outputDataDim[0] = static_cast<Dtype>(std::min(maxHeight, imgRow));
		outputDataDim[1] = static_cast<Dtype>(std::min(maxWidth, imgCol));
		//////////////////////////////////////////////////////////////////////////////////////

		// 2. set data ///////////////////////////////////////////////////////////////////////
		this->imgDataTransformer.transform(cvImg, this->_outputData[0], itemId);
		//////////////////////////////////////////////////////////////////////////////////////

		// 3. set seg ////////////////////////////////////////////////////////////////////////
		this->segDataTransformer.transform(cvSeg, this->_outputData[1], itemId);
		//////////////////////////////////////////////////////////////////////////////////////

		// go to the next std::vector<int>::iterator iter;
		this->lines_id_++;
		if (this->lines_id_ >= this->lines_.size()) {
			// We have reached the end. Restart from the first.
			STDOUT_LOG("Restarting data load from start.");
			this->lines_id_ = 0;
			if (SLPROP(ImageSegData, shuffle)) {
				shuffleTrainDataSet();
			}
		}
	}
}

template <typename Dtype>
int ImageSegDataLayer<Dtype>::getNumTrainData() {
	return this->lines_.size();
}

template <typename Dtype>
int ImageSegDataLayer<Dtype>::getNumTestData() {

}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::shuffleTrainDataSet() {
	std::random_shuffle(this->lines_.begin(), this->lines_.end());
}



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* ImageSegDataLayer<Dtype>::initLayer() {
	ImageSegDataLayer* layer = NULL;
	SNEW(layer, ImageSegDataLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ImageSegDataLayer<Dtype>::destroyLayer(void* instancePtr) {
    ImageSegDataLayer<Dtype>* layer = (ImageSegDataLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
 void ImageSegDataLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	SASSERT0(!isInput);
	SASSERT0(index < 3);

    ImageSegDataLayer<Dtype>* layer = (ImageSegDataLayer<Dtype>*)instancePtr;
    if (!isInput) {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ImageSegDataLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ImageSegDataLayer<Dtype>* layer = (ImageSegDataLayer<Dtype>*)instancePtr;
    layer->reshape();

    if (SNPROP(miniBatch) == 0) {
		int trainDataNum = layer->getNumTrainData();
		if (trainDataNum % SNPROP(batchSize) == 0) {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize);
		} else {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize) + 1;
		}
		WorkContext::curPlanInfo->miniBatchCount = SNPROP(miniBatch);
	}

    return true;
}

template<typename Dtype>
void ImageSegDataLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	ImageSegDataLayer<Dtype>* layer = (ImageSegDataLayer<Dtype>*)instancePtr;
	layer->feedforward();

}

template<typename Dtype>
void ImageSegDataLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void ImageSegDataLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool ImageSegDataLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    const int channels = 0;
    const int height = 0;
    const int width = 0;

    const int cropWidth = SLPROP(ImageSegData, cropSize);
    const int cropHeight = SLPROP(ImageSegData, cropSize);

	const int batchSize = SNPROP(batchSize);
    if (cropWidth > 0 && cropHeight > 0) {
        TensorShape outputShape1;
        outputShape1.N = batchSize;
        outputShape1.C = channels;
        outputShape1.H = cropHeight;
        outputShape1.W = cropWidth;
        outputShape.push_back(outputShape1);
        
        TensorShape outputShape2;
        outputShape2.N = batchSize;
        outputShape2.C = 1;
        outputShape2.H = cropHeight;
        outputShape2.W = cropWidth;
        outputShape.push_back(outputShape2);
    } else {
        SASSERT(false, "not implemented yet.");
        TensorShape outputShape1;
        outputShape1.N = batchSize;
        outputShape1.C = channels;
        outputShape1.H = height;
        outputShape1.W = width;
        outputShape.push_back(outputShape1);
        
        TensorShape outputShape2;
        outputShape2.N = batchSize;
        outputShape2.C = 1;
        outputShape2.H = height;
        outputShape2.W = width;
        outputShape.push_back(outputShape2);
    }

    return true;
}

template<typename Dtype>
uint64_t ImageSegDataLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class ImageSegDataLayer<float>;
