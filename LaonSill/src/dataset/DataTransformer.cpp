/*
 * DataTransformer.cpp
 *
 *  Created on: Jul 19, 2017
 *      Author: jkim
 */

#include "DataTransformer.h"
#include "ImTransforms.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "BBoxUtil.h"
#include "IO.h"
#include "MathFunctions.h"
#include "ImageUtil.h"

using namespace std;


template <typename Dtype>
DataTransformer<Dtype>::DataTransformer(DataTransformParam* param) {
	if (param == NULL) {
		this->param = DataTransformParam();
	} else {
		this->param = *param;
	}
	this->hasMean = false;
	this->hasCropSize = false;
	this->hasScale = false;
	this->hasMirror = false;

	if (this->param.mean.size() > 0) {
		this->hasMean = true;

		// fill mean value to 3 regardless of number of image channels.
		if (this->param.mean.size() == 1) {
			for (int i = 1; i < 3; i++) {
				this->param.mean.push_back(this->param.mean[0]);
			}
		}
	}

	if (this->param.cropSize != 0.0) {
		this->hasCropSize = true;
	}

	if (this->param.scale != 1.0) {
		this->hasScale = true;
	}

	if (this->param.mirror != false) {
		this->hasMirror = true;
	}

	srand((uint32_t)time(NULL));
}



template <typename Dtype>
DataTransformer<Dtype>::~DataTransformer() {

}

template <typename Dtype>
void DataTransformer<Dtype>::transformWithMeanScale(Datum* datum, const vector<float>& mean,
		const float scale, Dtype* dataPtr) {

	const bool hasMean = (mean.size() > 0);


	string decode;
	if (datum->encoded) {

	}


	const string& data = datum->data;



	const int datum_channels = datum->channels;
	const int datum_height = datum->height;
	const int datum_width = datum->width;
	int height = datum_height;
	int width = datum_width;
	int h_off = 0;
	int w_off = 0;

	Dtype datum_element;
	int top_index, data_index;
	for (int c = 0; c < datum_channels; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
				top_index = (c * height + h) * width + w;
				datum_element =
						static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));

				if (hasMean) {
					dataPtr[top_index] = (datum_element - mean[c]) * scale;
				} else {
					dataPtr[top_index] = datum_element * scale;
				}
			}
		}
	}
}


template <typename Dtype>
void DataTransformer<Dtype>::transform(Datum* datum, Dtype* dataPtr) {
	string decode;
	if (datum->encoded) {

	}

	const string& data = datum->data;
	const int datum_channels = datum->channels;
	const int datum_height = datum->height;
	const int datum_width = datum->width;
	int height = datum_height;
	int width = datum_width;
	int h_off = 0;
	int w_off = 0;

	Dtype datum_element;
	int top_index, data_index;
	for (int c = 0; c < datum_channels; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
				top_index = (c * height + h) * width + w;
				datum_element =
						static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));

				if (this->hasMean) {
					dataPtr[top_index] = (datum_element - this->param.mean[c]) *
							this->param.scale;
				} else {
					dataPtr[top_index] = datum_element * this->param.scale;
				}
			}
		}
	}
}

template <typename Dtype>
void DataTransformer<Dtype>::transform(cv::Mat& im, Data<Dtype>* data, int batchIdx) {
	const int cropSize = this->param.cropSize;
	const int imgChannels = im.channels();
	// height and width may change due to pad or cropping
	const int imgHeight = im.rows;
	const int imgWidth = im.cols;

	// Check dimensions
	const int channels = data->channels();
	const int height = data->height();
	const int width = data->width();
	const int batches = data->batches();

	SASSERT0(channels == imgChannels);
	SASSERT0(height <= imgHeight);
	SASSERT0(width <= imgWidth);
	SASSERT0(batches >= 1);

	SASSERT(im.depth() == CV_8U, "Image data type must be unsigned byte");

	const Dtype scale = Dtype(this->param.scale);
	const bool doMirror = this->hasMirror && rand(2);

	SASSERT0(imgChannels > 0);
	SASSERT0(imgHeight >= cropSize);
	SASSERT0(imgWidth >= cropSize);

	int hOff = 0;
	int wOff = 0;
	cv::Mat croppedIm = im;
	if (this->hasCropSize) {
		SASSERT0(cropSize == height);
		SASSERT0(cropSize == width);
		// We only do random crop when we do training.
		if (SNPROP(status) == NetworkStatus::Train) {
			hOff = rand(imgHeight - cropSize + 1);
			wOff = rand(imgWidth - cropSize + 1);
		} else {
			hOff = (imgHeight - cropSize) / 2;
			wOff = (imgWidth - cropSize) / 2;
		}
		cv::Rect roi(wOff, hOff, cropSize, cropSize);
		croppedIm = im(roi);
	} else {
		SASSERT0(imgHeight == height);
		SASSERT0(imgWidth == width);
	}

	SASSERT0(croppedIm.data);

	Dtype* dataPtr = data->mutable_host_data() + data->offset(batchIdx);
	int topIndex;
	for (int h = 0; h < height; h++) {
		const uchar* ptr = croppedIm.ptr<uchar>(h);
		int imgIndex = 0;
		for (int w = 0; w < width; w++) {
			for (int c = 0; c < imgChannels; c++) {
				if (doMirror) {
					topIndex = (c * height + h) * width + (width - 1 - w);
				} else {
					topIndex = (c * height + h) * width + w;
				}
				// int topIndex = (c * height + h) * width + w;
				Dtype pixel = static_cast<Dtype>(ptr[imgIndex++]);
				if (this->hasMean) {
					dataPtr[topIndex] = (pixel - this->param.mean[c]) * scale;
				} else {
					dataPtr[topIndex] = pixel * scale;
				}
			}
		}
	}
}


template <typename Dtype>
void DataTransformer<Dtype>::transform(AnnotatedDatum* annoDatum, Data<Dtype>* data,
		int batchIdx, std::vector<AnnotationGroup>& transformedAnnoVec) {
	bool doMirror;
	transform(annoDatum, data, batchIdx, transformedAnnoVec, &doMirror);
}

template <typename Dtype>
void DataTransformer<Dtype>::transform(AnnotatedDatum* annoDatum, Data<Dtype>* data,
		int batchIdx, std::vector<AnnotationGroup>& transformedAnnoVec, bool* doMirror) {
	vector<AnnotationGroup> transformedAnnoGroupAll;
	transform(annoDatum, data, batchIdx, &transformedAnnoGroupAll, doMirror);
	for (int g = 0; g < transformedAnnoGroupAll.size(); g++) {
		transformedAnnoVec.push_back(transformedAnnoGroupAll[g]);
	}
}

template <typename Dtype>
void DataTransformer<Dtype>::transform(AnnotatedDatum* annoDatum, Data<Dtype>* data,
		int batchIdx, std::vector<AnnotationGroup>* transformedAnnoGroupAll, bool* doMirror) {
	// Transform datum.
	NormalizedBBox cropBBox;
	transform(annoDatum, data, batchIdx, &cropBBox, doMirror);

	// Transform annotation.
	const bool doResize = true;
	transformAnnotation(annoDatum, doResize, cropBBox, *doMirror, *transformedAnnoGroupAll);
}

template <typename Dtype>
void DataTransformer<Dtype>::transform(const Datum* datum, Data<Dtype>* data, int batchIdx,
		NormalizedBBox* cropBBox, bool* doMirror) {
	// If datum is encoded, decoded and transform the cv::image.
	if (datum->encoded) {
		SASSERT(!(this->param.forceColor || this->param.forceGray),
				"cannot set both forceColor and forceGray");
		bool isColor = (datum->channels == 1) ? false : true;
		cv::Mat cv_img = DecodeDatumToCVMat(datum, isColor, true);
		return transform(cv_img, data, batchIdx, cropBBox, doMirror);
	} else {
		if (this->param.forceColor || this->param.forceGray) {
			SASSERT(false, "Only support encoded datum now");
		}
		SASSERT(!this->param.hasResizeParam(),
				"Only Encoded Datum is supported for Resize Function.");
	}

	const int cropSize = this->param.cropSize;
	const int datumChannels = datum->channels;
	const int datumHeight = datum->height;
	const int datumWidth = datum->width;

	// Check dimensions.
	const int channels = data->channels();
	const int height = data->height();
	const int width = data->width();
	const int batches = data->batches();

	SASSERT0(channels == datumChannels);
	SASSERT0(height <= datumHeight);
	SASSERT0(width <= datumWidth);
	SASSERT0(batches >= 1);

	if (cropSize) {
		SASSERT0(cropSize == height);
		SASSERT0(cropSize == width);
	} else {
		SASSERT0(datumHeight == height);
		SASSERT0(datumWidth == width);
	}

	Dtype* transformedData = data->mutable_host_data() + data->offset(batchIdx);
	transform(datum, transformedData, cropBBox, doMirror);
}

template <typename Dtype>
void DataTransformer<Dtype>::transform(const cv::Mat& cv_img, Data<Dtype>* data, int batchIdx,
			NormalizedBBox* cropBBox, bool* doMirror) {
	// Check dimensions.
	const int imgChannels = cv_img.channels();
	const int channels = data->channels();
	const int height = data->height();
	const int width = data->width();
	const int batches = data->batches();

	SASSERT0(imgChannels > 0);
	SASSERT(cv_img.depth() == CV_8U, "Image data type must be unsigned byte.");
	SASSERT0(channels == imgChannels);
	SASSERT0(batches >= 1);

	const int cropSize = this->param.cropSize;
	const Dtype scale = this->param.scale;
	*doMirror = this->param.mirror && rand(2);
	//if (*doMirror) {
	//	cout << "apply mirror" << endl;
	//}
	const bool hasMeanValues = this->param.mean.size() > 0;

	int croph = this->param.cropH;
	int cropw = this->param.cropW;
	if (cropSize) {
		croph = cropSize;
		cropw = cropSize;
	}

	cv::Mat cv_resized_image, cv_noised_image, cv_cropped_image;
	if (this->param.hasResizeParam()) {
		cv_resized_image = ApplyResize(cv_img, this->param.resizeParam);
#if 0
		ImageUtil<Dtype>::dispCVMat(cv_resized_image, "cvResizedImage");
#endif
	} else {
		cv_resized_image = cv_img;
	}
	if (this->param.hasNoiseParam()) {
		SASSERT(false, "Noise param not upported.");
	} else {
		cv_noised_image = cv_resized_image;
	}
	int imgHeight = cv_noised_image.rows;
	int imgWidth = cv_noised_image.cols;
	SASSERT0(imgHeight >= croph);
	SASSERT0(imgWidth >= cropw);

	int hoff = 0;
	int woff = 0;
	if ((croph > 0) && (cropw > 0)) {
		SASSERT0(croph == height);
		SASSERT0(cropw == width);
		// We only do random crop when we do training.
		if (SNPROP(status) == NetworkStatus::Train) {
			hoff = rand(imgHeight - croph + 1);
			woff = rand(imgWidth - cropw + 1);
		} else {
			hoff = (imgHeight - croph) / 2;
			woff = (imgWidth - cropw) / 2;
		}
		cv::Rect roi(woff, hoff, cropw, croph);
		cv_cropped_image = cv_noised_image(roi);
	} else {
		cv_cropped_image = cv_noised_image;
	}

	// Return the normalized crop bbox.
	cropBBox->xmin = Dtype(woff) / imgWidth;
	cropBBox->ymin = Dtype(hoff) / imgHeight;
	cropBBox->xmax = Dtype(woff + width) / imgWidth;
	cropBBox->ymax = Dtype(hoff + height) / imgHeight;

	SASSERT0(cv_cropped_image.data);

	Dtype* transformedData = data->mutable_host_data() + data->offset(batchIdx);
	int topIndex;
	for (int h = 0; h < height; h++) {
		const uchar* ptr = cv_cropped_image.ptr<uchar>(h);
		int imgIndex = 0;
		int hidx = h;
		for (int w = 0; w < width; w++) {
			int widx = w;
			if (*doMirror) {
				widx = (width - 1 - w);
			}
			int hidxreal = hidx;
			int widxreal = widx;
			for (int c = 0; c < imgChannels; c++) {
				topIndex = (c * height + hidxreal) * width + widxreal;
				Dtype pixel = static_cast<Dtype>(ptr[imgIndex++]);
				if (hasMeanValues) {
					transformedData[topIndex] = (pixel - this->param.mean[c]) * scale;
				} else {
					transformedData[topIndex] = pixel * scale;
				}
			}
		}
	}
}

template <typename Dtype>
void DataTransformer<Dtype>::transform(const Datum* datum, Dtype* transformedData,
		NormalizedBBox* cropBBox, bool* doMirror) {
	const string& data = datum->data;
	const int datumChannels = datum->channels;
	const int datumHeight = datum->height;
	const int datumWidth = datum->width;

	const int cropSize = this->param.cropSize;
	const Dtype scale = this->param.scale;
	*doMirror = this->param.mirror && rand(2);
	const bool hasUint8 = data.size() > 0;
	const bool hasMeanValues = this->param.mean.size() > 0;

	SASSERT0(datumChannels > 0);
	SASSERT0(datumHeight >= cropSize);
	SASSERT0(datumWidth >= cropSize);

	int height = datumHeight;
	int width = datumWidth;

	int hoff = 0;
	int woff = 0;
	if (cropSize) {
		height = cropSize;
		width = cropSize;
		// We only do random crop when we do training.
		if (SNPROP(status) == NetworkStatus::Train) {
			hoff = rand(datumHeight - cropSize + 1);
			woff = rand(datumWidth - cropSize + 1);
		} else {
			hoff = (datumHeight - cropSize) / 2;
			woff = (datumWidth - cropSize) / 2;
		}
	}

	// Return the normalized crop bbox.
	cropBBox->xmin = Dtype(woff) / datumWidth;
	cropBBox->ymin = Dtype(hoff) / datumHeight;
	cropBBox->xmax = Dtype(woff + width) / datumWidth;
	cropBBox->ymax = Dtype(hoff + height) / datumHeight;


	Dtype datumElement;
	int topIndex;
	int dataIndex;
	for (int c = 0; c < datumChannels; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				dataIndex = (c * datumHeight + hoff + h) * datumWidth + woff + w;
				if (*doMirror) {
					topIndex = (c * height + h) * width + (width - 1 - w);
				} else {
					topIndex = (c * height + h) * width + w;
				}
				if (hasUint8) {
					datumElement = static_cast<Dtype>(static_cast<uint8_t>(data[dataIndex]));
				} else {
					datumElement = datum->float_data[dataIndex];
				}

				if (hasMeanValues) {
					transformedData[topIndex] = (datumElement - this->param.mean[c]) * scale;
				} else {
					transformedData[topIndex] = datumElement * scale;
				}
			}
		}
	}
}

template <typename Dtype>
void DataTransformer<Dtype>::transform(const Datum* datum, Data<Dtype>* data, int batchIdx) {
	NormalizedBBox cropBBox;
	bool doMirror;
	transform(datum, data, batchIdx, &cropBBox, &doMirror);
}









template <typename Dtype>
int DataTransformer<Dtype>::rand(int n) {
#if 1
	SASSERT0(n > 0);
	//int result = std::rand() %n;
	//cout << "rand():" << result << endl;
	//return result;
	return std::rand() % n;
#else
	return 2 % n;
#endif
}



template <typename Dtype>
vector<uint32_t> DataTransformer<Dtype>::inferDataShape(const int channels, const int height, 
        const int width) {

	const int cropSize = this->param.cropSize;
	int cropH = this->param.cropH;
	int cropW = this->param.cropW;
	if (cropSize) {
		cropH = cropSize;
		cropW = cropSize;
	}

    int _height;
    int _width;
	// Check dimensions.
	SASSERT0(channels > 0);
	if (this->param.hasResizeParam()) {
		InferNewSize(this->param.resizeParam, width, height, &_width, &_height);
	}
	SASSERT0(_height >= cropH);
	SASSERT0(_width >= cropW);

	// Build DataShape
	vector<uint32_t> shape(4);
	shape[0] = 1;
	shape[1] = channels;
	shape[2] = (cropH) ? cropH : _height;
	shape[3] = (cropW) ? cropW : _width;

	return shape;

}



template <typename Dtype>
vector<uint32_t> DataTransformer<Dtype>::inferDataShape(const Datum* datum) {
	if (datum->encoded) {
		SASSERT(!(this->param.forceColor || this->param.forceGray),
				"cannot set both forceColor and forceGray");
		bool isColor = (datum->channels == 1) ? false : true;
		cv::Mat cv_img = DecodeDatumToCVMat(datum, isColor, true);
		return inferDataShape(cv_img);
	}

	const int channels = datum->channels;
	const int height = datum->height;
	const int width = datum->width;
    return inferDataShape(channels, height, width);

    /*
	const int cropSize = this->param.cropSize;
	int cropH = this->param.cropH;
	int cropW = this->param.cropW;
	if (cropSize) {
		cropH = cropSize;
		cropW = cropSize;
	}
	const int datumChannels = datum->channels;
	int datumHeight = datum->height;
	int datumWidth = datum->width;

	// Check dimensions.
	SASSERT0(datumChannels > 0);
	if (this->param.hasResizeParam()) {
		InferNewSize(this->param.resizeParam, datumWidth, datumHeight,
				&datumWidth, &datumHeight);
	}
	SASSERT0(datumHeight >= cropH);
	SASSERT0(datumWidth >= cropW);

	// Build DataShape
	vector<uint32_t> shape(4);
	shape[0] = 1;
	shape[1] = datumChannels;
	shape[2] = (cropH) ? cropH : datumHeight;
	shape[3] = (cropW) ? cropW : datumWidth;

	return shape;
    */
}

template <typename Dtype>
vector<uint32_t> DataTransformer<Dtype>::inferDataShape(const cv::Mat& cv_img) {
	const int channels = cv_img.channels();
	const int height = cv_img.rows;
	const int width = cv_img.cols;

    return inferDataShape(channels, height, width);

    /*
	const int cropSize = this->param.cropSize;
	int cropH = this->param.cropH;
	int cropW = this->param.cropW;
	if (cropSize) {
		cropH = cropSize;
		cropW = cropSize;
	}
	const int imgChannels = cv_img.channels();
	int imgHeight = cv_img.rows;
	int imgWidth = cv_img.cols;
	// Check dimensions.
	SASSERT0(imgChannels > 0);
	if (this->param.hasResizeParam()) {
		InferNewSize(this->param.resizeParam, imgWidth, imgHeight, &imgWidth, &imgHeight);
	}
	SASSERT0(imgHeight >= cropH);
	SASSERT0(imgWidth >= cropW);

	// Build DataShape.
	vector<uint32_t> shape(4);
	shape[0] = 1;
	shape[1] = imgChannels;
	shape[2] = (cropH) ? cropH : imgHeight;
	shape[3] = (cropW) ? cropW : imgWidth;

	return shape;
    */
}




template <typename Dtype>
void DataTransformer<Dtype>::distortImage(const Datum* datum, Datum* distortDatum) {
	if (!this->param.hasDistortParam()) {
		*distortDatum = *datum;
		return;
	}
	// If datum is encoded, decode and crop the cv::image.
	if (datum->encoded) {
		SASSERT(!(this->param.forceColor || this->param.forceGray),
				"cannot set both forceColor and forceGray");
		bool isColor = (datum->channels == 1) ? false : true;
		cv::Mat cv_img = DecodeDatumToCVMat(datum, isColor, true);
		// Distort the image.
		cv::Mat distort_img = ApplyDistort(cv_img, this->param.distortParam);
		// Save the image into datum
		EncodeCVMatToDatum(distort_img, "jpg", distortDatum);
		distortDatum->label = datum->label;
		return;
	} else {
		SASSERT(false, "Only support encoded datum now");
	}
}

template <typename Dtype>
void DataTransformer<Dtype>::expandImage(const cv::Mat& img, const float expandRatio,
		NormalizedBBox* expandBBox, cv::Mat* expand_img) {
	const int imgHeight = img.rows;
	const int imgWidth = img.cols;
	const int imgChannels = img.channels();

	// Get the bbox dimension.
	int height = static_cast<int>(imgHeight * expandRatio);
	int width = static_cast<int>(imgWidth * expandRatio);
	float hOff, wOff;
	soooa_rng_uniform(1, 0.f, static_cast<float>(height - imgHeight), &hOff);
	soooa_rng_uniform(1, 0.f, static_cast<float>(width - imgWidth), &wOff);
	hOff = floor(hOff);
	wOff = floor(wOff);

	// expand된 크기의 이미지를 (0, 0) 기준에서 woff, hoff만큼 shift
	expandBBox->xmin = - wOff / imgWidth;
	expandBBox->ymin = - hOff / imgHeight;
	expandBBox->xmax = (width - wOff) / imgWidth;
	expandBBox->ymax = (height - hOff) / imgHeight;

	expand_img->create(height, width, img.type());
	expand_img->setTo(cv::Scalar(0));
	const bool hasMeanValues = this->param.mean.size() > 0;

	if (hasMeanValues) {
		SASSERT(this->param.mean.size() == 1 || this->param.mean.size() == imgChannels,
				"Specify either 1 mean value or as many as channels: %d", imgChannels);
		vector<cv::Mat> channels(imgChannels);
		cv::split(*expand_img, channels);
		SASSERT0(channels.size() == this->param.mean.size());
		// 각 channel image마다 channel mean값으로 초기화
		for (int c = 0; c < imgChannels; c++) {
			channels[c] = this->param.mean[c];
		}
		cv::merge(channels, *expand_img);
	}
	// expand된 이미지 상의 원래 이미지 영역
	cv::Rect bboxROI(wOff, hOff, imgWidth, imgHeight);
	// expand 이미지의 원래 이미지 영역에 원 이미지 복사
	img.copyTo((*expand_img)(bboxROI));

}

template <typename Dtype>
void DataTransformer<Dtype>::expandImage(const Datum* datum, const float expandRatio,
		NormalizedBBox* expandBBox, Datum* expandDatum) {
	// If datum is encoded, decode and crop the cv::image
	if (datum->encoded) {
		SASSERT(!(this->param.forceColor || this->param.forceGray),
				"cannot set both forceColor and forceGray");
		bool isColor = (datum->channels == 1) ? false : true;
		cv::Mat cv_img = DecodeDatumToCVMat(datum, isColor, true);
		// Expand the image.
		cv::Mat expand_img;
		expandImage(cv_img, expandRatio, expandBBox, &expand_img);
		// Save the iage into datum.
		EncodeCVMatToDatum(expand_img, "jpg", expandDatum);
		expandDatum->label = datum->label;
		return;
	} else {
		if (this->param.forceColor || this->param.forceGray) {
			SASSERT(false, "Only support encoded datum now");
		}
	}

	const int datumChannels = datum->channels;
	const int datumHeight = datum->height;
	const int datumWidth = datum->width;

	// Get the bbox dimension.
	int height = static_cast<int>(datumHeight * expandRatio);
	int width = static_cast<int>(datumWidth * expandRatio);
	float hOff, wOff;
	soooa_rng_uniform(1, 0.f, static_cast<float>(height - datumHeight), &hOff);
	soooa_rng_uniform(1, 0.f, static_cast<float>(width - datumWidth), &wOff);
	hOff = floor(hOff);
	wOff = floor(wOff);
	expandBBox->xmin = - wOff / datumWidth;
	expandBBox->ymin = - hOff / datumHeight;
	expandBBox->xmax = (width - wOff) / datumWidth;
	expandBBox->ymax = (height - hOff) / datumHeight;

	// Crop the image using bbox.
	expandDatum->channels = datumChannels;
	expandDatum->height = height;
	expandDatum->width = width;
	expandDatum->label = datum->label;
	expandDatum->data.clear();
	expandDatum->float_data.clear();
	expandDatum->encoded = false;
	const int expandDatumSize = datumChannels * height * width;
	const string& datumBuffer = datum->data;
	string buffer(expandDatumSize, ' ');
	for (int h = hOff; h < hOff + datumHeight; h++) {
		for (int w = wOff; w < wOff + datumWidth; w++) {
			for (int c = 0; c < datumChannels; c++) {
				int datumIndex = (c * datumHeight + h - hOff) * datumWidth + w - wOff;
				int expandDatumIndex = (c * height + h) * width + w;
				buffer[expandDatumIndex] = datumBuffer[datumIndex];
			}
		}
	}
	expandDatum->data = buffer;
}

template <typename Dtype>
void DataTransformer<Dtype>::expandImage(const AnnotatedDatum* annoDatum,
		AnnotatedDatum* expandedAnnoDatum) {
	if (!this->param.hasExpandParam()) {
		*expandedAnnoDatum = *annoDatum;
		return;
	}
	const ExpansionParam& expandParam = this->param.expandParam;
	const float expandProb = expandParam.prob;
	float prob;
	soooa_rng_uniform(1, 0.f, 1.f, &prob);
	if (prob > expandProb) {
		*expandedAnnoDatum = *annoDatum;
		return;
	}
	const float maxExpandRatio = expandParam.maxExpandRatio;
	if (fabs(maxExpandRatio - 1.f) < 1e-2) {
		*expandedAnnoDatum = *annoDatum;
		return;
	}
	float expandRatio;
	soooa_rng_uniform(1, 1.f, maxExpandRatio, &expandRatio);
	// Expand the datum.
	NormalizedBBox expandBBox;
	expandImage(annoDatum, expandRatio, &expandBBox, expandedAnnoDatum);
	expandedAnnoDatum->type = annoDatum->type;

	// Transform the annotation according to cropBBox.
	const bool doResize = false;
	const bool doMirror = false;
	transformAnnotation(annoDatum, doResize, expandBBox, doMirror,
			expandedAnnoDatum->annotation_groups);

}




template <typename Dtype>
void DataTransformer<Dtype>::transformAnnotation(const AnnotatedDatum* annoDatum,
		const bool doResize, const NormalizedBBox& cropBBox, const bool doMirror,
		std::vector<AnnotationGroup>& transformedAnnoGroupAll) {

	const int imgHeight = annoDatum->height;
	const int imgWidth = annoDatum->width;
	if (annoDatum->type == AnnotationType::BBOX) {
		// Go through each AnnotationGroup.
		for (int g = 0; g < annoDatum->annotation_groups.size(); g++) {
			const AnnotationGroup& annoGroup = annoDatum->annotation_groups[g];
			AnnotationGroup transformedAnnoGroup;
			// Go through each Annotation.
			bool hasValidAnnotation = false;
			for (int a = 0; a < annoGroup.annotations.size(); a++) {
				// Annotation이 frcnn과 ssd에 각각 하나씩...
				// 이 곳에서는 ssd의 Annotation 즉 Annotation_s를 사용
				const Annotation_s& anno = annoGroup.annotations[a];
				const NormalizedBBox& bbox = anno.bbox;
				// Adjust bounding box annotation.
				NormalizedBBox resizeBBox = bbox;
				if (doResize && this->param.hasResizeParam()) {
					SASSERT0(imgHeight > 0);
					SASSERT0(imgWidth > 0);
					UpdateBBoxByResizePolicy(this->param.resizeParam, imgWidth, imgHeight,
							&resizeBBox);
				}
				// expand 또는 crop (batch sampling) 할 때, constraint를 만족하는지 여부
				// EmitConstraint::CENTER의 경우, expand, crop의 결과가 기존 이미지의 center를 포함해야 함.
				if (this->param.hasEmitConstraint() &&
						!MeetEmitConstraint(cropBBox, resizeBBox, this->param.emitConstraint)) {
					continue;
				}
				NormalizedBBox projBBox;
				if (ProjectBBox(cropBBox, resizeBBox, &projBBox)) {
					hasValidAnnotation = true;
					Annotation_s* transformedAnno = transformedAnnoGroup.add_annotation();
					transformedAnno->instance_id = anno.instance_id;
					NormalizedBBox* transformedBBox = &transformedAnno->bbox;
					*transformedBBox = projBBox;
					if (doMirror) {
						Dtype temp = transformedBBox->xmin;
						transformedBBox->xmin = 1 - transformedBBox->xmax;
						transformedBBox->xmax = 1 - temp;
					}
					if (doResize && this->param.hasResizeParam()) {
						ExtrapolateBBox(this->param.resizeParam, imgHeight, imgWidth,
								cropBBox, transformedBBox);
					}
				}
			}
			// Save for output.
			if (hasValidAnnotation) {
				transformedAnnoGroup.group_label = annoGroup.group_label;
				transformedAnnoGroupAll.push_back(transformedAnnoGroup);
			}
		}
	} else {
		SASSERT(false, "Unkown annotation type.");
	}
}


template <typename Dtype>
void DataTransformer<Dtype>::cropImage(const Datum* datum, const NormalizedBBox& bbox,
			Datum* cropDatum) {
	// If datum is encoded, decode and crop the cv::image.
	if (datum->encoded) {
		SASSERT(!(this->param.forceColor || this->param.forceGray),
				"cannot set both forceColor and forceGray");
		bool isColor = (datum->channels == 1) ? false : true;
		cv::Mat cv_img = DecodeDatumToCVMat(datum, isColor, true);
		// Crop the image.
		cv::Mat crop_img;
		cropImage(cv_img, bbox, &crop_img);
		// Save the image into datum.
		EncodeCVMatToDatum(crop_img, "jpg", cropDatum);
		cropDatum->label = datum->label;
		return;
	} else {
		if (this->param.forceColor || this->param.forceGray) {
			SASSERT(false, "Only support encoded datum now");
		}
	}

	const int datumChannels = datum->channels;
	const int datumHeight = datum->height;
	const int datumWidth = datum->width;

	// Get the bbox dimension.
	NormalizedBBox clippedBBox;
	ClipBBox(bbox, &clippedBBox);
	NormalizedBBox scaledBBox;
	ScaleBBox(clippedBBox, datumHeight, datumWidth, &scaledBBox);
	const int woff = static_cast<int>(scaledBBox.xmin);
	const int hoff = static_cast<int>(scaledBBox.ymin);
	const int width = static_cast<int>(scaledBBox.xmax - scaledBBox.xmin);
	const int height = static_cast<int>(scaledBBox.ymax - scaledBBox.ymin);

	// Crop the image using bbox.
	cropDatum->channels = datumChannels;
	cropDatum->height = height;
	cropDatum->width = width;
	cropDatum->label = datum->label;
	cropDatum->data.clear();
	cropDatum->float_data.clear();
	cropDatum->encoded = false;
	const int cropDatumSize = datumChannels * height * width;
	const std::string& datumBuffer = datum->data;
	std::string buffer(cropDatumSize, ' ');
	for (int h = hoff; h < hoff + height; h++) {
		for (int w = woff;  w < woff + width; w++) {
			for (int c = 0; c < datumChannels; c++) {
				int datumIndex = (c * datumHeight + h) * datumWidth + w;
				int cropDatumIndex = (c * height + h - hoff) * width + w - woff;
				buffer[cropDatumIndex] = datumBuffer[datumIndex];
			}
		}
	}
	cropDatum->data = buffer;
}

template <typename Dtype>
void DataTransformer<Dtype>::cropImage(const AnnotatedDatum* annoDatum,
		const NormalizedBBox& bbox, AnnotatedDatum* croppedAnnoDatum) {
	// Crop the datum.
	cropImage((const Datum*)annoDatum, bbox, (Datum*)croppedAnnoDatum);
	croppedAnnoDatum->type = annoDatum->type;

	// Transform the annotation according to cropBBox.
	const bool doResize = false;
	const bool doMirror = false;
	NormalizedBBox cropBBox;
	ClipBBox(bbox, &cropBBox);
	transformAnnotation(annoDatum, doResize, cropBBox, doMirror,
			croppedAnnoDatum->annotation_groups);
}

template <typename Dtype>
void DataTransformer<Dtype>::cropImage(const cv::Mat& img, const NormalizedBBox& bbox,
		cv::Mat* crop_img) {
	const int imgHeight = img.rows;
	const int imgWidth = img.cols;

	// Get the bbox dimension.
	NormalizedBBox clippedBBox;
	ClipBBox(bbox, &clippedBBox);
	NormalizedBBox scaledBBox;
	ScaleBBox(clippedBBox, imgHeight, imgWidth, &scaledBBox);

	// Crop the image using bbox.
	int woff = static_cast<int>(scaledBBox.xmin);
	int hoff = static_cast<int>(scaledBBox.ymin);
	int width = static_cast<int>(scaledBBox.xmax - scaledBBox.xmin);
	int height = static_cast<int>(scaledBBox.ymax - scaledBBox.ymin);
	cv::Rect bboxROI(woff, hoff, width, height);

	img(bboxROI).copyTo(*crop_img);
}


































template <typename Dtype>
void transformInv(const int num, const int singleImageSize,
		const int imageHeight, const int imageWidth,
		const int height, const int width, const vector<Dtype>& pixelMeans,
		const Dtype* dataData, Data<Dtype>& temp, cv::Mat& im) {

	for (int i = 0; i < num; i++) {
		temp.reshape({1, 3, uint32_t(imageHeight), uint32_t(imageWidth)});
		Dtype* tempData = temp.mutable_host_data();

		std::copy(dataData + i * singleImageSize, dataData + (i + 1) * singleImageSize, tempData);

		// transpose
		temp.transpose({0, 2, 3, 1});

		// pixel mean
		for (int j = 0; j < singleImageSize; j += 3) {
			tempData[j + 0] += pixelMeans[0];
			tempData[j + 1] += pixelMeans[1];
			tempData[j + 2] += pixelMeans[2];
		}

		im = cv::Mat(imageHeight, imageWidth, CV_32FC3, tempData);
		cv::resize(im, im, cv::Size(width, height), 0, 0, CV_INTER_LINEAR);

		im.convertTo(im, CV_8UC3);
		//cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
		cv::imshow("result", im);

    	char key = cv::waitKey(0);
    	if (key != 'n') {
			cv::destroyAllWindows();
			exit(1);
		}

		//cv::destroyAllWindows();
	}
}

template void transformInv(const int num, const int singleImageSize,
		const int imageHeight, const int imageWidth,
		const int height, const int width, const vector<float>& pixelMeans,
		const float* dataData, Data<float>& data, cv::Mat& im);






template class DataTransformer<float>;



























