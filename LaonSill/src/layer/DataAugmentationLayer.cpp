/**
 * @file	DataAugmentationLayer.cpp
 * @date	2018/3/29
 * @author	heim
 * @brief
 * @details
 */


#include "DataAugmentationLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "MemoryMgmt.h"

#include <math.h>
#define PI	3.141592653589


// #define CV_SHOW	// 실제 이미지를 표시하여 동작이 되는지 디버깅 용
#define DATAAUGMENTATIONLAYER_LOG 1

using namespace std;

template <typename Dtype>
DataAugmentationLayer<Dtype>::DataAugmentationLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::DataAugmentation;
		
}


template <typename Dtype>
DataAugmentationLayer<Dtype>::~DataAugmentationLayer() {

}


template <typename Dtype>
void DataAugmentationLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* inputData = this->_inputData[0]->host_data();
	Dtype* outputData =  this->_outputData[0]->mutable_host_data();

	const int inputBatch =  this->_inputData[0]->getShape(0);
	const int inputChannel =  this->_inputData[0]->getShape(1);
	const int inputRow = this->_inputData[0]->getShape(2);
	const int inputColumn = this->_inputData[0]->getShape(3);

	this->scale = SLPROP(DataAugmentation, scale);
	this->fliplr = SLPROP(DataAugmentation, fliplr);
	this->flipud = SLPROP(DataAugmentation, flipud);
	this->translate = SLPROP(DataAugmentation, translate);
	this->rotation = SLPROP(DataAugmentation, rotation);
	this->shear = SLPROP(DataAugmentation, shear);
	this->filtering = SLPROP(DataAugmentation, filtering);
	this->noise = SLPROP(DataAugmentation, noise);


	this->interpolation = SLPROP(DataAugmentation, interpolation);
	this->fillMode = SLPROP(DataAugmentation, fillMode);
	this->filled = SLPROP(DataAugmentation, filled);
	
	
	// Image 1장당 fliplr, flipud, scale, translate... 순서로 진행된다. 순서가 중요한 작업이면 별도의 레이어를 생성하여 순서를 정하면 된다.
	// 작성시, 순서에 variant한 작업을 나중에 실행되게 작성할 것. ex) Add noise
	for (size_t i = 0; i < inputBatch; i++) {
		const Dtype* inputDatum = inputData + i * inputChannel  * inputRow * inputColumn;
		Dtype* outputDatum =  outputData + i * inputChannel  * inputRow* inputColumn;
		cv::Mat src;
		Dtype* data;
		if (inputChannel == 3) {
			cv::Mat src_split[3];
			src_split[0] = cv::Mat(inputRow, inputColumn, CV_32FC1);
			Dtype* data_0 = (Dtype*)src_split[0] .data;
			std::memcpy(data_0, inputDatum, 1 * inputRow* inputColumn * sizeof(Dtype));

			src_split[1] = cv::Mat(inputRow, inputColumn, CV_32FC1);
			Dtype* data_1 = (Dtype*)src_split[1].data;
			std::memcpy(data_1, inputDatum + 1 * inputRow* inputColumn, 1 * inputRow* inputColumn * sizeof(Dtype));

			src_split[2] = cv::Mat(inputRow, inputColumn, CV_32FC1);
			Dtype* data_2 = (Dtype*)src_split[2].data;
			std::memcpy(data_2, inputDatum + 2 * inputRow* inputColumn, 1 * inputRow* inputColumn * sizeof(Dtype));

			merge(src_split, 3, src);
			data = (Dtype*)src.data;

		} else {
			// cv::Mat_<Dtype> src(inputRow, inputColumn);
			src = cv::Mat(inputRow, inputColumn, CV_32FC(inputChannel));
			data = (Dtype*)src.data;
			std::memcpy(data, inputDatum, inputChannel * inputRow* inputColumn * sizeof(Dtype));
		}


#if defined(CV_SHOW)
		if (inputChannel == 3) {
			cv::Mat imgtmp(inputRow, inputColumn, CV_32FC(inputChannel));
			imgtmp.setTo(cv::Scalar(172.49, 172.94, 179.25));
			cv::add(src, imgtmp, src);
			cv::normalize(src, src, 255.0,0);
		}
#endif
		float prob;

		// 결과를 저장할 Mat과 Affine Transform을 정의하는 2D점을 저장하는 배열
		cv::Point2f srcTri[3];	// 2D point를 저장하는 원본배열
		cv::Point2f dstTri[3];	// 2D point를 저장하는 결과배열
		cv::Mat rot_mat;	// 3행 2열,32비트 실수형 채널1 Mat 생성
		cv::Mat warp_mat;	// 3행 2열,32비트 실수형 채널1 Mat 생성

		srcTri[0] = cv::Point2f(0, 0);
		srcTri[1] = cv::Point2f(src.cols - 1.f, 0);
		srcTri[2] = cv::Point2f(0, src.rows - 1.f);
		dstTri[0] = srcTri[0];
		dstTri[1] = srcTri[1];
		dstTri[2] = srcTri[2];

		cv::Point center = cv::Point(src.cols / 2, src.rows / 2);	//dst 중앙점

#if defined(CV_SHOW)
		cv::imshow("src",src);
		cv::waitKey(0);
#endif
		prob = rand() / (RAND_MAX + 1.0);
		if (prob < fliplr.probability) {
			// cv::flip(src, src, 0);	
			float tmp = dstTri[0].x;
			dstTri[0].x = dstTri[1].x;
			dstTri[1].x = tmp;
			dstTri[2].x = dstTri[0].x;
		}

		prob = rand() / (RAND_MAX + 1.0);
		if (prob < flipud.probability) {
			// cv::flip(src, src, 1);
			float tmp = dstTri[0].y;
			dstTri[0].y = dstTri[2].y;
			dstTri[1].y = dstTri[2].y;
			dstTri[2].y = tmp;

		}

		prob = rand() / (RAND_MAX + 1.0);
		if (prob < rotation.probability) {
			float angle = rand() / ((float)RAND_MAX + 1)*(rotation.angle[1] - rotation.angle[0]) + rotation.angle[0];

			// FIXME : src가 아닌 dstTri를 수정하는 방식으로 바껴야 함.
			rot_mat = cv::getRotationMatrix2D(center, angle, 1);
			cv::warpAffine(src, src, rot_mat, src.size());
		}

		// scale
		prob = rand() / (RAND_MAX + 1.0);
		if (prob < scale.probability) {
			

			float x_scale = rand() / ((float)RAND_MAX + 1)*(scale.x[1] - scale.x[0]) + scale.x[0];
			float y_scale = rand() / ((float)RAND_MAX + 1)*(scale.y[1] - scale.y[0]) + scale.y[0];
			
			dstTri[0].x = dstTri[0].x + src.cols*(1-x_scale)/2 * (center.x>dstTri[0].x?-1:1);
			dstTri[1].x = dstTri[1].x + src.cols*(1-x_scale)/2 * (center.x>dstTri[1].x?-1:1);
			dstTri[2].x = dstTri[2].x + src.cols*(1-x_scale)/2 * (center.x>dstTri[2].x?-1:1);

			dstTri[0].y = dstTri[0].y + src.rows*(1-y_scale)/2 * (center.y>dstTri[0].y?-1:1);
			dstTri[1].y = dstTri[1].y + src.rows*(1-y_scale)/2 * (center.y>dstTri[1].y?-1:1);
			dstTri[2].y = dstTri[2].y + src.rows*(1-y_scale)/2 * (center.y>dstTri[2].y?-1:1);
		}

		prob = rand() / (RAND_MAX + 1.0);
		if (prob < shear.probability) {
			float angle = rand() / ((float)RAND_MAX + 1)*(shear.angle[1] - shear.angle[0]) + shear.angle[0];
			float degree = tan(angle*PI/180);

			dstTri[0].x = dstTri[0].x + src.cols*degree/2 * (center.x>dstTri[0].x?-1:1);
			dstTri[1].x = dstTri[1].x - src.cols*degree/2 * (center.x>dstTri[1].x?-1:1);
			dstTri[2].x = dstTri[2].x - src.cols*degree/2 * (center.x>dstTri[2].x?-1:1);
		}


		prob = rand() / (RAND_MAX + 1.0);
		if (prob < translate.probability) {
			float x_trans = rand() / ((float)RAND_MAX + 1)*(translate.x[1] - translate.x[0]) + translate.x[0];
			float y_trans = rand() / ((float)RAND_MAX + 1)*(translate.y[1] - translate.y[0]) + translate.y[0];

			dstTri[0].x += src.cols*x_trans;
			dstTri[1].x += src.cols*x_trans;
			dstTri[2].x += src.cols*x_trans;
			dstTri[0].y += src.rows*y_trans;
			dstTri[1].y += src.rows*y_trans;
			dstTri[2].y +=src.rows* y_trans;
			
		}

		warp_mat = cv::getAffineTransform(srcTri, dstTri);	// 입력 매칭쌍으로 부터 affine변환을 구해서 warp_mat에 넣어줌
		int flag_interpolation;
		if (this->interpolation == "nearest")
			flag_interpolation = CV_INTER_NN;
		else if (this->interpolation == "linear")
			flag_interpolation = CV_INTER_LINEAR;
		else if (this->interpolation == "cubic")
			flag_interpolation = CV_INTER_CUBIC;
    	else if (this->interpolation == "area")
			flag_interpolation = CV_INTER_AREA; 
    	else if (this->interpolation == "lanczos4")
			flag_interpolation = CV_INTER_LANCZOS4;
		else
			flag_interpolation = 1;
		
		cv::warpAffine(src, src, warp_mat, src.size(), flag_interpolation);

		prob = rand() / (RAND_MAX + 1.0);
		if (prob < filtering.probability) {
			int ks = rand() / ((float)RAND_MAX + 1)*(filtering.kernelSize[1] - filtering.kernelSize[0]) + filtering.kernelSize[0];
			ks = (ks%2==0) ? ks+1: ks;

			if (filtering.filterType == "normal") {
				cv::blur(src, src, cv::Size(ks, ks), cv::Point(-1,-1) );
			} else if (filtering.filterType == "gaussian" || filtering.filterType == "gauss") {
				cv::GaussianBlur(src, src, cv::Size(ks, ks), 0, 0 );
			} else if (filtering.filterType == "median") {
				cv::medianBlur(src, src, ks);
			} else if (filtering.filterType == "bilateral") {
				cv::bilateralFilter(src, src, ks, ks*2, ks/2);
			} else if (filtering.filterType == "sharpen") {	// original에 gaussian 을 빼서 구현
				cv::Mat m_tmp;
				cv::GaussianBlur(src, m_tmp, cv::Size(ks, ks), 0, 0 );

				SASSERT(filtering.sharpen[0]>=0 || filtering.sharpen[1]>=0, "sharpen must be positive or zero!!!");
				float sharpen = rand() / ((float)RAND_MAX + 1)*(filtering.sharpen[1] - filtering.sharpen[0]) + filtering.sharpen[0];

				cv::addWeighted(src, 1+sharpen , m_tmp, -sharpen, 0, src);
			} else {
				SASSERT(0, "Wrong filter type!!!");
			}
		}

		prob = rand() / (RAND_MAX + 1.0);
		if (prob < noise.probability) {
			if (noise.noiseType == "gaussian" or noise.noiseType == "gauss") {
				cv::Mat noise_mat(src.size(), CV_32FC(inputChannel));
				cv::randn(noise_mat, cv::Scalar(noise.mean, noise.mean, noise.mean), cv::Scalar(noise.std, noise.std, noise.std));
				cv::add(src, noise_mat, src);
			} else if (noise.noiseType == "snp") {

				float min = noise.min_max[0];
				float max = noise.min_max[1];
				int num_of_noise_pixels = noise.amount * inputRow * inputColumn;
				for (int i = 0; i < num_of_noise_pixels; i++) {
					int r = rand() % inputRow;  // noise로 바꿀 행을 임의로 선택
					int c = rand() % inputColumn;  // noise로 바꿀 열을 임의로 선택
					int _ch = rand() % inputChannel;  // noise로 바꿀 채널의 임의로 선택

					// img.ptr<uchar>(r)은 r번째 행의 첫번째 픽셀, 첫번째 채널에 대한 주소값을 반환한다.
					Dtype* pixel = src.ptr<Dtype>(r) +(c*inputChannel) + _ch; // noise로 바꿀 위치를 계산

					*pixel = (rand() % 2 == 1) ? max : min; // black(min) 혹은 white(max)로 교체
				}
			} else{
				SASSERT(0, "Wrong noise type!!!");
			}
		}

#if defined(CV_SHOW)
		cv::imshow("src",src);
		cv::waitKey(0);
#endif

		memcpy(outputDatum, data, inputChannel * inputRow* inputColumn * sizeof(Dtype));
	}
}


/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* DataAugmentationLayer<Dtype>::initLayer() {
	DataAugmentationLayer* layer = NULL;
	SNEW(layer, DataAugmentationLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void DataAugmentationLayer<Dtype>::destroyLayer(void* instancePtr) {
    DataAugmentationLayer<Dtype>* layer = (DataAugmentationLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void DataAugmentationLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

    DataAugmentationLayer<Dtype>* layer = (DataAugmentationLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DataAugmentationLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DataAugmentationLayer<Dtype>* layer = (DataAugmentationLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void DataAugmentationLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	DataAugmentationLayer<Dtype>* layer = (DataAugmentationLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void DataAugmentationLayer<Dtype>::backwardTensor(void* instancePtr) {
	// do nothing
}

template<typename Dtype>
void DataAugmentationLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool DataAugmentationLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    if (inputShape.size() != 4) {
		SASSERT0("Data Augmentation Layer shape match error!!!");
        return false;
    }
    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);
    return true;
}

template<typename Dtype>
uint64_t DataAugmentationLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}


template class DataAugmentationLayer<float>;
