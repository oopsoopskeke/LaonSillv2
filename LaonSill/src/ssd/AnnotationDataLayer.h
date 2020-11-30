/*
 * AnnotationDataLayer.h
 *
 *  Created on: Apr 19, 2017
 *      Author: jkim
 */

#ifndef ANNOTATIONDATALAYER_H_
#define ANNOTATIONDATALAYER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "InputLayer.h"
#include "ssd_common.h"
#include "LayerFunc.h"



template <typename Dtype>
class AnnotationDataLayer : public InputLayer<Dtype> {
public:
	AnnotationDataLayer();
	virtual ~AnnotationDataLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

	virtual void reshape();

	virtual int getNumTrainData();
	virtual void shuffleTrainDataSet();

	virtual void feedImage(const int channels, const int height, const int width,
            float* image);

private:
	void initialize();
	void shuffle();

	void getNextMiniBatch();
	void getNextMiniBatchInds(std::vector<int>& inds);
	void getMiniBatch(const std::vector<int>& inds);

	/**
	 * @details 데이터셋 파일로부터 img, anno 파일 경로로 ODRawDataList 초기화
	 */
	void loadODRawDataPath();
	/**
	 * @details img 파일을 cv::Mat 형태로 ODRawDataList에 읽어들임. 스케일링 적용.
	 */
	void loadODRawDataIm();
	/**
	 * @details anno 파일을 ODRawDataList에 읽어들임.
	 */
	void loadODRawDataAnno();
	void readAnnotation(ODRawData<Dtype>& odRawData);
	void loadODMetaData();

	void buildLabelData(ODMetaData<Dtype>& odMetaData, int bbIdx, Dtype buf[8]);


	void verifyData();
	void printMat(cv::Mat& im, int type);
	void printArray(Dtype* array, int n);
private:
	/*
	bool flip;
	uint32_t imageHeight;	///< 네트워크로 입력되는 이미지 높이. 리사이즈 높이.
	uint32_t imageWidth;	///< 네트워크로 입력되는 이미지 너비. 리사이즈 너비.

	std::string imageSetPath;	///< OD 데이터셋을 정의한 파일의 경로. <img, anno> 페어정보
	std::string baseDataPath;

	std::vector<Dtype> pixelMeans;
	bool bShuffle;
	*/

	std::vector<ODRawData<Dtype>> odRawDataList;
	std::vector<ODMetaData<Dtype>> odMetaDataList;

	LabelMap<Dtype> labelMap;

	std::vector<int> perm;
	int cur;

	Data<Dtype> data;	///< feedforward할 data를 준비하는 buffer.

	int channels;
	int height;
	int width;
	float* image;

public:
	std::map<std::string, int> refCount;





public:
    /****************************************************************************
     * layer callback functions
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
    static bool checkShape(std::vector<TensorShape> inputShape,
            std::vector<TensorShape> &outputShape);
    static uint64_t calcGPUSize(std::vector<TensorShape> inputShape);
};

#endif /* ANNOTATIONDATALAYER_H_ */
