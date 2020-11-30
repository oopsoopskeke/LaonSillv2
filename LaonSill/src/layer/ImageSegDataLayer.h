/*
 * ImageSegDataLayer.h
 *
 *  Created on: Aug 3, 2017
 *      Author: jkim
 */

#ifndef IMAGESEGDATALAYER_H_
#define IMAGESEGDATALAYER_H_

#include "InputLayer.h"
#include "DataTransformer.h"
#include "LayerFunc.h"


/*
 * Transform
 * 		- mirror, sub mean, crop applied
 */

template <typename Dtype>
class ImageSegDataLayer : public InputLayer<Dtype> {
public:
	ImageSegDataLayer();
	virtual ~ImageSegDataLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	virtual void reshape();

private:
	void load_batch();

private:
	std::vector<std::pair<std::string, std::string>> lines_;
	int lines_id_;
	DataTransformer<Dtype> imgDataTransformer;
	DataTransformer<Dtype> segDataTransformer;


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

#endif /* IMAGESEGDATALAYER_H_ */
