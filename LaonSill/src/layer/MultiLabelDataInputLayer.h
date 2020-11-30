/*
 * MultiLabelDataInputLayer.h
 *
 *  Created on: Jul 12, 2017
 *      Author: jkim
 */

#ifndef MULTILABELDATAINPUTLAYER_H_
#define MULTILABELDATAINPUTLAYER_H_

#include "InputLayer.h"
#include "DataReader.h"
#include "Datum.h"
#include "InputDataProvider.h"
#include "LayerFunc.h"

template <typename Dtype>
class MultiLabelDataInputLayer : public InputLayer<Dtype> {
public:
	MultiLabelDataInputLayer();
	virtual ~MultiLabelDataInputLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	virtual void reshape();
	InputPool* inputPool;

private:
	void load_batch();

private:
	DataReader<Datum> dataReader;




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

#endif /* MULTILABELDATAINPUTLAYER_H_ */
