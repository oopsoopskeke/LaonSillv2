/*
 * DummyInputLayer.h
 *
 *  Created on: Jan 21, 2017
 *      Author: jkim
 */

#ifndef DUMMYINPUTLAYER_H_
#define DUMMYINPUTLAYER_H_

#include "InputLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class DummyInputLayer
: public InputLayer<Dtype> {

public:
	DummyInputLayer();
	virtual ~DummyInputLayer();

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	virtual void feedforward();
	virtual void reshape();


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

#endif /* DUMMYINPUTLAYER_H_ */
