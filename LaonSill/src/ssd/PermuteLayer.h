/*
 * PermuteLayer.h
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#ifndef PERMUTELAYER_H_
#define PERMUTELAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LayerFunc.h"

/*
 * @brief Permute the input blob by changing the memory order of the data
 */
template <typename Dtype>
class PermuteLayer : public Layer<Dtype> {
public:
	PermuteLayer();
	virtual ~PermuteLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	//void initialize();


private:
	//std::vector<uint32_t> orders;


	uint32_t numAxes;
	bool needPermute;

	Data<int> permuteOrder_;
	Data<int> oldSteps_;
	Data<int> newSteps_;



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

#endif /* PERMUTELAYER_H_ */
