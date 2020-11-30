/*
 * BatchNorm2Layer.h
 *
 *  Created on: Dec 18, 2017
 *      Author: jkim
 */

#ifndef BATCHNORM2LAYER_H_
#define BATCHNORM2LAYER_H_

#include <memory>

#include "common.h"
#include "LearnableLayer.h"
#include "LayerFunc.h"

template <typename Dtype>
class BatchNorm2Layer : public LearnableLayer<Dtype> {
public:
	BatchNorm2Layer();
	virtual ~BatchNorm2Layer();

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

	virtual void update();
	void applyChanges(LearnableLayer<Dtype> *targetLayer);
	void syncParams(LearnableLayer<Dtype> *targetLayer);
    bool hasScaleBias() { return this->scaleBias; }

private:
	void multicast_gpu(int N, int C, int S, const Dtype *x, Dtype *y );
	void compute_sum_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y);
	void compute_mean_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y);

private:
	double movingAverageFraction;
	double eps;
	int channels;
	int iter;
	bool useGlobalStats;
	//bool clipVariance;
	bool scaleBias;

	std::vector<update_param> updatePolicies;

    // cuDNN descriptors / handles
    cudnnTensorDescriptor_t fwdInputDesc, fwdOutputDesc;
    cudnnTensorDescriptor_t bwdInputDesc, bwdOutputDesc;
    cudnnTensorDescriptor_t fwdScaleBiasMeanVarDesc;
    cudnnTensorDescriptor_t bwdScaleBiasMeanVarDesc;
    cudnnBatchNormMode_t mode;

    bool handlesSetup;

    std::shared_ptr<Data<Dtype>> saveMean;
    std::shared_ptr<Data<Dtype>> saveInvVar;
    std::shared_ptr<Data<Dtype>> scaleOnes;
    std::shared_ptr<Data<Dtype>> biasZeros;


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


#endif /* BATCHNORM2LAYER_H_ */
