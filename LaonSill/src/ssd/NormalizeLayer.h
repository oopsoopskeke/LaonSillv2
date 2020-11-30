/*
 * NormalizeLayer.h
 *
 *  Created on: Apr 21, 2017
 *      Author: jkim
 */

#ifndef NORMALIZELAYER_H_
#define NORMALIZELAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LearnableLayer.h"
#include "LayerFunc.h"

/**
 * @brief Normalizes the input to have L_p norm of 1 with scale learnable
 */
template <typename Dtype>
class NormalizeLayer : public LearnableLayer<Dtype> {
public:
	NormalizeLayer();
	virtual ~NormalizeLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

	virtual void update();
	void applyChanges(LearnableLayer<Dtype> *targetLayer);
	void syncParams(LearnableLayer<Dtype> *targetLayer);


protected:
	/*
	void initialize();
	void _updateParam(const uint32_t paramSize, const Dtype regScale, const Dtype learnScale,
		const Dtype epsilon, const Dtype decayRate, const Dtype beta1, const Dtype beta2,
		Data<Dtype>* dataHistory, Data<Dtype>* dataHistory2, Data<Dtype>* data);
		*/


private:
	/*
	bool acrossSpatial;
	bool channelShared;
	update_param scaleUpdateParam;
	param_filler<Dtype> scaleFiller;
	Dtype eps;
	*/

	// 적용될 norm term을 store
	// acrossSpatial인 경우 이미지 하나 전체에 대해 1개의 norm term. 이미지 갯수만큼
	// 아닌 경우 이미지 하나에 대해 채널간 norm term, spatialDim만큼의 norm term
	// norm 자체는 채널간 합, spatial 단위로 통합하느냐 여부에 따라 결정
	// acrossSpatial-true:  batches x 1 x 1 x 1
	// acrossSpatial-false: batches x 1 x height x width
	Data<Dtype> norm_;
	// 각 spatialDim 단위로 channel간 sum하기 위해 1로 채워진 vector
	// initialized to all 1s
	// 1 x channels x 1 x 1
	Data<Dtype> sumChannelMultiplier_;
	// 1 x 1 x height x width
	Data<Dtype> sumSpatialMultiplier_;

	// 1장의 이미지 각 element에 대한 처리 결과를 담기 위한 buffer
	// 1 x channels x height x width
	Data<Dtype> buffer_;
	// 이미지당 channel별 scalar를 저장하기 위한 buffer
	// 1 x channels x 1 x 1
	Data<Dtype> bufferChannel_;
	// 이미지당 spatialDim별 scalar를 저장하기 위한 buffer
	// 1 x 1 x height x width
	Data<Dtype> bufferSpatial_;



	int tempCount;




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

#endif /* NORMALIZELAYER_H_ */
