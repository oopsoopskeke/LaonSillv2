/**
 * @file	FullyConnectedLayer.h
 * @date	2016/5/10
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_FULLYCONNECTEDLAYER_H_
#define LAYER_FULLYCONNECTEDLAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"
#include "LayerFunc.h"

/**
 * @brief Fully Connected (Inner Product) 레이어
 * @details 이전 레이어와 현재 레이어의 모든 노드들에 대해 연결성이 있고
 *          연결성을 통해 weighted sum, activation을 수행 출력값을 계산하는 레이어이다.
 *          입력 레이어가 다차원인 경우(이미지의 경우 height x width x channel의 3차원) 
 *          1차원으로 flatten((height*width*channel) x 1 x 1)된다.
 *          출력 역시 1차원 flatten 결과이며 필요에 따라서 입력받는 레이어에서 다시 차원을
 *          복구해야 한다.
 */
template <typename Dtype>
class FullyConnectedLayer : public LearnableLayer<Dtype> {
public:
    FullyConnectedLayer();
	virtual ~FullyConnectedLayer();

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

	virtual void update();
    void applyChanges(LearnableLayer<Dtype> *targetLayer);
    void syncParams(LearnableLayer<Dtype> *targetLayer);

    virtual void saveParams(std::ofstream& ofs);

protected:
	void _computeWeightedData();
	void _computeWeightBiasedData();
	void _computeWeightGrad();
	void _computeBiasGrad();
	void _computeInputGrad();

	enum ParamType {
		Weight = 0,
		Bias = 1
	};

protected:
	//Dtype* d_onevec;    ///< batch 사이즈의 1 벡터, bias를 weighted sum에 더해 줄 때 사용
	SyncMem<Dtype> _onevec;
	SyncMem<Dtype> _mask;


public:
    uint32_t batches;
    uint32_t in_rows;
    uint32_t out_rows;

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

#endif /* LAYER_FULLYCONNECTEDLAYER_H_ */
