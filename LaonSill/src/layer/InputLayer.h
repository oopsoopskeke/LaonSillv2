/**
 * @file	InputLayer.h
 * @date	2016/5/11
 * @author	jhkim
 * @brief
 * @details
 */


#ifndef LAYER_INPUTLAYER_H_
#define LAYER_INPUTLAYER_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "Util.h"
#include "BaseLayer.h"
#include "DataSet.h"
#include "LayerFunc.h"

/**
 * @brief 입력 레이어 클래스
 * @details 입력 데이터를 그대로 출력 데이터로 전달하는 역할을 한다.
 *          특별한 기능을 담당하지 않고 입력 데이터를 한 레벨 추상화하고
 *          약간의 레어어 쓰기, 읽기 등의 부가 기능을 수행
 *          입력 레이어의 경우 자신의 레이어값 읽기, 쓰기뿐 아니라 최초의 레이어로써 뒤에 
 *          연결된 모든 레이어의 메타 정보를 읽기, 쓰기를 수행한다.
 */
template <typename Dtype>
class InputLayer : public Layer<Dtype> {
public:
	InputLayer();
	virtual ~InputLayer();

	//void feedforward(uint32_t idx, Data<Dtype>* input, const char* end=0);
	virtual void feedforward();
	using Layer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

	void reshape();

    virtual int getNumTrainData();
    virtual int getNumTestData();
    virtual void shuffleTrainDataSet();


public:
	DataSet<Dtype>* _dataSet;
	Data<Dtype>* _dataMean;
    virtual void feedImage(const int channels, const int height, const int width,
        float* image);

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


#endif /* LAYER_INPUTLAYER_H_ */
