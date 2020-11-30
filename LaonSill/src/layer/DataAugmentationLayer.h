/**
 * @file	DataAugmentationLayer.h
 * @date	2018/3/29
 * @author	heim
 * @brief
 * @details
 */


#ifndef LAYER_DATAAUGMENTATIONLAYER_H_
#define LAYER_DATAAUGMENTATIONLAYER_H_

#include <string>
#include <vector>

#include "common.h"
#include "Util.h"
#include "BaseLayer.h"
#include "LayerFunc.h"
#include "LayerPropList.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
 * @brief Data Augmentation Layer
 * @details 입력 데이터를 변형하여 출력 데이터로 전달한다. 반드시 1 또는 3 channel 인  input layer 다음에 연결한다.
 *          augmentation은 학습 데이터가 적고 편향되 있는 경우 효과를 낼 수 있다. 이 layer는 학습 데이터가 적을 때, 다양한 변형을 가하여 dataset을 늘리는 효과를 낸다.
 *          각 기능 별로 확률과 범위을 설정할 수 있으며, 여러 augmentation 레이어를 직렬로 연결하여 sequential 한 구현이 가능하다.
 *          affine에 포함된 scale, translate, sheer은 하나의 layer에 적용 할 것을 추천한다.
 *         rotation 은 입력 이미지의 중심을 기준으로 회전하므로 첫 번째 레이어에서 먼저 동작하도록 설정하는 것을 추천한다.
 * 
 */
template <typename Dtype>
class DataAugmentationLayer : public Layer<Dtype> {
public:
	DataAugmentationLayer();
	virtual ~DataAugmentationLayer();

	virtual void reshape();
	virtual void feedforward();

public:
    AugmentationParam fliplr;
    AugmentationParam flipud;
    AugmentationParam scale;
    AugmentationParam translate;
    AugmentationParam rotation;
    AugmentationParam shear;
    AugmentationParam filtering;    // filter 이름 사용 안됨
    AugmentationParam noise;

    std::string interpolation;
    std::string fillMode;
    float filled;



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


#endif /* LAYER_DATAAUGMENTATIONLAYER_H_ */
