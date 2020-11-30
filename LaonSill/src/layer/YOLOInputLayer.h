/**
 * @file YOLOInputLayer.h
 * @date 2017-12-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef YOLOINPUTLAYER_H
#define YOLOINPUTLAYER_H 

#include "InputLayer.h"
#include "DataReader.h"
#include "Datum.h"
#include "InputDataProvider.h"
#include "DataTransformer.h"
#include "LayerPropList.h"
#include "LayerFunc.h"

template<typename Dtype>
class YOLOInputLayer : public InputLayer<Dtype> {
public:
	YOLOInputLayer();
	YOLOInputLayer(_YOLOInputPropLayer* prop);
	virtual ~YOLOInputLayer();

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
	DataReader<class AnnotatedDatum> dataReader;
	std::string labelMapFile;
	DataTransformer<Dtype> dataTransformer;
	bool outputLabels;
	bool hasAnnoType;

	_YOLOInputPropLayer* prop;
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
#endif /* YOLOINPUTLAYER_H */
