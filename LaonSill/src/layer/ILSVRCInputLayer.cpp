/**
 * @file ILSVRCInputLayer.cpp
 * @date 2017-04-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <sys/types.h>
#include <dirent.h>
#include <fcntl.h>

#include <opencv2/opencv.hpp>

#include <boost/range/algorithm.hpp>

#include "common.h"
#include "ILSVRCInputLayer.h"
#include "InputLayer.h"
#include "Network.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "WorkContext.h"
#include "MemoryMgmt.h"

using namespace std;

#define ILSVRCINPUTLAYER_LOG        0

const int ILSVRC_IMAGE_CHANNEL   = 3;
const int ILSVRC_IMAGE_ROW       = 224;
const int ILSVRC_IMAGE_COL       = 224;

const int ILSVRC_CLASS_COUNT     = 1000;

#define MAKE_LABEL_FOR_SOFTMAX_OUTPUT       1

template<typename Dtype>
ILSVRCInputLayer<Dtype>::ILSVRCInputLayer() : InputLayer<Dtype>() {
    this->type = Layer<Dtype>::ILSVRCInput;

    this->imageRow = ILSVRC_IMAGE_ROW;
    this->imageCol = ILSVRC_IMAGE_COL;

    if (SLPROP(ILSVRCInput, resizeImage)) {
        this->imageRow = SLPROP(ILSVRCInput, resizedImageRow);
        this->imageCol = SLPROP(ILSVRCInput, resizedImageCol);
    }

    this->imageChannel = ILSVRC_IMAGE_CHANNEL;

    this->images = NULL;
    this->labels = NULL;

    this->currentBatchIndex = 0;
}

template<typename Dtype>
ILSVRCInputLayer<Dtype>::~ILSVRCInputLayer() {
    if (this->images != NULL) {
        SFREE(this->images); 
    }

    if (this->labels != NULL) {
        SFREE(this->labels);
    }
}

template <typename Dtype>
void ILSVRCInputLayer<Dtype>::reshape() {
    int batchSize = SNPROP(batchSize);

	if (this->images == NULL) {
        SASSERT0(this->labels == NULL);
        fillMetas();

        unsigned long allocImageSize = 
            (unsigned long)sizeof(Dtype) * 
            (unsigned long)this->imageRow * 
            (unsigned long)this->imageCol * 
            (unsigned long)this->imageChannel * 
            (unsigned long)batchSize;

        this->images = NULL;
        SMALLOC(this->images, Dtype, allocImageSize);
        SASSERT0(this->images != NULL);

        unsigned long allocLabelSize = 
            (unsigned long)sizeof(Dtype) * 
#if MAKE_LABEL_FOR_SOFTMAX_OUTPUT
            1UL *
#else
            (unsigned long)ILSVRC_CLASS_COUNT *
#endif
            (unsigned long)batchSize;

        this->labels = NULL;
        SMALLOC(this->labels, Dtype, allocLabelSize);
        SASSERT0(this->labels != NULL);
	} else {
        SASSERT0(this->labels != NULL);
    }

	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}

	Layer<Dtype>::_adjustInputShape();

    this->_inputShape[0][0] = batchSize;
    this->_inputShape[0][1] = this->imageChannel;
    this->_inputShape[0][2] = this->imageRow;
    this->_inputShape[0][3] = this->imageCol;
    this->_inputData[0]->reshape(this->_inputShape[0]);

    this->_inputShape[1][0] = batchSize;
    this->_inputShape[1][1] = 1;
#if MAKE_LABEL_FOR_SOFTMAX_OUTPUT
    this->_inputShape[1][2] = 1;
#else
    this->_inputShape[1][2] = ILSVRC_CLASS_COUNT;
#endif
    this->_inputShape[1][3] = 1;
    this->_inputData[1]->reshape(this->_inputShape[1]);

#if ILSVRCINPUTLAYER_LOG
    printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batchSize, this->imageChannel, this->imageRow, this->imageCol);
    printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
#if MAKE_LABEL_FOR_SOFTMAX_OUTPUT
        SLPROP_BASE(name).c_str(), batchSize, 1, 1, 1);
#else
        SLPROP_BASE(name).c_str(), batchSize, 1, ILSVRC_CLASS_COUNT, 1);
#endif

#endif

    loadImages(this->currentBatchIndex);
    loadLabels(this->currentBatchIndex);

    int inputImageSize = this->imageChannel * this->imageRow * this->imageCol * batchSize;
    this->_inputData[0]->set_device_with_host_data(this->images, 0, inputImageSize);

#if MAKE_LABEL_FOR_SOFTMAX_OUTPUT
    int inputLabelSize = batchSize;
#else
    int inputLabelSize = ILSVRC_CLASS_COUNT * batchSize;
#endif
    this->_inputData[1]->set_device_with_host_data(this->labels, 0, inputLabelSize);
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::loadPixels(cv::Mat image, int imageIndex) {
    unsigned long offset = (unsigned long)imageIndex * 
        (this->imageRow * this->imageCol * this->imageChannel);

    // XXX: find better method T_T
    // Red
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[2] / 127.5 - 1.0;
            offset++;
        }
    }

    // Green
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[1] / 127.5 - 1.0;
            offset++;
        }
    }

    // Blue
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[0] / 127.5 - 1.0;
            offset++;
        }
    }
}

#define ILSVRC_CLASSFILE_NAME        "ilsvrc.txt"

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::fillMetas() {
    string filePath = SLPROP(ILSVRCInput, imageDir) + "/" + ILSVRC_CLASSFILE_NAME;
    FILE *fp = fopen(filePath.c_str(), "r");
    SASSERT0(fp != NULL);

    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    char imageFilePath[1024];
    int classID;

    int metaIndex = 0;

    while ((read = getline(&line, &len, fp)) != -1) {
        ILSVRCMeta meta;
        int ret = sscanf(line, "%s %d", imageFilePath, &classID);
        SASSERT0(ret == 2);
        SASSERT0(classID < ILSVRC_CLASS_COUNT);
        meta.classID = classID;
        meta.filePath = imageFilePath;
        this->metas.push_back(meta);
        this->metaIndexes.push_back(metaIndex);
        metaIndex++;
    }
   
    if (line)
        free(line);     // do not use SFREE because line is allocated from getline()

    fclose(fp);
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::loadImages(int baseIdx) {
    int batchSize = SNPROP(batchSize);

    for (int i = 0; i < batchSize; i++) {
        int index = i + baseIdx;
        if (index >= this->metas.size())
            break;

        int shuffledIndex = this->metaIndexes[index];
        string imagePath = this->metas[shuffledIndex].filePath;

        cv::Mat image;
        image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

        // XXX: 좀더 general 하게 만들자.
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(this->imageRow, this->imageCol));

        loadPixels(resizedImage, i);
    }
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::loadLabels(int baseIdx) {
    int batchSize = SNPROP(batchSize);

    for (int i = 0; i < batchSize; i++) {
        int index = i + baseIdx;
        if (index >= this->metas.size())
            break;

        int shuffledIndex = this->metaIndexes[index];
        int classID = this->metas[shuffledIndex].classID;
        
        SASSERT0(classID < ILSVRC_CLASS_COUNT);
        SASSERT0(classID >= 0);
#if MAKE_LABEL_FOR_SOFTMAX_OUTPUT
        this->labels[i] = (Dtype)classID;
#else
        for (int j = 0; j < ILSVRC_CLASS_COUNT; j++) {
            this->labels[i * ILSVRC_CLASS_COUNT + j] = 0.0;
        }

        this->labels[i * ILSVRC_CLASS_COUNT + classID] = 1.0;
#endif
    }
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::shuffleImages() {
    // FIXME: move it to other source.
#if 1
    srand(time(NULL)); 
    boost::range::random_shuffle(this->metaIndexes);
#endif
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    if (baseIndex == 0)
        shuffleTrainDataSet();

    this->currentBatchIndex = baseIndex;
    reshape();
}

template<typename Dtype>
int ILSVRCInputLayer<Dtype>::getNumTrainData() {
    if (this->images == NULL) {
        reshape();
    }
    return this->metas.size();
}

template<typename Dtype>
int ILSVRCInputLayer<Dtype>::getNumTestData() {
    return 0;
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->images == NULL) {
        reshape();
    }
    shuffleImages();
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* ILSVRCInputLayer<Dtype>::initLayer() {
	ILSVRCInputLayer* layer = NULL;
	SNEW(layer, ILSVRCInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    ILSVRCInputLayer<Dtype>* layer = (ILSVRCInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    ILSVRCInputLayer<Dtype>* layer = (ILSVRCInputLayer<Dtype>*)instancePtr;

    SASSERT0(!isInput);
    SASSERT0(index < 2);
    SASSERT0(layer->_outputData.size() == index);
    layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool ILSVRCInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ILSVRCInputLayer<Dtype>* layer = (ILSVRCInputLayer<Dtype>*)instancePtr;
    layer->currentBatchIndex = 0;
    layer->reshape();

    if (SNPROP(miniBatch) == 0) {
        int trainDataNum = layer->getNumTrainData();
        if (trainDataNum % SNPROP(batchSize) == 0) {
            SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize);
        } else {
            SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize) + 1;
        }
        WorkContext::curPlanInfo->miniBatchCount = SNPROP(miniBatch);
    }

    return true;
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    ILSVRCInputLayer<Dtype>* layer = (ILSVRCInputLayer<Dtype>*)instancePtr;
    layer->feedforward(miniBatchIdx);
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing..
}

template<typename Dtype>
void ILSVRCInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool ILSVRCInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    // XXX: 현재 ILSVRCInputLayer를 사용하는 모듈은 오직 YOLO pretrain 모듈이다. 
    //      해당 레이어는 추후에 삭제가 될 예정이기 때문에 YOLO pretrain 모듈에서만
    //      동작하는 수준으로만 checkShape 함수를 맞추도록 하겠다.
    
    TensorShape outputShape1;
    outputShape1.N = SNPROP(batchSize);
    outputShape1.C = 3;
    outputShape1.H = SLPROP(ILSVRCInput, resizedImageRow);
    outputShape1.W = SLPROP(ILSVRCInput, resizedImageCol);
    outputShape.push_back(outputShape1);

    TensorShape outputShape2;
    outputShape2.N = SNPROP(batchSize);
    outputShape2.C = 1;
    outputShape2.H = ILSVRC_CLASS_COUNT;
    outputShape2.W = 1;
    outputShape.push_back(outputShape2);

    return true;
}

template<typename Dtype>
uint64_t ILSVRCInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class ILSVRCInputLayer<float>;
