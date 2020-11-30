/**
 * @file KistiInputLayer.cpp
 * @date 2017-03-28
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <fcntl.h>

#include <iostream>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "KistiInputLayer.h"
#include "InputLayer.h"
#include "Network.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "ImageUtil.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

#define KISTIINPUTLAYER_LOG        0
// FIXME: 다른 방식으로 file path를 얻자. 
#define ETRI_TOP1000_KEYWORD_FILENAME       "top1000keywords.txt"
#define ETRI_KEYWORD_FILENAME               "keywords.txt"

const int ETRIDATA_IMAGE_CHANNEL = 3;

template<typename Dtype>
KistiInputLayer<Dtype>::KistiInputLayer() : InputLayer<Dtype>() {
    this->type = Layer<Dtype>::KistiInput;
    this->imageRow = SLPROP(KistiInput, resizedImageRow);
    this->imageCol = SLPROP(KistiInput, resizedImageCol);
    this->imageChannel = ETRIDATA_IMAGE_CHANNEL;

    this->images = NULL;
    this->labels = NULL;
    this->currentBatchIndex = 0;
}

template<typename Dtype>
KistiInputLayer<Dtype>::~KistiInputLayer() {
    if (this->images != NULL) {
        SFREE(this->images); 
    }

    if (this->labels != NULL) {
        SFREE(this->labels);
    }
}

template<typename Dtype>
void KistiInputLayer<Dtype>::prepareKeywordMap() {
    int index = 0;
    string top1000KeywordFilePath =
        SLPROP(KistiInput, imageDir) + "/" + ETRI_TOP1000_KEYWORD_FILENAME;

    ifstream input(top1000KeywordFilePath.c_str());
    string line;
    
    SASSERT0(input.is_open());

    while (input.good()) {
        getline(input, line);
        if (line == "")
            break;

        this->keywordMap[line] = index;
        index++;
    }

    input.close();


#if 1
    map <string, int>::iterator iter;
    cout << "Keyword Map" << endl;
    cout << "=================================================================" << endl;
    for (iter = this->keywordMap.begin(); iter != this->keywordMap.end(); iter++) {
        cout << iter->second << " : " << iter->first << endl;
    }
    cout << "=================================================================" << endl;

#endif


    SASSERT((this->keywordMap.size() <= SLPROP(KistiInput, labelCount)),
        "keyword count of etri data should be less than %d but %d.",
        SLPROP(KistiInput, labelCount), (int)this->keywordMap.size());
}

template<typename Dtype>
void KistiInputLayer<Dtype>::registerData(string filePath, bool isTrainData) {
    // (1) read keywords
    string keywordFilePath = filePath + "/" + ETRI_KEYWORD_FILENAME;

    ifstream input(keywordFilePath.c_str());
    string line;
    
    SASSERT0(input.is_open());

    vector<int> labels;

    while (input.good()) {
        getline(input, line);
        if (this->keywordMap.find(line) != this->keywordMap.end()) {
            labels.push_back(this->keywordMap[line]);
        } 
    }

    input.close();

    if (labels.size() == 0)
        return;

    // (2) walk directory
    struct dirent *entry;
    DIR *dp;

    dp = opendir(filePath.c_str());
    SASSERT0(dp != NULL);

    vector<string> imageFileList;

    while ((entry = readdir(dp))) {
        string imageFileName(entry->d_name);
        string imageFilePath = filePath + "/" + imageFileName;

        if (imageFilePath.find(".jpg") != string::npos) {
            imageFileList.push_back(imageFilePath); 
        }
    }

    closedir(dp);

    // (3) register data
    //   1st data => test data
    //   others   => training data
    //   FIXME: inefficient..
  
    if (SLPROP(KistiInput, useKistiPolicy) && (imageFileList.size() < 4))
        return;

    for (int i = 0; i < imageFileList.size(); i++) {
        KistiData newData;
        newData.filePath = imageFileList[i];

        for (int j = 0; j < labels.size(); j++) {
            newData.labels.push_back(labels[j]);
        }
        if (SLPROP(KistiInput, useKistiPolicy)) {
            if (i % 4 == 0)
                this->testData.push_back(newData);        
            else
                this->trainData.push_back(newData);
        } else if (isTrainData) {
            this->trainData.push_back(newData);
        } else {
            this->testData.push_back(newData);        
        }
    }
}


template<typename Dtype>
void KistiInputLayer<Dtype>::prepareData() {
    struct dirent *entry;
    DIR *dp;

    dp = opendir(SLPROP(KistiInput, imageDir).c_str());
    SASSERT0(dp != NULL);

    int step = 0;

    struct stat s;
    while ((entry = readdir(dp))) {
        string fileName(entry->d_name);
        if (fileName == "." || fileName == "..")
            continue;

        string filePath = SLPROP(KistiInput, imageDir) + "/" + fileName;

        if (stat (filePath.c_str(), &s) == 0) {
            if (s.st_mode & S_IFDIR) {
                if (step % 4 == 3)
                    registerData(filePath, false);
                else
                    registerData(filePath, true);
            }
        }

        step++;
    }

    closedir(dp);
}

template<typename Dtype>
void KistiInputLayer<Dtype>::loadPixels(cv::Mat image, int imageIndex) {
    unsigned long offset = (unsigned long)imageIndex * 
        (this->imageRow * this->imageCol * this->imageChannel);

    // XXX: find better method T_T
    // Red
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[2] / 127.5 - 1.0;
            //this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[2];
            offset++;
        }
    }

    // Green
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[1] / 127.5 - 1.0;
            //this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[1];
            offset++;
        }
    }

    // Blue
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[0] / 127.5 - 1.0;
            //this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[0];
            offset++;
        }
    }
}

template<typename Dtype>
void KistiInputLayer<Dtype>::loadImages(int batchIndex) {
    int batchSize = SNPROP(batchSize);
    int baseIndex = batchIndex;

    for (int i = 0; i < batchSize; i++) {
        int index = baseIndex + i;

        if (SLPROP(KistiInput, train)) {
            SASSERT(index < this->trainData.size(),
                "index sholud be less than data count. index=%d, data count=%d",
                index, (int)this->trainData.size());
        } else {
            SASSERT(index < this->testData.size(),
                "index sholud be less than data count. index=%d, data count=%d",
                index, (int)this->testData.size());
        }

        cv::Mat image;
        string imagePath;

        // XXX: 
        if (SLPROP(KistiInput, train))
            imagePath = this->trainData[index].filePath;
        else
            imagePath = this->testData[index].filePath;

        image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

        int imageChannels = image.channels();
        SASSERT(imageChannels == ETRIDATA_IMAGE_CHANNEL, "channel : %d", imageChannels);

        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(this->imageRow, this->imageCol));
        loadPixels(resizedImage, i);
    }
}

template<typename Dtype>
void KistiInputLayer<Dtype>::loadLabels(int batchIndex) {
    int batchSize = SNPROP(batchSize);
    int baseIndex = batchIndex;

    int totalSize = sizeof(Dtype) * SLPROP(KistiInput, labelCount) * batchSize;
    memset(this->labels, 0x00, totalSize);

    for (int i = 0; i < batchSize; i++) {
        int index = baseIndex + i;

        if (SLPROP(KistiInput, train)) {
            SASSERT(index < this->trainData.size(),
                "index sholud be less than data count. index=%d, data count=%d",
                index, (int)this->trainData.size());
        } else {
            SASSERT(index < this->testData.size(),
                "index sholud be less than data count. index=%d, data count=%d",
                index, (int)this->testData.size());
        }

        vector<int> curLabels;

        // XXX: 
        if (SLPROP(KistiInput, train))
            curLabels = this->trainData[index].labels;
        else
            curLabels = this->testData[index].labels;

        for (int j = 0; j < curLabels.size(); j++) {
            int pos = curLabels[j];
            SASSERT0(pos < SLPROP(KistiInput, labelCount));
            this->labels[i * SLPROP(KistiInput, labelCount) + pos] = 1.0;
        }
    }
}

template <typename Dtype>
void KistiInputLayer<Dtype>::reshape() {
    int batchSize = SNPROP(batchSize);

    if (this->images == NULL) {
        unsigned long allocSize = 
            (unsigned long)sizeof(Dtype) * 
            (unsigned long)this->imageRow * 
            (unsigned long)this->imageCol * 
            (unsigned long)this->imageChannel * 
            (unsigned long)batchSize;

        this->images = NULL;
        SMALLOC(this->images, Dtype, allocSize);
        SASSERT0(this->images != NULL);
        // prepare keyword map

        prepareKeywordMap();

        // prepare training & test data
        prepareData();

        SASSERT0(this->labels == NULL);
        unsigned long labelAllocSize = 
            (unsigned long)sizeof(Dtype) * 
            (unsigned long)SLPROP(KistiInput, labelCount) *
            (unsigned long)batchSize;

        SMALLOC(this->labels, Dtype, labelAllocSize);
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

#if KISTIINPUTLAYER_LOG
    printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batchSize, this->imageChannel, this->imageRow, this->imageCol);
#endif

    this->_inputShape[1][0] = batchSize;
    this->_inputShape[1][1] = 1;
    this->_inputShape[1][2] = SLPROP(KistiInput, labelCount);
    this->_inputShape[1][3] = 1;

    this->_inputData[1]->reshape(this->_inputShape[1]);

#if KISTIINPUTLAYER_LOG
    printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batchSize, 1, SLPROP(KistiInput, labelCount), 1);
#endif

    loadImages(this->currentBatchIndex);
    loadLabels(this->currentBatchIndex);

    int inputImageSize = this->imageChannel * this->imageRow * this->imageCol * batchSize;
    int inputLabelSize = SLPROP(KistiInput, labelCount) * batchSize;

    this->_inputData[0]->set_device_with_host_data(this->images, 0, inputImageSize);
    this->_inputData[1]->set_device_with_host_data(this->labels, 0, inputLabelSize);
}

template<typename Dtype>
void KistiInputLayer<Dtype>::shuffleImages() {
    // FIXME: move it to other source.
#if 0
    srand(time(NULL)); 
    boost::range::random_shuffle(this->imageIndexes);
#endif
}

template<typename Dtype>
void KistiInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void KistiInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    this->currentBatchIndex = baseIndex;    // FIXME: ...
    reshape();
}

template<typename Dtype>
int KistiInputLayer<Dtype>::getNumTrainData() {
    if (this->images == NULL) {
        reshape();
    }
    return this->trainData.size();
}

template<typename Dtype>
int KistiInputLayer<Dtype>::getNumTestData() {
    if (this->images == NULL) {
        reshape();
    }

    return this->testData.size();
}

template<typename Dtype>
void KistiInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->images == NULL) {
        reshape();
    }
    shuffleImages();
}

template<typename Dtype>
void KistiInputLayer<Dtype>::setTrain(bool train) {
    SLPROP(KistiInput, train) = train;
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* KistiInputLayer<Dtype>::initLayer() {
	KistiInputLayer* layer = NULL;
	SNEW(layer, KistiInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void KistiInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    KistiInputLayer<Dtype>* layer = (KistiInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void KistiInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index < 2);

    KistiInputLayer<Dtype>* layer = (KistiInputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(false);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool KistiInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    KistiInputLayer<Dtype>* layer = (KistiInputLayer<Dtype>*)instancePtr;
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
void KistiInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    KistiInputLayer<Dtype>* layer = (KistiInputLayer<Dtype>*)instancePtr;
    layer->feedforward(miniBatchIdx);
}

template<typename Dtype>
void KistiInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void KistiInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool KistiInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {
  
    // XXX: 현재는 KistiInputLayer를 사용하는 모듈은 오직 KistiKeyword 예제이다. 
    //      해당 레이어는 추후에 삭제가 될 예정이기 때문에 KistiKeryword 모듈에만 적합하게
    //      동작하는 수준으로만 checkShape 함수를 맞추도록 하겠다.

    TensorShape outputShape1;
    outputShape1.N = SNPROP(batchSize);
    outputShape1.C = 3;
    outputShape1.H = SLPROP(KistiInput, resizedImageRow);
    outputShape1.W = SLPROP(KistiInput, resizedImageCol);
    outputShape.push_back(outputShape1);

    TensorShape outputShape2;
    outputShape2.N = SNPROP(batchSize);
    outputShape2.C = 1;
    outputShape2.H = SLPROP(KistiInput, labelCount);
    outputShape2.W = 1;
    outputShape.push_back(outputShape2);

    return true;
}

template<typename Dtype>
uint64_t KistiInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class KistiInputLayer<float>;
