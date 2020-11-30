/**
 * @file CelebAInputLayer.cpp
 * @date 2017-02-20
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
#include "CelebAInputLayer.h"
#include "InputLayer.h"
#include "Network.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

#define CELEBAINPUTLAYER_LOG        0

const int CELEBA_IMAGE_CHANNEL = 3;
const int CELEBA_IMAGE_ROW = 218;
const int CELEBA_IMAGE_COL = 178;

//const int CELEBA_CENTER_CROP_LEN = 108;

template<typename Dtype>
CelebAInputLayer<Dtype>::CelebAInputLayer() : InputLayer<Dtype>() {
    this->type = Layer<Dtype>::CelebAInput;

    this->imageRow = CELEBA_IMAGE_ROW;
    this->imageCol = CELEBA_IMAGE_COL;

    if (SLPROP(CelebAInput, cropImage)) {
        this->imageRow = SLPROP(CelebAInput, cropLen);
        this->imageCol = SLPROP(CelebAInput, cropLen);
    } 
    
    if (SLPROP(CelebAInput, resizeImage)) {
        this->imageRow = SLPROP(CelebAInput, resizedImageRow);
        this->imageCol = SLPROP(CelebAInput, resizedImageCol);
    }

    this->imageChannel = CELEBA_IMAGE_CHANNEL;

    this->images = NULL;
    this->currentBatchIndex = 0;
}

template<typename Dtype>
CelebAInputLayer<Dtype>::~CelebAInputLayer() {
    if (this->images != NULL) {
        SFREE(this->images); 
    }
}

template <typename Dtype>
void CelebAInputLayer<Dtype>::reshape() {
    int batchSize = SNPROP(batchSize);

	if (this->images == NULL) {
        fillImagePaths();

        unsigned long allocSize = 
            (unsigned long)sizeof(Dtype) * 
            (unsigned long)this->imageRow * 
            (unsigned long)this->imageCol * 
            (unsigned long)this->imageChannel * 
            (unsigned long)batchSize;

        SMALLOC(this->images, Dtype, allocSize);
        SASSERT0(this->images != NULL);
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

#if CELEBAINPUTLAYER_LOG
    printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batchSize, this->imageChannel, this->imageRow, this->imageCol);
#endif

    loadImages(this->currentBatchIndex);

    int inputImageSize = this->imageChannel * this->imageRow * this->imageCol * batchSize;

    this->_inputData[0]->set_device_with_host_data(this->images, 0, inputImageSize);
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::loadPixels(cv::Mat image, int imageIndex) {
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

template<typename Dtype>
void CelebAInputLayer<Dtype>::fillImagePaths() {
    vector<string> filePath;
    struct dirent *entry;
    DIR *dp;

    dp = opendir(SLPROP(CelebAInput, imageDir).c_str());
    if (dp == NULL) {
        COLD_LOG(ColdLog::ERROR, true, "opendir() is failed. errno=%d", errno); 
        SASSERT0("opendir() is failed.");
    }

    while (entry = readdir(dp)) {
        string filename(entry->d_name);
        if (filename.find(".jpg") != string::npos) {
            this->imageIndexes.push_back(this->imagePaths.size());
            this->imagePaths.push_back(SLPROP(CelebAInput, imageDir) + "/" + filename);
        }
    }

    closedir(dp);
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::loadImages(int baseIdx) {
    int batchSize = SNPROP(batchSize);

    // (1) load jpeg
    for (int i = 0; i < batchSize; i++) {
        int index = i + baseIdx;
        if (index >= this->imagePaths.size())
            break;

        int shuffledIndex = this->imageIndexes[index];
        string imagePath = this->imagePaths[shuffledIndex];

        cv::Mat image;
        image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

        // XXX: 좀더 general 하게 만들자.
        //int imageCols = image.cols;
        //int imageRows = image.rows;
        //int imageChannels = image.channels();
        //SASSERT(imageCols == CELEBA_IMAGE_COL, "col : %d", imageCols);
        //SASSERT(imageRows == CELEBA_IMAGE_ROW, "row : %d", imageRows);
        //SASSERT(imageChannels == CELEBA_IMAGE_CHANNEL, "channel : %d", imageChannels);

        cv::Mat croppedImage;
        if (SLPROP(CelebAInput, cropImage)) {
            cv::Rect roi;
            roi.x = (CELEBA_IMAGE_COL - SLPROP(CelebAInput, cropLen)) / 2;
            roi.y = (CELEBA_IMAGE_ROW - SLPROP(CelebAInput, cropLen)) / 2;
            roi.width = SLPROP(CelebAInput, cropLen);
            roi.height = SLPROP(CelebAInput, cropLen);
            croppedImage = image(roi);

            if (!SLPROP(CelebAInput, resizeImage)) {
                loadPixels(croppedImage, i);
            }
        }

        if (SLPROP(CelebAInput, resizeImage)) {
            cv::Mat resizedImage;

            if (SLPROP(CelebAInput, cropImage)) {
                cv::resize(croppedImage, resizedImage, cv::Size(this->imageRow, this->imageCol));
            } else {
                cv::resize(image, resizedImage, cv::Size(this->imageRow, this->imageCol));
            }

            loadPixels(resizedImage, i);
        }
    }
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::shuffleImages() {
    // FIXME: move it to other source.
#if 0
    srand(time(NULL)); 
    boost::range::random_shuffle(this->imageIndexes);
#endif
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    this->currentBatchIndex = baseIndex;    // FIXME: ...
    reshape();
}

template<typename Dtype>
int CelebAInputLayer<Dtype>::getNumTrainData() {
    if (this->images == NULL) {
        reshape();
    }
    return this->imagePaths.size();
}

template<typename Dtype>
int CelebAInputLayer<Dtype>::getNumTestData() {
    return 0;
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->images == NULL) {
        reshape();
    }
    shuffleImages();
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* CelebAInputLayer<Dtype>::initLayer() {
	CelebAInputLayer* layer = NULL;
	SNEW(layer, CelebAInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    CelebAInputLayer<Dtype>* layer = (CelebAInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    CelebAInputLayer<Dtype>* layer = (CelebAInputLayer<Dtype>*)instancePtr;

    SASSERT0(!isInput);
    SASSERT0(index < 2);
    SASSERT0(layer->_outputData.size() == index);
    layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool CelebAInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    CelebAInputLayer<Dtype>* layer = (CelebAInputLayer<Dtype>*)instancePtr;
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
void CelebAInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    CelebAInputLayer<Dtype>* layer = (CelebAInputLayer<Dtype>*)instancePtr;
    layer->feedforward(miniBatchIdx);
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing 
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool CelebAInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    // XXX: 현재는 celebAInputLayer를 사용하는 모듈은 오직 GAN 모듈이다. 
    //      해당 레이어는 추후에 삭제가 될 예정이기 때문에 GAN 모듈에만 적합하게
    //      동작하는 수준으로만 checkShape 함수를 맞추도록 하겠다.
    
    TensorShape outputShape1;
    outputShape1.N = SNPROP(batchSize);
    outputShape1.C = 3;
    outputShape1.H = SLPROP(CelebAInput, resizedImageRow);
    outputShape1.W = SLPROP(CelebAInput, resizedImageCol);
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t CelebAInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class CelebAInputLayer<float>;
