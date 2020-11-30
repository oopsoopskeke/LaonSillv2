/**
 * @file VOCPascalInputLayer.cpp
 * @date 2017-04-18
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
#include "VOCPascalInputLayer.h"
#include "InputLayer.h"
#include "Network.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

#define VOCPASCALINPUTLAYER_LOG        0

const int VOCPASCAL_IMAGE_CHANNEL   = 3;
const int VOCPASCAL_IMAGE_ROW       = 448;
const int VOCPASCAL_IMAGE_COL       = 448;

const int VOCPASCAL_BOX_COUNT       = 1;
const int VOCPASCAL_CLASS_COUNT     = 20;
const int VOCPASCAL_BOX_ELEM_COUNT  = (6 + VOCPASCAL_CLASS_COUNT);
/*********************************************************************************
 * Label
 * +-------+-------+---+---+-------+--------+-----------+
 * | gridX | gridY | x | y | width | height | class(20) |
 * +-------+-------+---+---+-------+--------+-----------+
 */

const int VOCPASCAL_GRID_COUNT      = 7;

template<typename Dtype>
VOCPascalInputLayer<Dtype>::VOCPascalInputLayer() : InputLayer<Dtype>() {
    this->type = Layer<Dtype>::VOCPascalInput;

    this->imageRow = VOCPASCAL_IMAGE_ROW;
    this->imageCol = VOCPASCAL_IMAGE_COL;

    if (SLPROP(VOCPascalInput, resizeImage)) {
        this->imageRow = SLPROP(VOCPascalInput, resizedImageRow);
        this->imageCol = SLPROP(VOCPascalInput, resizedImageCol);
    }

    this->imageChannel = VOCPASCAL_IMAGE_CHANNEL;

    this->images = NULL;
    this->labels = NULL;

    this->currentBatchIndex = 0;
}

template<typename Dtype>
VOCPascalInputLayer<Dtype>::~VOCPascalInputLayer() {
    if (this->images != NULL) {
        SFREE(this->images); 
    }

    if (this->labels != NULL) {
        SFREE(this->labels);
    }
}

template <typename Dtype>
void VOCPascalInputLayer<Dtype>::reshape() {
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
            (unsigned long)VOCPASCAL_BOX_COUNT *
            (unsigned long)VOCPASCAL_BOX_ELEM_COUNT *
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
    this->_inputShape[1][2] = VOCPASCAL_BOX_COUNT * VOCPASCAL_BOX_ELEM_COUNT;
    this->_inputShape[1][3] = 1;
    this->_inputData[1]->reshape(this->_inputShape[1]);

#if VOCPASCALINPUTLAYER_LOG
    printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batchSize, this->imageChannel, this->imageRow, this->imageCol);
    printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batchSize, 1, VOCPASCAL_BOX_COUNT * VOCPASCAL_BOX_ELEM_COUNT, 1);
#endif

    loadImages(this->currentBatchIndex);
    loadLabels(this->currentBatchIndex);

    int inputImageSize = this->imageChannel * this->imageRow * this->imageCol * batchSize;
    this->_inputData[0]->set_device_with_host_data(this->images, 0, inputImageSize);

    int inputLabelSize = VOCPASCAL_BOX_COUNT * VOCPASCAL_BOX_ELEM_COUNT * batchSize;
    this->_inputData[1]->set_device_with_host_data(this->labels, 0, inputLabelSize);
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::loadPixels(cv::Mat image, int imageIndex) {
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

#define VOCPASCAL_METAFILE_NAME        "pascal_voc.txt"

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::fillMetas() {
    string filePath = SLPROP(VOCPascalInput, imageDir) + "/" + VOCPASCAL_METAFILE_NAME;
    FILE *fp = fopen(filePath.c_str(), "r");
    SASSERT0(fp != NULL);

    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    char imagePath[1024];
    int boxCount = 0;
    int imageWidth;
    int imageHeight;
    VOCPascalMeta meta;

    int metaIndex = 0;

    while ((read = getline(&line, &len, fp)) != -1) {
        if (boxCount == 0) {
            int ret = sscanf(line, "%s %d", imagePath, &boxCount);
            SASSERT0(ret == 2);
            meta.imagePath = imagePath;

            cv::Mat image = cv::imread(meta.imagePath, CV_LOAD_IMAGE_COLOR);
            imageWidth = image.cols;
            imageHeight = image.rows;
        } else {
            int xmin, ymin, xmax, ymax, classID;
            int ret = sscanf(line, "%d %d %d %d %d", &xmin, &ymin, &xmax, &ymax, &classID);
            SASSERT0(ret == 5);

            float centerX = ((float)xmin + (float)xmax) / 2.0 / (float)imageWidth;
            float centerY = ((float)ymin + (float)ymax) / 2.0 / (float)imageHeight;

            meta.gridX = (int)(centerX * (float)VOCPASCAL_GRID_COUNT);
            SASSERT0((meta.gridX >= 0) && (meta.gridX < VOCPASCAL_GRID_COUNT));

            meta.gridY = (int)(centerY * (float)VOCPASCAL_GRID_COUNT);
            SASSERT0((meta.gridY >= 0) && (meta.gridY < VOCPASCAL_GRID_COUNT));

            meta.x = (centerX - (float)meta.gridX / (float)VOCPASCAL_GRID_COUNT) * 
                (float)VOCPASCAL_GRID_COUNT;
            meta.y = (centerY - (float)meta.gridY / (float)VOCPASCAL_GRID_COUNT) * 
                (float)VOCPASCAL_GRID_COUNT;

            SASSERT0(xmax > xmin);
            SASSERT0(ymax > ymin);

            meta.width = ((float)xmax - (float)xmin) / (float)imageWidth;
            meta.height = ((float)ymax - (float)ymin) / (float)imageHeight;

            meta.classID = classID;

            this->metas.push_back(meta);
            this->metaIndexes.push_back(metaIndex);
            metaIndex++;
            boxCount--;
        }
    }
   
    if (line)
        free(line);     // do not use SFREE because line is allocated in the getline() func

    fclose(fp);
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::loadImages(int baseIdx) {
    int batchSize = SNPROP(batchSize);

    for (int i = 0; i < batchSize; i++) {
        int index = i + baseIdx;
        if (index >= this->metas.size())
            break;

        int shuffledIndex = this->metaIndexes[index];
        string imagePath = this->metas[shuffledIndex].imagePath;

        cv::Mat image;
        image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

        // XXX: 좀더 general 하게 만들자.
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(this->imageRow, this->imageCol));

        loadPixels(resizedImage, i);
    }
}

#define EPSILON     0.001       // to solve converting issue int to float(double)

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::loadLabels(int baseIdx) {
    int batchSize = SNPROP(batchSize);

    for (int i = 0; i < batchSize; i++) {
        int index = i + baseIdx;
        if (index >= this->metas.size())
            break;

        int shuffledIndex = this->metaIndexes[index];
       
        VOCPascalMeta *meta = &this->metas[shuffledIndex];
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 0] = (Dtype)meta->gridX + EPSILON;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 1] = (Dtype)meta->gridY + EPSILON;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 2] = (Dtype)meta->x;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 3] = (Dtype)meta->y;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 4] = (Dtype)meta->width;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 5] = (Dtype)meta->height;
        
        for (int j = 0; j < VOCPASCAL_CLASS_COUNT; j++) {
            this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 6 + j] = 0.0;
        }

        SASSERT0(meta->classID < VOCPASCAL_CLASS_COUNT);
        SASSERT0(meta->classID >= 0);
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 6 + meta->classID] = 1.0;
    }
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::shuffleImages() {
    // FIXME: move it to other source.
#if 1
    srand(time(NULL)); 
    boost::range::random_shuffle(this->metaIndexes);
#endif
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    this->currentBatchIndex = baseIndex;    // FIXME: ...
    reshape();
}

template<typename Dtype>
int VOCPascalInputLayer<Dtype>::getNumTrainData() {
    if (this->images == NULL) {
        reshape();
    }
    return this->metas.size();
}

template<typename Dtype>
int VOCPascalInputLayer<Dtype>::getNumTestData() {
    return 0;
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->images == NULL) {
        reshape();
    }
    shuffleImages();
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* VOCPascalInputLayer<Dtype>::initLayer() {
	VOCPascalInputLayer* layer = NULL;
	SNEW(layer, VOCPascalInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    VOCPascalInputLayer<Dtype>* layer = (VOCPascalInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    VOCPascalInputLayer<Dtype>* layer = (VOCPascalInputLayer<Dtype>*)instancePtr;

    SASSERT0(!isInput);
    SASSERT0(index < 2);
    SASSERT0(layer->_outputData.size() == index);
    layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool VOCPascalInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    VOCPascalInputLayer<Dtype>* layer = (VOCPascalInputLayer<Dtype>*)instancePtr;
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
void VOCPascalInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    VOCPascalInputLayer<Dtype>* layer = (VOCPascalInputLayer<Dtype>*)instancePtr;
    layer->feedforward(miniBatchIdx);
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing..
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool VOCPascalInputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

    // XXX: 본 레이어를 사용하는 모듈이 없다.
    //      해당 레이어는 추후에 삭제가 될 예정이기 때문에 
    //      적당히 동작하는 수준으로만 checkShape 함수를 맞추도록 하겠다.
    
    int imageRow = VOCPASCAL_IMAGE_ROW;
    int imageCol = VOCPASCAL_IMAGE_COL;

    if (SLPROP(VOCPascalInput, resizeImage)) {
        imageRow = SLPROP(VOCPascalInput, resizedImageRow);
        imageCol = SLPROP(VOCPascalInput, resizedImageCol);
    }

    int imageChannel = VOCPASCAL_IMAGE_CHANNEL;

    TensorShape outputShape1;
    outputShape1.N = SNPROP(batchSize);
    outputShape1.C = imageChannel;
    outputShape1.H = imageRow;
    outputShape1.W = imageCol;
    outputShape.push_back(outputShape1);

    TensorShape outputShape2;
    outputShape2.N = SNPROP(batchSize);
    outputShape2.C = 1;
    outputShape2.H = VOCPASCAL_BOX_COUNT * VOCPASCAL_BOX_ELEM_COUNT;
    outputShape2.W = 1;
    outputShape.push_back(outputShape2);

    return true;
}

template<typename Dtype>
uint64_t VOCPascalInputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class VOCPascalInputLayer<float>;
