/**
 * @file KISTIKeyword.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "common.h"
#include "Debug.h"
#include "KISTIKeyword.h"
#include "StdOutLog.h"
#include "Network.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"
#include "MultiLabelDataInputLayer.h"

using namespace std;

#define USE_VOCPASCAL                   1

#ifdef USE_VOCPASCAL
#define EXAMPLE_KISTIKEYWORD_NETDEFPATH   SPATH("examples/KISTIKeyword/networkPASCALVOC.json")
#else
#if 0
#define EXAMPLE_KISTIKEYWORD_NETDEFPATH   SPATH("examples/KISTIKeyword/network.json")
#else
#define EXAMPLE_KISTIKEYWORD_NETDEFPATH   SPATH("examples/KISTIKeyword/networkESP.json")
#endif
#endif

template<typename Dtype>
float KISTIKeyword<Dtype>::getTopKAvgPrecision(int topK, const float* data,
    const float* label, int batchCount, int depth) {

    float mAP = 0.0;

    for (int i = 0; i < batchCount; i++) {
        vector<int> curLabel;
        vector<top10Sort> tempData;

        for (int j = 0; j < depth; j++) {
            int index = i * depth + j;

            if (label[index] > 0.99) {
                curLabel.push_back(j);
            }

            tempData.push_back({data[index], j});
        }

        sort(tempData.begin(), tempData.end());

        float sumAP = 0.0;
        int truePositiveCnt = 0;

        for (int j = 0; j < topK; j++) {
            int reverseIndex = depth - 1 - j;
            int target = tempData[reverseIndex].index;

            for (int k = 0; k < curLabel.size(); k++) {
                if (curLabel[k] == target) {
                    truePositiveCnt++;

                    sumAP += (float)truePositiveCnt / (float)(j + 1);
                    break;
                }
            }
        }

        SASSUME0(curLabel.size() > 0);
        sumAP = sumAP / (float)(curLabel.size());
        mAP += sumAP;
    }

    mAP = mAP / (float)batchCount;
    return mAP;
}

// XXX: inefficient..
template<typename Dtype>
int KISTIKeyword<Dtype>::getTop10GuessSuccessCount(const float* data,
    const float* label, int batchCount, int depth, bool train, int epoch, 
    const float* image, int imageBaseIndex, vector<KistiData> etriData) {

    int successCnt = 0;

#if 0
    string folderName;
        if (train) {
            folderName = "train_" + to_string(epoch) + "_" + to_string(imageBaseIndex); 
        } else {
            folderName = "test_" + to_string(epoch) + "_" + to_string(imageBaseIndex); 
        }
        
        ImageUtil<float>::saveImage(image, batchCount, 3, 224, 224, folderName);
#endif

    for (int i = 0; i < batchCount; i++) {
        vector<int> curLabel;
        vector<top10Sort> tempData;

        for (int j = 0; j < depth; j++) {
            int index = i * depth + j;

            if (label[index] > 0.99) {
                curLabel.push_back(j);
            }

            tempData.push_back({data[index], j});
        }

        sort(tempData.begin(), tempData.end());

        bool found = false;
        for (int j = 0; j < 10; j++) {
            int reverseIndex = depth - 1 - j;
            int target = tempData[reverseIndex].index;

            for (int k = 0; k < curLabel.size(); k++) {
                if (curLabel[k] == target) {
                    found = true;
                    break;
                }
            }

            if (found)
                break;
        }

#if 0
        printf ("Labels[%d] : ", i);
        for (int j = 0; j < curLabel.size(); j++) {
            printf(" %d", curLabel[j]);
        }
        printf ("\n");

        printf ("top 10 data[%d] : ", i);
        for (int j = 0; j < 10; j++) {
            int reverseIndex = depth - 1 - j;
            int target = tempData[reverseIndex].index;
            printf(" %d", target);
        }
        printf("\n");

        int imageIndex = i + imageBaseIndex;
        cout << "[folder:" << folderName << "] : " << etriData[imageIndex].filePath <<
            ", labels : ";
        for (int k = 0; k < etriData[imageIndex].labels.size(); k++) {
            cout << etriData[imageIndex].labels[k] << " ";
        }
        cout << endl;
#endif

        if (found)
            successCnt++;
    }

#if 0
    for (int i = 0; i < batchCount; i++) {
        printf("Labels[%d] : ", i);
        for (int j = 0; j < depth; j++) {
            int index = i * depth + j;
            printf("%d ", label[index]);
        }
        printf("\n");
        printf("data[%d] : ", i);
        for (int j = 0; j < depth; j++) {
            int index = i * depth + j;
            printf("%d ", data[index]);
        }
        printf("\n");
    }
#endif

    return successCnt;
}

#ifdef USE_VOCPASCAL
template<typename Dtype>
void KISTIKeyword<Dtype>::run() {
    string networkID = PlanParser::loadNetwork(EXAMPLE_KISTIKEYWORD_NETDEFPATH);
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(1);

    for (int epoch = 0; epoch < 50; epoch++) {
        STDOUT_BLOCK(cout << "epoch #" << epoch << " starts" << endl;); 

        MultiLabelDataInputLayer<Dtype>* inputLayer = 
            (MultiLabelDataInputLayer<Dtype>*)network->findLayer("data");
        CrossEntropyWithLossLayer<Dtype>* lossLayer = 
            (CrossEntropyWithLossLayer<Dtype>*)network->findLayer("loss");

        const uint32_t trainDataSize = inputLayer->getNumTrainData();
        const uint32_t numTrainBatches = trainDataSize / SNPROP(batchSize) - 1;

        for (int i = 0; i < numTrainBatches; i++) {
            STDOUT_BLOCK(cout << "train data(" << i << "/" << numTrainBatches << ")" <<
                endl;);
            network->runMiniBatch(false, i);
        }

        STDOUT_BLOCK(cout << "evaluate train data(num train batches =" << numTrainBatches <<
            ")" << endl;);

        float trainLoss = 0.0;
        float mAP = 0.0;
        for (int i = 0; i < numTrainBatches; i++) {
            network->runMiniBatch(true, i);

            const Dtype* outputData = lossLayer->_inputData[0]->host_data();
            //const Dtype* outputLabel = lossLayer->_inputData[1]->host_data();
            const Dtype* outputLabel = inputLayer->_outputData[1]->host_data();

            trainLoss += lossLayer->cost();
            mAP += getTopKAvgPrecision(3, outputData, outputLabel, SNPROP(batchSize), 20);
        }
        trainLoss = trainLoss / (float)(numTrainBatches);
        mAP = mAP / (float)(numTrainBatches);

        STDOUT_BLOCK(cout << "[RESULT #" << epoch << "] train loss : " << trainLoss <<
            ", mAP : " << mAP << endl;);
    }
}
#else
template<typename Dtype>
void KISTIKeyword<Dtype>::run() {
    string networkID = PlanParser::loadNetwork(EXAMPLE_KISTIKEYWORD_NETDEFPATH);
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(1);

    // (3) 학습한다.
    for (int epoch = 0; epoch < 50; epoch++) {
        STDOUT_BLOCK(cout << "epoch #" << epoch << " starts" << endl;); 

        KistiInputLayer<Dtype>* etriInputLayer = 
            (KistiInputLayer<Dtype>*)network->findLayer("data");
        CrossEntropyWithLossLayer<Dtype>* lossLayer = 
            (CrossEntropyWithLossLayer<Dtype>*)network->findLayer("loss");

        const uint32_t trainDataSize = etriInputLayer->getNumTrainData();
        const uint32_t numTrainBatches = trainDataSize / SNPROP(batchSize) - 1;

        // (3-1) 네트워크를 학습한다.
        for (int i = 0; i < numTrainBatches; i++) {
            STDOUT_BLOCK(cout << "train data(" << i << "/" << numTrainBatches << ")" <<
                endl;);
            network->runMiniBatch(false, i);
        }

        // (3-2) 트레이닝 데이터에 대한 평균 Loss와 정확도를 구한다.
        STDOUT_BLOCK(cout << "evaluate train data(num train batches =" << numTrainBatches <<
            ")" << endl;);
        float trainLoss = 0.0;
        int trainSuccessCnt = 0;
        for (int i = 0; i < numTrainBatches; i++) {
            network->runMiniBatch(true, i);
            trainLoss += lossLayer->cost();

            const Dtype* inputData = etriInputLayer->_inputData[0]->host_data();
            const Dtype* outputData = lossLayer->_inputData[0]->host_data();
            const Dtype* outputLabel = lossLayer->_inputData[1]->host_data();
            trainSuccessCnt += getTop10GuessSuccessCount(outputData, outputLabel,
                SNPROP(batchSize), 22569, true, epoch, inputData,
                (int)(SNPROP(batchSize) * i), etriInputLayer->trainData);
        }
        trainLoss = trainLoss / (float)(numTrainBatches);

        // (3-3) 테스트 데이터에 대한 평균 Loss와 정확도를 구한다.
        etriInputLayer->setTrain(false);

        const uint32_t testDataSize = etriInputLayer->getNumTestData();
        const uint32_t numTestBatches = testDataSize / SNPROP(batchSize) - 1;

        STDOUT_BLOCK(cout << "evaluate test data(num test batches =" << numTestBatches <<
            ")" << endl;);
        float testLoss = 0.0;
        int testSuccessCnt = 0;
        for (int i = 0; i < numTestBatches; i++) {
            network->runMiniBatch(true, i);
            testLoss += lossLayer->cost();

            const Dtype* inputData = etriInputLayer->_inputData[0]->host_data();
            const Dtype* outputData = lossLayer->_inputData[0]->host_data();
            const Dtype* outputLabel = lossLayer->_inputData[1]->host_data();
            testSuccessCnt += getTop10GuessSuccessCount(outputData, outputLabel,
                SNPROP(batchSize), 22569, false, epoch, inputData,
                (int)(SNPROP(batchSize) * i), etriInputLayer->testData);
        }
        testLoss = testLoss / (float)(numTestBatches);

        etriInputLayer->setTrain(true);

        float trainAcc = (float)trainSuccessCnt / (float)numTrainBatches /
            (float)SNPROP(batchSize);
        float testAcc = (float)testSuccessCnt / (float)numTestBatches /
            (float)SNPROP(batchSize);
        STDOUT_BLOCK(cout << "[RESULT #" << epoch << "] train loss : " << trainLoss <<
            ", test losss : " << testLoss << ", train accuracy : " << trainAcc << "(" <<
            trainSuccessCnt << "/" << numTrainBatches * SNPROP(batchSize) <<
            "), test accuracy : " << testAcc << "(" << testSuccessCnt << "/" <<
            numTestBatches * SNPROP(batchSize) << ")" << endl;);
    }
}
#endif

template class KISTIKeyword<float>;
