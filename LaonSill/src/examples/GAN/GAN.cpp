/**
 * @file GAN.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>
#include <string>

#include "common.h"
#include "Debug.h"
#include "GAN.h"
#include "StdOutLog.h"
#include "Network.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

#include "DebugUtil.h"

using namespace std;

template<typename Dtype>
void GAN<Dtype>::setLayerTrain(Network<Dtype>* network, bool train) {
    vector<Layer<Dtype>*> layers = network->findLayersByType((int)Layer<Dtype>::BatchNorm);

    for (int i = 0; i < layers.size(); i++) {
        BatchNormLayer<Dtype>* bnLayer = dynamic_cast<BatchNormLayer<Dtype>*>(layers[i]);
        SASSUME0(bnLayer != NULL);

        bnLayer->setTrain(train);
    }
}

#define EXAMPLE_GAN_NETWORKG0_FILEPATH              SPATH("examples/GAN/networkG0.json")
#define EXAMPLE_GAN_NETWORKD_FILEPATH               SPATH("examples/GAN/networkD.json")
#define EXAMPLE_GAN_NETWORKG1_FILEPATH              SPATH("examples/GAN/networkG1.json")

template<typename Dtype>
void GAN<Dtype>::run() {
    string networkID = PlanParser::loadNetwork(string(EXAMPLE_GAN_NETWORKG0_FILEPATH));
    Network<Dtype>* networkG0 = Network<Dtype>::getNetworkFromID(networkID);
    networkG0->build(1);

    networkID = PlanParser::loadNetwork(string(EXAMPLE_GAN_NETWORKD_FILEPATH));
    Network<Dtype>* networkD = Network<Dtype>::getNetworkFromID(networkID);
    networkD->build(1);
    CelebAInputLayer<Dtype>* inputLayer =
        (CelebAInputLayer<Dtype>*)networkD->findLayer("celebAInput");
    int miniBatchCount = inputLayer->getNumTrainData() / SNPROP(batchSize) - 1;

    networkID = PlanParser::loadNetwork(string(EXAMPLE_GAN_NETWORKG1_FILEPATH));
    Network<Dtype>* networkG1 = Network<Dtype>::getNetworkFromID(networkID);
    networkG1->build(1);

    for (int i = 0; i < 10000; i++) {
        cout << "epoch : " << i << endl;
        for (int j = 0; j < miniBatchCount; j++) {

            networkG0->runMiniBatch(false, 0);
            networkD->runMiniBatch(false, j);
            networkG1->runMiniBatch(false, 0);
            networkG1->runMiniBatch(false, 0);

            if (j % 100 == 0)
                cout << "minibatch " << j << " is done." << endl;
        }

        setLayerTrain(networkG1, false);
        networkG1->runMiniBatch(true, 0);

        ConvLayer<Dtype>* convLayer = (ConvLayer<Dtype>*)networkG1->findLayer("conv1");
        const Dtype* host_data = convLayer->_inputData[0]->host_data();
        ImageUtil<Dtype>::saveImage(host_data, 64, 3, 64, 64, "");

        setLayerTrain(networkG1, true);
    }
}

template class GAN<float>;
