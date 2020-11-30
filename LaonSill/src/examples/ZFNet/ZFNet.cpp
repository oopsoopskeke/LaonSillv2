/**
 * @file ZFNet.cpp
 * @date 2018-03-21
 * @author Jongha Lim
 * @brief 
 * @details Final target accuracy  of top-5 is, 0.835, top-1 is 0.669 on ilsvrc12_train_224px. 
 */

#include <vector>
#include <string>

#include "common.h"
#include "ZFNet.h"
#include "StdOutLog.h"
#include "Network.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"
#include "LiveDataInputLayer.h"
#include "ConvLayer.h"

using namespace std;

// #define EXAMPLE_ZFNET_TRAIN_NETWORK_FILEPATH	SPATH("examples/ZFNet/zfnet_train.json")
// #define EXAMPLE_ZFNET_TRAIN_NETWORK_FILEPATH	SPATH("examples/ZFNet/zfnet_union.json")
#define EXAMPLE_ZFNET_TRAIN_NETWORK_FILEPATH	SPATH("examples/ZFNet/zfnet_test_live.json")
#define EXAMPLE_ZFNET_VISUALIZER_NETWORK_FILEPATH	SPATH("examples/ZFNet/zfnet_test_deconv.json")

template<typename Dtype>
void ZFNet<Dtype>::run() {
#if 0   // HEIM : Deconv Debug Test,
    string networkID = PlanParser::loadNetwork(string(EXAMPLE_ZFNET_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(0);

	network->run(false);

#else
    // network 생성하여 빌드
    string networkID = PlanParser::loadNetwork(string(EXAMPLE_ZFNET_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(1);
    vector<Layer<Dtype>*> inputLayers = network->findLayersByType(Layer<Dtype>::LiveDataInput);
    LiveDataInputLayer<Dtype>* inputLayer = (LiveDataInputLayer<Dtype>*)inputLayers[0];
    Dtype* imageData;
    cv::Mat testImage = cv::imread("0.jpg");
    cv::resize(testImage, testImage, cv::Size(227,227));
    testImage.convertTo(testImage, CV_32FC3);
    imageData = (Dtype*)testImage.data;
    // Live input의 경우 image를 넣어줘야 함
    inputLayer->feedImage(3, 227, 227, imageData);
    // inference 로 실행
    // network->run(false);
    network->runMiniBatch(false, 0);

    // Data<Dtype>* inputTensor = (Data<Dtype>*)network->findTensor(0, 0, "data");
    // const Dtype* inputHostData = inputTensor->host_data();
    // ImageUtil<Dtype>::showImage(inputHostData, 0, 3, 227, 227);

    // tensor print
    // DebugUtil<Dtype>::printNetworkEdges(stdout, "original", networkID, 2);
    // network 에서 conv1 레이어를 찾아 host data 가져옴

    // conv1에 들어간 image 출력
    ConvLayer<Dtype>* convLayer = (ConvLayer<Dtype>*)network->findLayer("conv1");

    ImageUtil<Dtype>::showImage(convLayer->_inputData[0]->host_data(), 0, 3, 227, 227);

    // network에서 conv1 레이어의 tensor를 찾아 host_data를 가져옴
    Data<Dtype>* conv1Tensor = (Data<Dtype>*)network->findTensor(0, 0, "conv1");
    const Dtype* conv1HostData = conv1Tensor->host_data();
    // network visualizer 에서 input 레이어를 가져옴
    string networkVisualizerID = PlanParser::loadNetwork(string(EXAMPLE_ZFNET_VISUALIZER_NETWORK_FILEPATH));
    Network<Dtype>* networkVisualizer = Network<Dtype>::getNetworkFromID(networkVisualizerID);
    networkVisualizer->build(1);
    // InputLayer<Dtype>* InputLayer = (InputLayer<Dtype>*)networkVisualizer->findLayer("input");
    Data<Dtype>* dummyInput_tensor = (Data<Dtype>*)networkVisualizer->findTensor(0, 0, "dummyInput");
    dummyInput_tensor->set_host_data(conv1HostData);
    // 시각화를 위한 deconv 진행
    networkVisualizer->runMiniBatch(false, 0);
    // tensor print
    // DebugUtil<Dtype>::printNetworkEdges(stdout, "deconv", networkVisualizerID, 2);
    // deconv1 tensor에서 최종 이미지 저장
    //WorkContext::updateNetwork(networkVisualizerID);
    //WorkContext::updatePlan(0, false);
    
    ConvLayer<Dtype>* deconvLayer = (ConvLayer<Dtype>*)networkVisualizer->findLayer("deconv1");
    std::cout << "Deconv outputData channel : "<< deconvLayer->_outputData[0]->getShape(1) << std::endl;
    std::cout << "Deconv outputData height : "<< deconvLayer->_outputData[0]->getShape(2) << std::endl;
    std::cout << "Deconv outputData width : "<< deconvLayer->_outputData[0]->getShape(3) << std::endl;

    // Data<Dtype>* deconv1Tensor = (Data<Dtype>*)networkVisualizer->findTensor(0, 0, "conv1");
    // const Dtype* visualizeHostData = deconv1Tensor->host_data();
    ImageUtil<Dtype>::showImage(deconvLayer->_outputData[0]->host_data(), 0, 3, 227, 227);

#endif
}


template class ZFNet<float>;
