#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/variant.hpp>
#include <boost/serialization/map.hpp>
#include <string>

#include "LayerTestInterface.h"
#include "LayerTest.h"
#include "LearnableLayerTest.h"

#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "ReluLayer.h"
#include "PoolingLayer.h"
#include "SoftmaxWithLossLayer.h"
#include "DataInputLayer.h"
#include "MultiLabelDataInputLayer.h"

#include "NetworkTestInterface.h"
#include "NetworkTest.h"

#include "TestUtil.h"
#include "Debug.h"

#include "PlanParser.h"
#include "WorkContext.h"
#include "PhysicalPlan.h"
#include "LearnableLayer.h"
#include "PropMgmt.h"
#include "InitParam.h"
#include "Perf.h"
#include "ColdLog.h"
#include "Broker.h"
#include "ResourceManager.h"
#include "PlanOptimizer.h"
#include "LayerFunc.h"
#include "gpu_nms.hpp"
#include "cnpy.h"
#include "ParamManipulator.h"
#include "Datum.h"
#include "SDF.h"
#include "IO.h"
#include "DataReader.h"
#include "ssd_common.h"
#include "Tools.h"
#include "StdOutLog.h"
#include "DataTransformer.h"
#include "LayerPropParam.h"
//#include "jsoncpp/json/json.h"
#include "MathFunctions.h"
#include "ImTransforms.h"
#include "AnnotatedDataLayer.h"
#include "MemoryMgmt.h"
#include "MeasureManager.h"

#include <execinfo.h>



using namespace std;
namespace fs = ::boost::filesystem;






void plainTest(int argc, char** argv);

void dataReaderTest(int argc, char** argv);
void annoDataReaderTest(int argc, char** argv);
void dataReaderMemoryLeakTest();
void runNetwork();
void jsonTest();
void dataTransformerTest();
void imTransformTest();
void readAnnoDataSetTest();
void randTest(int argc, char** argv);
void signalTest();
void convertAnnoSetTest();
void convertImageSetTest();
void convertMnistDataTest();
void convertVOC07SetTest();

void layerTest(int argc, char** argv);
void layerTest2(int argc, char** argv);
void networkTest(int argc, char** argv);
void saveNetwork();



const string LAONSILL_HOME = string(std::getenv("LAONSILL_HOME"));
const string HOME = string(std::getenv("HOME"));


void testJsonType(Json::Value& value) {
	cout << value.type() << endl;
}





#if 0
int main(int argc, char** argv) {
	cout << "begin test ... " << endl;
	cout.precision(8);
	cout.setf(ios::fixed);
	//cout.setf(ios::scientific);

	//SDFHeader header = SDF::retrieveSDFHeader(LAONSILL_HOME + "/data/sdf/version_test_imageset");
	//header.print();

	plainTest(argc, argv);
	//layerTest(argc, argv);
	//networkTest(argc, argv);
	//saveNetwork();
	//layerTest2(argc, argv);

	cout << "end test ... " << endl;
	return 0;
}
#endif




void jsonTest() {
	const string filePath = "/home/jkim/Dev/git/soooa/SoooA/src/examples/SSD/ssd_multiboxloss_test.json";

	filebuf fb;
	if (fb.open(filePath.c_str(), ios::in) == NULL) {
		SASSERT(false, "cannot open cluster configuration file. file path=%s",
			filePath.c_str());
	}

	Json::Value rootValue;
	istream is(&fb);
	Json::Reader reader;
	bool parse = reader.parse(is, rootValue);

	if (!parse) {
		SASSERT(false, "invalid json-format file. file path=%s. error message=%s",
			filePath.c_str(), reader.getFormattedErrorMessages().c_str());
	}



	stringstream softmaxDef;
	softmaxDef << "{\n";
	softmaxDef << "\t\"name\" : \"inner_softmax\",\n";
	softmaxDef << "\t\"id\" : 7001,\n";
	softmaxDef << "\t\"layer\" : \"Softmax\",\n";
	softmaxDef << "\t\"input\" : [\"inner_softmax_7001_input\"],\n";
	softmaxDef << "\t\"output\" : [\"inner_softmax_7001_output\"],\n";
	softmaxDef << "\t\"softmaxAxis\" : 2\n";
	softmaxDef << "}\n";

	cout << softmaxDef.str() << endl;
	Json::Value tempValue;
	reader.parse(softmaxDef, tempValue);

	cout << tempValue["output"] << endl;
	testJsonType(tempValue);

}


void initializeNetwork() {
	WorkContext::curBootMode = BootMode::DeveloperMode;

	MemoryMgmt::init();
	InitParam::init();
	Perf::init();
	SysLog::init();
	ColdLog::init();
	Job::init();
	Task::init();
	Broker::init();
	Network<float>::init();
	MeasureManager::init();

	ResourceManager::init();
	PlanOptimizer::init();
	LayerFunc::init();
	LayerPropList::init();



	Util::setOutstream(&cout);
	Util::setPrint(false);
}


void plainTest(int argc, char** argv) {
	//denormalizeTest(argc, argv);
	//dataReaderTest(argc, argv);
	//computeImageMean(argc, argv);
	//runNetwork();
	//dataReaderMemoryLeakTest();
	//dataTransformerTest();
	//imTransformTest();
	//annoDataReaderTest(argc, argv);
	//randTest(argc, argv);
	//signalTest();
	convertMnistDataTest();
	//convertImageSetTest();
	//convertAnnoSetTest(argc, argv);
	//convertVOC07SetTest();
}


void signalTest() {
	void *trace[16];
	char **messages = (char **) NULL;
	int i, trace_size = 0;

	trace_size = backtrace(trace, 16);
	/* overwrite sigaction with caller's address */
	messages = backtrace_symbols(trace, trace_size);
	/* skip first stack frame (points here) */
	printf("[bt] Execution path:\n");
	for (i = 1; i < trace_size; ++i) {
		printf("[bt] #%d %s\n", i, messages[i]);

		/* find first occurence of '(' or ' ' in message[i] and assume
		 * everything before that is the file name. (Don't go beyond 0 though
		 * (string terminator)*/
		size_t p = 0;
		while (messages[i][p] != '(' && messages[i][p] != ' '
				&& messages[i][p] != 0)
			++p;

		char syscom[256];
		sprintf(syscom, "addr2line %p -e %.*s", trace[i], p, messages[i]);
		//last parameter is the file name of the symbol
		system(syscom);
	}
}



void randTest(int argc, char** argv) {

	for (int i = 0; i < 100; i++) {
		soooa_rng_rand();
	}

	//float r;
	//soooa_rng_uniform(1, 0.f, 1.f, &r);
}


void imTransformTest() {
	const string windowName = "result";

	DistortionParam distortParam;
	//distortParam.brightnessProb = 1.f;
	//distortParam.brightnessDelta = 100.f;
	//distortParam.contrastProb = 1.f;
	//distortParam.contrastLower = 0.5f;
	//distortParam.contrastUpper = 2.0f;
	//distortParam.saturationProb = 1.f;
	//distortParam.saturationLower = 0.5f;
	//distortParam.saturationUpper = 2.0f;
	//distortParam.hueProb = 1.0f;
	//distortParam.hueDelta = 100.0f;
	distortParam.randomOrderProb = 1.0f;


	for (int i  = 0; i < 100; i++) {
		cv::Mat im = cv::imread("/home/jkim/Downloads/sample.jpg");
		const int height = im.rows;
		const int width = im.cols;
		const int channels = im.channels();

		im = ApplyDistort(im, distortParam);

		cv::imshow("result", im);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}


void dataTransformerTest() {
	const string windowName = "result";
	DataTransformParam dtp;
	//dtp.mirror = flase;

	DistortionParam distortParam;
	distortParam.brightnessProb = 1.f;
	distortParam.brightnessDelta = 100.f;

	dtp.distortParam = distortParam;


	DataTransformer<float> dt(&dtp);

	cv::Mat im = cv::imread("/home/jkim/Downloads/sample.jpg");
	const int height = im.rows;
	const int width = im.cols;
	const int channels = im.channels();

	//im.convertTo(im, CV_32FC3);
	//cv::resize(im, im, cv::Size(300, 300), 0, 0, CV_INTER_LINEAR);

	Data<float> data("data", true);
	data.reshape({1, (uint32_t)channels, (uint32_t)height, (uint32_t)width});

	dt.transform(im, &data, 0);


	Data<float> temp("temp");
	temp.reshapeLike(&data);
	const int singleImageSize = data.getCount();
	cv::Mat result;
	transformInv(1, singleImageSize, height, width, height, width, {0.f, 0.f, 0.f},
			data.host_data(), temp, result);
}




void dataReaderTest(int argc, char** argv) {
	DataReader<Datum> dr("/home/jkim/Dev/SOOOA_HOME/data/sdf/plantynet_train_0.25/");
	//DataReader<Datum> dr("/home/jkim/Dev/SOOOA_HOME/data/sdf/test_train/");
	int numData = dr.getNumData();
	cout << "numData: " << numData << endl;

	const string windowName = "result";
	cv::namedWindow(windowName);

	for (int i = 0; i < std::min(numData, 100); i++) {
		Datum* datum = dr.getNextData();
		cout << i << " label: " << datum->label;
		if (datum->float_data.size() > 0) {
			for (int j = 0; j < datum->float_data.size(); j++) {
				cout << "," << (int)datum->float_data[j];
			}
		}
		cout << endl;
		datum->print();
		//PrintDatumData(datum, false);

		cv::Mat cv_img = DecodeDatumToCVMat(datum, true, true);
		//PrintCVMatData(cv_img);

		cv::imshow(windowName, cv_img);
		cv::waitKey(0);
	}
	cv::destroyWindow(windowName);
}


void annoDataReaderTest(int argc, char** argv) {
	DataReader<AnnotatedDatum> dr("/home/jkim/Dev/SOOOA_HOME/data/sdf/voc2007_train_sdf/");
	int numData = dr.getNumData();
	cout << "numData: " << numData << endl;

	const string windowName = "result";
	cv::namedWindow(windowName);

	for (int i = 0; i < std::min(numData, 100); i++) {
		AnnotatedDatum* datum = dr.getNextData();
		cout << i << " label: " << datum->label;
		if (datum->float_data.size() > 0) {
			for (int j = 0; j < datum->float_data.size(); j++) {
				cout << "," << (int)datum->float_data[j];
			}
		}
		cout << endl;
		datum->print();
		//PrintDatumData(datum, false);

		cv::Mat cv_img = DecodeDatumToCVMat(datum, true, true);
		//PrintCVMatData(cv_img);

		cv::imshow(windowName, cv_img);
		cv::waitKey(0);
	}
	cv::destroyWindow(windowName);
}

void dataReaderMemoryLeakTest() {
	const string source = "/home/jkim/Dev/SOOOA_HOME/data/sdf/plantynet_train_0.25/";
	int cnt = 0;

	/*
	DataReader<Datum> dr(source);
	int numData = dr.getNumData();
	cout << "numData: " << numData << endl;

	while (true) {
		Datum* datum = dr.getNextData();
		SDELETE(datum);
		if (++cnt % 1000 == 0) {
			STDOUT_LOG("Processed %d images.", cnt);
		}
	}
	*/

	SDF db(source, Mode::READ);
	db.open();

	string value = db.getNextValue();
	int numData = atoi(value.c_str());

	while (true) {
		string value = db.getNextValue();
		Datum* datum = NULL;
		SNEW(datum, Datum);
		SASSUME0(datum != NULL);
		deserializeFromString(value, datum);
		SDELETE(datum);
		if (++cnt % 1000 == 0) {
			STDOUT_LOG("Processed %d images.", cnt);
		}
	}
}

bool readKeywords(const string& keywordPath, vector<string>& keywordList) {
	if (keywordPath.empty()) {
		return false;
	}

	ifstream infile(keywordPath);
	string line;
	keywordList.clear();

	while (std::getline(infile, line)) {
		keywordList.push_back(line);
	}

	if (keywordList.size() < 1) {
		return false;
	} else {
		return true;
	}
}

void runNetwork() {
	cout << "runNetwork ... " << endl;

	int gpuid = 0;
	initializeNetwork();
	NetworkTestInterface<float>::globalSetUp(gpuid);

	string networkID = PlanParser::loadNetwork("/home/jkim/Dev/SOOOA_HOME/network_def/data_input_network.json");
	const string keywordPath = "/home/jkim/Dev/data/image/ESP-ImageSet/keywordList.txt";
	vector<string> keywordList;
	bool hasKeyword = readKeywords(keywordPath, keywordList);


	Network<float>* network = Network<float>::getNetworkFromID(networkID);
	network->build(100);

	WorkContext::updateNetwork(networkID);
	WorkContext::updatePlan(0, true);

	PhysicalPlan* pp = WorkContext::curPhysicalPlan;


	Layer<float>* inputLayer = network->findLayer("data");
	//SASSERT0(dynamic_cast<DataInputLayer<float>*>(inputLayer));
	SASSERT0(dynamic_cast<MultiLabelDataInputLayer<float>*>(inputLayer));


	const string windowName = "result";
	cv::namedWindow(windowName);

	for (int i = 0; i < 100; i++) {
		cout << i << "th iteration" << endl;
		network->runPlanType(PlanType::PLANTYPE_FORWARD, true);
		network->reset();

		// has label ...
		if (inputLayer->_outputData.size() > 1) {
			Data<float>* label = inputLayer->_outputData[1];
			const int numLabels = label->getShape(1);
			// single label
			if (numLabels == 1) {
				int labelIdx = (int)label->host_data()[i];
				if (hasKeyword) {
					cout << "label: " << labelIdx << " (" << keywordList[labelIdx] << ")" << endl;
				} else {
					cout << "label: " << labelIdx << endl;
				}
			}
			else if (numLabels > 1) {
				cout << "---------" << endl;
				for (int i = 0; i < numLabels; i++) {
					if ((int)label->host_data()[i] > 0) {
						if (hasKeyword) {
							cout << "label: " << i << " (" << keywordList[i] << ")" << endl;
						} else {
							cout << "label: " << i << endl;
						}
					}
				}
				cout << "---------" << endl;
			}
		}

		Data<float>* data = inputLayer->_outputData[0];
		data->print_shape();

		int height = data->getShape(2);
		int width = data->getShape(3);
		int channels = data->getShape(1);

		if (channels == 1) {
			cv::Mat cv_img(height, width, CV_32F, data->mutable_host_data());
			cv_img.convertTo(cv_img, CV_8U);
			cv::imshow(windowName, cv_img);
			cv::waitKey(0);
		} else if (channels = 3) {
			data->transpose({0, 2, 3, 1});
			cv::Mat cv_img(height, width, CV_32FC3, data->mutable_host_data());
			cv_img.convertTo(cv_img, CV_8UC3);
			cv::imshow(windowName, cv_img);
			data->transpose({0, 3, 1, 2});
			cv::waitKey(0);
		}
	}

	cv::destroyWindow(windowName);
	NetworkTestInterface<float>::globalCleanUp();

}



void layerTest(int argc, char** argv) {
	initializeNetwork();

	const string laonsillHome = string(std::getenv("LAONSILL_DEV_HOME"));

	const int gpuid = 0;
	const string networkName = "inception_v3";
	const string networkFilePath = laonsillHome +
			string("/src/examples/Inception/batchnorm_test.json");
	const string targetLayerName = "conv1/bn";

	const int numAfterSteps = 0;
	const int numSteps = 1;
	const NetworkStatus status = NetworkStatus::Train;

	LayerTestInterface<float>::globalSetUp(gpuid);
	LayerTestInterface<float>* layerTest = NULL;
	SNEW(layerTest, LayerTest<float>, networkFilePath,
			networkName, targetLayerName, numSteps, numAfterSteps, status);
	SASSUME0(layerTest != NULL);
	cout << "LayerTest initialized ... " << endl;

	layerTest->setUp();
	cout << "setUp completed ... " << endl;

	layerTest->forwardTest();

	cout << "forwardTest completed ... " << endl;
	if (status == NetworkStatus::Train) {
		layerTest->backwardTest();
	}

	layerTest->cleanUp();
	LayerTestInterface<float>::globalCleanUp();
}

void layerTest2(int argc, char** argv) {
	string laonsillHome = string(std::getenv("LAONSILL_DEV_HOME"));
	std::ifstream layerDefJsonStream(laonsillHome + string("/src/test/annotated_data.json"));
	SASSERT0(layerDefJsonStream);
	std::stringstream annotateddataDef;
	annotateddataDef << layerDefJsonStream.rdbuf();
	layerDefJsonStream.close();
	cout << annotateddataDef.str() << endl;

	_AnnotatedDataPropLayer* prop = NULL;
	SNEW(prop, _AnnotatedDataPropLayer);
	SASSUME0(prop != NULL);

	Json::Reader reader;
	Json::Value layer;
	reader.parse(annotateddataDef, layer);

	vector<string> keys = layer.getMemberNames();
	string layerType = layer["layer"].asCString();

	for (int j = 0; j < keys.size(); j++) {
		string key = keys[j];
		Json::Value val = layer[key.c_str()];
		if (strcmp(key.c_str(), "layer") == 0) continue;
		if (strcmp(key.c_str(), "innerLayer") == 0) continue;

		PlanParser::setPropValue(val, true, layerType, key,  (void*)prop);
	}

	AnnotatedDataLayer<float>* l = NULL;
	SNEW(l, AnnotatedDataLayer<float>, prop);
	SASSUME0(l != NULL);

	Data<float> data("data");
	Data<float> label("label");
	l->_outputData.push_back(&data);
	l->_outputData.push_back(&label);

	l->reshape();
	l->feedforward();
}



#define NETWORK_LENET		0
#define NETWORK_VGG19		1
#define NETWORK_FRCNN		2
#define NETWORK_FRCNN_TEST	3
#define NETWORK_SSD			4
#define NETWORK_SSD_TEST	5
#define NETWORK_VGG16		6
#define NETWORK_INCEPTION_V3	7
#define NETWORK_RESNET		8
#define NETWORK				NETWORK_INCEPTION_V3

#define EXAMPLE_LENET_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/LeNet/lenet_train_test.json")
#define EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/frcnn/frcnn_train_test.json")
#define EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/VGG16/vgg16_train_test.json")
#define EXAMPLE_DUMMY_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/SOOOA_HOME/network_def/data_input_network.json")


//void saveNetworkParams(LayersConfig<float>* layersConfig);







void networkTest(int argc, char** argv) {
	const string laonsillHome = std::getenv("LAONSILL_DEV_HOME");
	const int gpuid = 0;
	initializeNetwork();

#if NETWORK == NETWORK_SSD
	//const string networkFilePath = laonsillHome + string("/src/examples/SSD/ssd_300_train_test.json");
	//const string networkFilePath = laonsillHome + string("/src/examples/SSD/ssd_300_train.json");
	//const string networkFilePath = laonsillHome + string("/src/examples/SSD/ssd_300_infer_test.json");
	//const string networkFilePath = laonsillHome + string("/src/examples/SSD/ssd_512_train_test.json");
	const string networkFilePath = laonsillHome + string("/src/examples/SSD/ssd_512_infer_test.json");
	const string networkName = "ssd";
	const int numSteps = 1;
	const vector<string> forwardGTLayers = {"mbox_loss"};
	//const NetworkStatus status = NetworkStatus::Train;
	const NetworkStatus status = NetworkStatus::Test;

#elif NETWORK == NETWORK_VGG16

#elif NETWORK == NETWORK_INCEPTION_V3
	const string networkFilePath = laonsillHome + string("/src/examples/Inception/"
			//"inception_v3_train_nvidia_test_t3.json");
			//"batchnorm.test.json");
			//"inception_v3_train_nvidia.json");
			//"deploy_inception_v3.json");
			//"scale_test.json");
			"deploy_inception_v3_caffe_test.json");
	const string networkName = "inception_v3";
	const int numSteps = 1;
	const vector<string> forwardGTLayers = {};
	const NetworkStatus status = NetworkStatus::Train;
	//const NetworkStatus status = NetworkStatus::Test;


#elif NETWORK == NETWORK_RESNET
	const string networkFilePath = laonsillHome + string("/src/examples/ResNet/"
			//"inception_v3_train_nvidia_test_t3.json");
			//"batchnorm.test.json");
			//"resnet18.test.json");
			"resnet50.test.json");
	const string networkName = "resnet";
	const int numSteps = 1;
	const vector<string> forwardGTLayers = {};
	// backward 과정에서 데이터값을 확인할 레이어 목록
	const vector<string> backwardPeekLayers = {};
	const NetworkStatus status = NetworkStatus::Train;
	//const NetworkStatus status = NetworkStatus::Test;

#else
	cout << "invalid network ... " << endl;
	exit(1);
#endif
	NetworkTestInterface<float>::globalSetUp(gpuid);

	NetworkTest<float>* networkTest = NULL;
	SNEW(networkTest, NetworkTest<float>, networkFilePath, networkName, numSteps, status,
			NULL, NULL);
	SASSUME0(networkTest != NULL);

	networkTest->setUp();
	networkTest->updateTest();
	//Network<float>* network = networkTest->network;
	//network->save("/home/jkim/Dev/LAONSILL_HOME/param/INCEPTION_V3_CAFFE_TRAINED.param");
	networkTest->cleanUp();

	NetworkTestInterface<float>::globalCleanUp();
	cout << "networkTest() Done ... " << endl;
}



void saveNetwork() {
	const int gpuid = 0;
	initializeNetwork();

	const string networkFilePath = string("/home/jkim/Dev/git/soooa/SoooA/src/examples/VGG16/"
			"vgg16_train.json");
	const string networkName = "vgg16";
	const int numSteps = 0;
	const vector<string> forwardGTLayers = {};
	const NetworkStatus status = NetworkStatus::Train;

#if 0
	Network<float>* network = NULL;
	SNEW(network, Network<float>, networkConfig);
	SASSUME0(network != NULL);
	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
	network->loadPretrainedWeights();
#endif

	NetworkTestInterface<float>::globalSetUp(gpuid);

	NetworkTest<float>* networkTest = NULL;
	SNEW(networkTest, NetworkTest<float>, networkFilePath, networkName, numSteps, status);
	SASSUME0(networkTest != NULL);

	networkTest->setUp();

	Network<float>* network = networkTest->network;
	//network->load("/home/jkim/Dev/SOOOA_HOME/VGG16_CAFFE_TRAINED.param");
	network->save("/home/jkim/Dev/SOOOA_HOME/VGG16_CAFFE_TRAINED.param");



	NetworkTestInterface<float>::globalCleanUp();
}


#if 0
void saveNetworkParams(LayersConfig<float>* layersConfig) {
	const string savePathPrefix = "/home/jkim/Dev/SOOOA_HOME/param";
	ofstream paramOfs(
			(savePathPrefix+"/SSD_CAFFE_TRAINED.param").c_str(),
			ios::out | ios::binary);

	uint32_t numLearnableLayers = layersConfig->_learnableLayers.size();
	uint32_t numParams = 0;
	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		numParams += layersConfig->_learnableLayers[i]->numParams();
	}

	paramOfs.write((char*)&numParams, sizeof(uint32_t));
	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		layersConfig->_learnableLayers[i]->saveParams(paramOfs);
	}
	paramOfs.close();
}
#endif









void convertAnnoSetTest() {
#if 1
	const string baseImagePath = "/home/jkim/Dev/git/caffe_ssd/data/";
	const string baseSdfPath = "/home/jkim/Dev/LAONSILL_HOME/data/sdf/";

	const string labelMapFilePath = baseImagePath + "VOC0712/labelmap_voc.json";
	const string basePath = baseImagePath + "VOCdevkit/";
	const string outFilePath = baseSdfPath + "_voc2007_train_sdf/";

	ImageSet trainImageSet;
	trainImageSet.name = "train";
	trainImageSet.dataSetPath = baseImagePath + "VOC0712/trainval.txt";

	ImageSet testImageSet;
	testImageSet.name = "test";
	testImageSet.dataSetPath = baseImagePath + "VOC0712/test.txt";

	ConvertAnnoSetParam param;
	param.addImageSet(trainImageSet);
	param.addImageSet(testImageSet);
	param.labelMapFilePath = labelMapFilePath;
	param.basePath = basePath;
	param.outFilePath = outFilePath;


	if (boost::filesystem::exists(param.outFilePath)) {
		boost::filesystem::remove_all((param.outFilePath));
	}

	param.print();
	convertAnnoSet(param);
#else
	const string baseSdfPath = LAONSILL_HOME + "data/sdf/";
	const string sdfPath = baseSdfPath + "_voc0712_train_sdf/";

	DataReader<AnnotatedDatum> dataReader(sdfPath);

	AnnotatedDatum* datum = 0;
	cout << "selected train dataset ... " << endl;
	dataReader.selectDataSetByName("train");
	for (int i = 0; i < 10; i++) {
		datum = dataReader.getNextData();
		datum->print();
		SDELETE(datum);
		datum = 0;
	}

	cout << "selected test dataset ... " << endl;
	dataReader.selectDataSetByName("test");
	for (int i = 0; i < 10; i++) {
		datum = dataReader.getNextData();
		datum->print();
		SDELETE(datum);
		datum = 0;
	}
#endif
}


void convertVOC07SetTest() {
#if 1
	ImageSet trainImageSet;
	trainImageSet.name = "train";
	trainImageSet.dataSetPath = LAONSILL_HOME + "/data/VOC0712/trainval.txt.5.07only";

	ConvertAnnoSetParam param;
	param.addImageSet(trainImageSet);
	param.labelMapFilePath = LAONSILL_HOME + "/data/VOC0712/labelmap_voc.json";
	param.basePath = LAONSILL_HOME + "/data/VOCdevkit";
	param.outFilePath = LAONSILL_HOME + "/data/sdf/_voc2007_train_sdf_5";

	if (boost::filesystem::exists(param.outFilePath)) {
		boost::filesystem::remove_all((param.outFilePath));
	}

	param.print();
	convertAnnoSet(param);

	cout << "resultCode: " << param.resultCode << endl;
	cout << "resultMsg: " << param.resultMsg << endl;

#else
	const string baseSdfPath = LAONSILL_HOME + "data/sdf/";
	const string sdfPath = baseSdfPath + "_voc0712_train_sdf/";

	DataReader<AnnotatedDatum> dataReader(sdfPath);

	AnnotatedDatum* datum = 0;
	cout << "selected train dataset ... " << endl;
	dataReader.selectDataSetByName("train");
	for (int i = 0; i < 10; i++) {
		datum = dataReader.getNextData();
		datum->print();
		SDELETE(datum);
		datum = 0;
	}

	cout << "selected test dataset ... " << endl;
	dataReader.selectDataSetByName("test");
	for (int i = 0; i < 10; i++) {
		datum = dataReader.getNextData();
		datum->print();
		SDELETE(datum);
		datum = 0;
	}
#endif
}



void convertImageSetTest() {

#if 1

	//const string baseImagePath = LAONSILL_HOME + "/data/ilsvrc12_train";
	//const string baseSdfPath = LAONSILL_HOME + "/data/sdf";

	//const string imagePath = baseImagePath + "/images";
	//const string sdfPath = baseSdfPath + "/ilsvrc12_train_10000";


	ImageSet trainImageSet;
	trainImageSet.name = "noname";
	trainImageSet.dataSetPath = HOME + "/Dev/data/image/plantynet.1000/train.txt";

	ConvertImageSetParam param;
	param.addImageSet(trainImageSet);
	param.shuffle = true;
	param.resizeWidth = 250;
	param.resizeHeight = 375;
	param.basePath = HOME + "/Dev/data/image/plantynet.1000/";
	param.outFilePath = HOME + "/Dev/LAONSILL_HOME/data/sdf/version_test_imageset";
	param.labelMapFilePath = HOME + "/Dev/LAONSILL_HOME/labelmap/labelmap_plantynet.json";
	param.encoded = false;
	param.encodeType = "";

	if (boost::filesystem::exists(param.outFilePath)) {
		boost::filesystem::remove_all((param.outFilePath));
	}

	convertImageSet(param);

	cout << "resultCode: " << param.resultCode << endl;
	cout << "resultMsg: " << param.resultMsg << endl;


#else
	const string baseSdfPath = LAONSILL_HOME + "/data/sdf";
	const string sdfPath = baseSdfPath + "/_ilsvrc12_train_1000";

	DataReader<Datum> dataReader(sdfPath);

	Datum* datum = 0;
	cout << "selected train dataset ... " << endl;
	dataReader.selectDataSetByName("train");
	for (int i = 0; i < 10; i++) {
		datum = dataReader.getNextData();
		datum->print();
		SDELETE(datum);
		datum = 0;
	}

	cout << "selected test dataset ... " << endl;
	dataReader.selectDataSetByName("test");
	for (int i = 0; i < 10; i++) {
		datum = dataReader.getNextData();
		datum->print();
		SDELETE(datum);
		datum = 0;
	}
#endif
}

void convertMnistDataTest() {
#if 1
	MnistDataSet trainDataSet;
	trainDataSet.name = "train";
	trainDataSet.imageFilePath = "/home/jkim/Dev/git/caffe/data/mnist/train-images-idx3-ubyte";
	trainDataSet.labelFilePath = "/home/jkim/Dev/git/caffe/data/mnist/train-labels-idx1-ubyte";

	MnistDataSet testDataSet;
	testDataSet.name = "test";
	testDataSet.imageFilePath = "/home/jkim/Dev/git/caffe/data/mnist/t10k-images-idx3-ubyte";
	testDataSet.labelFilePath = "/home/jkim/Dev/git/caffe/data/mnist/t10k-labels-idx1-ubyte";

	ConvertMnistDataParam param;
	param.addDataSet(trainDataSet);
	param.addDataSet(testDataSet);
	param.outFilePath = "/home/jkim/imageset/lmdb/_mnist_train_sdf/";
	param.labelMapFilePath = "/home/jkim/Dev/git/caffe_ssd/data/VOC0712/labelmap_mnist.json";
	//param.outFilePath = "/home/jkim/Dev/LAONSILL_HOME/data/custum_sdf/mnist_test_sdf_8/";
	//param.labelMapFilePath = "/home/jkim/Dev/LAONSILL_HOME/labelmap/labelmap_mnist_8.json";

	if (boost::filesystem::exists(param.outFilePath)) {
		boost::filesystem::remove_all((param.outFilePath));
	}

	convertMnistData(param);
	//convertMnistDataTemp(param);


#else
	const string sdfPath = "/home/jkim/imageset/lmdb/_mnist_train_sdf/";
	DataReader<Datum> dataReader(sdfPath);

	Datum* datum = 0;
	cout << "selected train dataset ... " << endl;
	dataReader.selectDataSetByName("train");

	datum = dataReader.getNextData();
	SDELETE(datum);

	datum = dataReader.getNextData();
	SDELETE(datum);

	datum = dataReader.peekNextData();
	SDELETE(datum);

	datum = dataReader.peekNextData();
	SDELETE(datum);


#endif
}







