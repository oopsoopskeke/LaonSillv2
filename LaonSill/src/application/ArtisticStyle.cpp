/*
 * ArtisticStyle.cpp
 *
 *  Created on: Mar 17, 2017
 *      Author: jkim
 */
#if 0
#include "ArtisticStyle.h"

#include <stdint.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Debug.h"
#include "Param.h"
#include "MathFunctions.h"
#include "cnpy.h"
#include "TestUtil.h"
#include "CudaUtils.h"
#include "ApplicationCudaFunctions.h"

using namespace std;
using namespace cnpy;

#define ARTISTICSTYLE_LOG	1

#define LOAD_WEIGHT 		1

#define STYLE_SCALE			1.2

// DEBUGGING 용도의 SMALL TEST로 RUN
#define SMALL_TEST			0

// SOOOA 모델로 테스트인지 여부(RGB 모델),
// false인 경우 CAFFE MODEL이고 이는 BGR 기준의 모델을 말함
#define SOOOA				0


// BGRBGRBGR ... 순으로 들어있음. 255 scale (8, 12, 3), BGR순
// 33, 65, 128 x 12
// x 8
// python의 caffe.io.load_image의 경우 1 scale (8, 12, 3), RGB순
//

template <typename Dtype>
ArtisticStyle<Dtype>::ArtisticStyle() {
	//"/home/jkim/Downloads/sampleR32G64B128.png";
#if SMALL_TEST
	this->style_img			= "/data/backup/artistic/starry_night_64.jpg";
	this->content_img		= "/data/backup/artistic/tubingen_64.jpg";
#else
	//this->style_img			= "/home/jkim/Downloads/sampleR32G64B128.png";
	//this->content_img		= "/home/jkim/Downloads/sampleR32G64B128.png";
	//this->style_img			= "/data/backup/artistic/starry_night.jpg";
	//this->content_img		= "/data/backup/artistic/johannesburg.jpg";
	this->style_img			= "/home/jkim/Backups/artistic/starry_night.jpg";
	this->content_img		= "/home/jkim/Backups/artistic/tubingen_800.jpg";
	//this->content_img		= "/home/jkim/Backups/artistic/johannesburg.jpg";
#endif


	cv::namedWindow("CONTENT IMAGE");





	// 파이썬 기준 테스트를 위해 BGR2RGB COMMENT OUT
	this->cv_img_style 		= cv::imread(this->style_img);

	const string styleImageWindowName = "STYLE IMAGE";
	cv::namedWindow(styleImageWindowName);
	cv::moveWindow(styleImageWindowName, 0, 0);
	cv::imshow(styleImageWindowName, this->cv_img_style);


	this->cv_img_style.convertTo(this->cv_img_style, CV_32F);
#if SOOOA
	cv::cvtColor(this->cv_img_style, this->cv_img_style, CV_BGR2RGB);
#endif

	this->cv_img_content	= cv::imread(this->content_img);

	const string contentImageWindowName = "CONTENT IMAGE";
	cv::namedWindow(contentImageWindowName);
	cv::moveWindow(contentImageWindowName, 0, this->cv_img_style.rows + 100);
	cv::imshow("CONTENT IMAGE", this->cv_img_content);


	cv::namedWindow("RESULT");
	cv::moveWindow("RESULT", this->cv_img_content.cols + 10, this->cv_img_style.rows + 100);

	this->cv_img_content.convertTo(this->cv_img_content, CV_32F);
#if SOOOA
	cv::cvtColor(this->cv_img_content, this->cv_img_content, CV_BGR2RGB);
#endif





	load_model();
	rescale_net({1, 3, 224, 224});
	//rescale_net({1, 3, (uint32_t)this->cv_img_style.rows, (uint32_t)this->cv_img_style.cols});

	map<string, Dtype> content_weight;
	map<string, Dtype> style_weight;
	content_weight["conv4_2"]	= Dtype(1.0);
	style_weight["conv1_1"]		= Dtype(0.2);
	style_weight["conv2_1"]		= Dtype(0.2);
	style_weight["conv3_1"]		= Dtype(0.2);
	style_weight["conv4_1"]		= Dtype(0.2);
	style_weight["conv5_1"]		= Dtype(0.2);
	this->weights["content"]	= content_weight;
	this->weights["style"]		= style_weight;

	for (int i = 0; i < this->_layers.size(); i++) {
		const string& layerName = this->_layers[i]->name;
		if (this->weights["style"].find(layerName) != this->weights["style"].end() ||
				this->weights["content"].find(layerName) != this->weights["content"].end()) {
			this->layers.push_back(layerName);
		}
	}

	cout << "listing target layers ... " << endl;
	for (int i = 0; i < this->layers.size(); i++) {
		cout << "\t" << this->layers[i] << endl;
	}
	cout << endl;

	this->content_type = "content";
	//this->length	= 512;
	this->length	= 800;
	this->ratio		= 10000;
	//this->ratio		= 1000;
	//this->n_iter	= 512;
	this->n_iter	= 5120;
	this->init		= -1;

	this->optimizer_type = "adam";
	this->lr		= Dtype(1);
	this->wd		= Dtype(0.0005);
	this->mt		= Dtype(0.9);
	this->eps		= Dtype(0.000000001);
	this->bt1 		= Dtype(0.9);
	this->bt2 		= Dtype(0.999);

	this->hist		= new Data<Dtype>("hist");
	this->hist2		= new Data<Dtype>("hist2");

	this->end		= "conv5_2";
}

template <typename Dtype>
ArtisticStyle<Dtype>::~ArtisticStyle() {
	if (this->mean)
		delete this->mean;
	if (this->data_bounds)
		delete this->data_bounds;

	if (this->img_style)
		delete this->img_style;
	if (this->img_content)
		delete this->img_content;

	_clearNameDataMap(this->G_style);
	_clearNameDataMap(this->F_content);

	if (this->hist)
		delete this->hist;
	if (this->hist2)
		delete this->hist2;

	if (this->network)
		delete this->network;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::_clearNameDataMapVector(vector<map<string, Data<Dtype>*>>& v) {
	for (int i = 0; i < v.size(); i++) {
		_clearNameDataMap(v[i]);
	}
	v.clear();
}

template <typename Dtype>
void ArtisticStyle<Dtype>::_clearNameDataMap(map<string, Data<Dtype>*>& m) {
	typename map<string, Data<Dtype>*>::iterator it;
	for (it = m.begin(); it != m.end(); it++) {
		if (it->second)
			delete it->second;
	}
	m.clear();
}


template <typename Dtype>
void ArtisticStyle<Dtype>::transfer_style() {
	const vector<uint32_t>& inputShape = this->_layers[0]->_outputData[0]->getShape();
	Dtype orig_dim = Dtype(_min_from_shape(inputShape, 2, 3));
	for (int i = 0; i < inputShape.size(); i++) {
		cout << inputShape[i] << ", ";
	}
	cout << endl;

#if !SMALL_TEST
	// rescale the images
	Dtype scale = max(this->length / Dtype(_max_dim_from_mat(this->cv_img_style)),
			orig_dim / Dtype(_min_dim_from_mat(this->cv_img_style))) * STYLE_SCALE;
	cv::resize(this->cv_img_style, this->cv_img_style, cv::Size(), scale, scale,
			CV_INTER_LINEAR);
	cout << "scale: " << scale << endl;

	scale = max(this->length / Dtype(_max_dim_from_mat(this->cv_img_content)),
			orig_dim / Dtype(_min_dim_from_mat(this->cv_img_content)));
	cv::resize(this->cv_img_content, this->cv_img_content, cv::Size(), scale, scale,
			CV_INTER_LINEAR);
	cout << "scale: " << scale << endl;
#endif

	this->img_style = _from_mat_to_data(this->cv_img_style);
	this->img_content = _from_mat_to_data(this->cv_img_content);

	// compute style representations
	// 아직 preprocess 전이라 transpose되지 않아 ...
	transformer_preprocess(this->img_style);
	rescale_net(this->img_style->getShape());
	vector<string> layers = _map_keys(this->weights["style"]);
	/*
	//_on();
	cout << "listing style layers ... " << endl;
	for (int i = 0; i < layers.size(); i++) {
		cout << "\t" << layers[i] << endl;
	}
	this->img_style->print_data({}, false);
	//_off();
	*/
	Dtype gram_scale = Dtype(this->img_content->getCount()) / this->img_style->getCount();
	this->G_style = compute_reprs(this->img_style, layers, {})[0];

	/*
	_on();
	for (int i = 0; i < layers.size(); i++) {
		this->G_style[layers[i]]->print_data();
	}
	_off();
	*/
	// compute content representations
	transformer_preprocess(this->img_content);
	rescale_net(this->img_content->getShape());
	layers = _map_keys(this->weights["content"]);

	/*
	_on();
	cout << "listing style layers ... " << endl;
	for (int i = 0; i < layers.size(); i++) {
		cout << "\t" << layers[i] << endl;
	}
	this->img_content->print_data({}, false);
	_off();
	*/
	this->F_content = compute_reprs(this->img_content, {}, layers)[1];
	/*
	_on();
	for (int i = 0; i < layers.size(); i++) {
		this->F_content[layers[i]]->print_data();
	}
	_off();
	exit(1);
	*/

	// generate initial net input
	// "content" = content image
	Data<Dtype>* img0 = _generateInitialInput();
	this->hist->reshapeLike(img0);
	this->hist->reset_device_data();
	this->hist2->reshapeLike(img0);
	this->hist2->reset_device_data();
	/*
	_on();
	img0->print_data({}, false);
	_off();
	//exit(1);
	*/

	// optimization params

	// optimize
	Dtype loss = Dtype(0.0);
	Data<Dtype>* img0Disp = new Data<Dtype>("img0Disp");
	//cv::namedWindow("RESULT");

	for (int i = 0; i < this->n_iter; i++) {
		loss += style_optfn(img0);

		// update img0
		// img0 += lr * grad;
		Data<Dtype>* data = this->_layerDataMap["data"];

		/*
		_on();
		data->print_grad({}, false);
		img0->print_data({}, false);
		_off();
		*/
		int n = (int)data->getCount();
		Dtype* d_img0_data = img0->mutable_device_data();
		Dtype* d_data_grad = data->mutable_device_grad();
		Dtype* d_hist_data = this->hist->mutable_device_data();

		if (this->optimizer_type == "sgd") {
			soooa_gpu_axpy(n, this->wd, d_img0_data, d_data_grad);
			soooa_gpu_axpby(n, this->lr, d_data_grad, this->mt, d_hist_data);
			soooa_copy(n, d_hist_data, d_data_grad);
			soooa_gpu_axpy((int)n, Cuda::negativeOne, d_data_grad, d_img0_data);
		} else if (this->optimizer_type == "adagrad") {
			optimize_adagrad(n, d_data_grad, d_hist_data, d_img0_data, this->lr, this->eps);
		} else if (this->optimizer_type == "rmsprop") {
			optimize_rmsprop(n, d_data_grad, d_hist_data, d_img0_data, this->lr, this->eps,
					this->wd);
		} else if (this->optimizer_type == "adam") {
			Dtype* d_hist2_data = this->hist2->mutable_device_data();
			optimize_adam(n, d_data_grad, d_hist_data, d_hist2_data, d_img0_data, this->lr,
					this->eps, this->bt1, this->bt2);
		} else {
			cout << "invalid optimize_type: " << this->optimizer_type << endl;
			exit(1);
		}

		/*
		_on();
		img0->print_data({}, false);
		_off();
		*/
		// bound img0
		soooa_bound_data(n, img0->getCountByAxis(2),
				this->data_bounds->device_data(), img0->mutable_device_data());
		/*
		_on();
		img0->print_data({}, false);
		_off();

		exit(1);
		*/

		if (i % 1 == 0) {
			img0Disp->set(img0);
			transformer_deprocess(img0Disp);
			cv::Mat result(img0Disp->getShape(1), img0Disp->getShape(2), CV_32FC3,
					img0Disp->mutable_host_data());

#if SOOOA
			cv::cvtColor(result, result, CV_RGB2BGR);
#endif
			cv::imshow("RESULT", result);
			cv::waitKey(1);
			cout << "progress: (" << (i+1) << " / " << this->n_iter << ")" << endl;
		}
	}

	delete img0;
	delete img0Disp;
}


template <typename Dtype>
void ArtisticStyle<Dtype>::load_model() {
	const vector<string> lossLayers = {"loss"};
	const NetworkPhase phase = NetworkPhase::TrainPhase;

#if LOAD_WEIGHT
	vector<WeightsArg> weightsArgs(1);
#if SOOOA
	weightsArgs[0].weightsPath = "/home/jkim/Dev/SOOOA_HOME/network/VGG19.param";
#else
	weightsArgs[0].weightsPath = "/home/jkim/Dev/SOOOA_HOME/network/VGG19_CAFFE_ARTISTICPARTONLY.param";
	//weightsArgs[0].weightsPath = "/home/jkim/Dev/SOOOA_HOME/network/VGG19_LMDB_0.01.param";
#endif
#endif
	const uint32_t batchSize = 10;
	const uint32_t testInterval = 1000;			// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 10000;		// 1000000 / batchSize
	const float baseLearningRate = 0.001f;
	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0005f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.0001;
	const LRPolicy lrPolicy = LRPolicy::Fixed;

	NetworkConfig<Dtype>* networkConfig =
			(new typename NetworkConfig<Dtype>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->testInterval(testInterval)
			->saveInterval(saveInterval)
			->stepSize(stepSize)
			->clipGradientsLevel(clipGradientsLevel)
			->lrPolicy(lrPolicy)
			->networkPhase(phase)
			->gamma(gamma)
			->savePathPrefix(SPARAM(NETWORK_SAVE_DIR))
#if LOAD_WEIGHT
			->weightsArgs(weightsArgs)
#endif
			->networkListeners({
				new NetworkMonitor("loss", NetworkMonitor::PLOT_ONLY)
				})
			->lossLayers(lossLayers)
			->build();

	// 네트워크를 등록한다.
	Network<Dtype>* network = new Network<Dtype>(networkConfig);
	LayersConfig<Dtype>* layersConfig = createVGG19NetArtisticLayersConfig<Dtype>();
	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
	network->loadPretrainedWeights();

	this->network 		= network;
	this->_layers		= layersConfig->_layers;
	this->_layerDataMap = layersConfig->_layerDataMap;
	this->_nameLayerMap	= layersConfig->_nameLayerMap;

	this->mean			= new Data<Dtype>("mean");
	this->mean->reshape({1, 1, 1, 3});

	Dtype mean_b = Dtype(104.00698793);
	Dtype mean_g = Dtype(116.66876762);
	Dtype mean_r = Dtype(122.67891434);

#if SOOOA
	this->mean->mutable_host_data()[0] = mean_r;
	this->mean->mutable_host_data()[1] = mean_g;
	this->mean->mutable_host_data()[2] = mean_b;

	this->data_bounds	= new Data<Dtype>("bound");
	this->data_bounds->reshape({1, 1, 1, 6});
	this->data_bounds->mutable_host_data()[0] = -mean_r;
	this->data_bounds->mutable_host_data()[1] = -mean_r + Dtype(255);
	this->data_bounds->mutable_host_data()[2] = -mean_g;
	this->data_bounds->mutable_host_data()[3] = -mean_g + Dtype(255);
	this->data_bounds->mutable_host_data()[4] = -mean_b;
	this->data_bounds->mutable_host_data()[5] = -mean_b + Dtype(255);
#else
	// 현재 BGR 기준
	this->mean->mutable_host_data()[0] = mean_b;
	this->mean->mutable_host_data()[1] = mean_g;
	this->mean->mutable_host_data()[2] = mean_r;

	this->data_bounds	= new Data<Dtype>("bound");
	this->data_bounds->reshape({1, 1, 1, 6});
	this->data_bounds->mutable_host_data()[0] = -mean_b;
	this->data_bounds->mutable_host_data()[1] = -mean_b + Dtype(255);
	this->data_bounds->mutable_host_data()[2] = -mean_g;
	this->data_bounds->mutable_host_data()[3] = -mean_g + Dtype(255);
	this->data_bounds->mutable_host_data()[4] = -mean_r;
	this->data_bounds->mutable_host_data()[5] = -mean_r + Dtype(255);
#endif

#if 0
	// ***************************************************************************************
	// CAFFE MODEL NPZ로부터 WEIGHT를 LOAD하는 코드,
	// SOOOA PARAM으로 변환후에는 아래의 코드 대신 정상 루트를 통해 WEIGHT LOAD하면 된다.
	map<string, Data<Dtype>*> nameParamsOldMap;
	const string& path = "/home/jkim/";
	buildNameDataMapFromNpzFile("/home/jkim/", "vgg19_params_old", nameParamsOldMap);
	printNameDataMap("nameParamsOldMap", nameParamsOldMap, false);

	vector<LearnableLayer<Dtype>*>& learnableLayers = layersConfig->_learnableLayers;
	for (int i = 0; i < learnableLayers.size(); i++) {
		fillParam(nameParamsOldMap, learnableLayers[i]->name + "*params*",
				learnableLayers[i]->_params);
		// 반드시 외부에서 params init되었음을 설정해야 한다.
		for (int j = 0; j < learnableLayers[i]->_params.size(); j++) {
			learnableLayers[i]->_paramsInitialized[j] = true;
		}
	}

	network->save();

#if 0
	_on();
	learnableLayers[0]->_params[0]->print_data({}, false);
	_off();
	exit(1);
#endif
	// ***************************************************************************************
#endif
}

template <typename Dtype>
void ArtisticStyle<Dtype>::rescale_net(const std::vector<uint32_t>& shape) {
	this->_layers[0]->_outputData[0]->reshape(shape);
	for (int i = 0; i < this->_layers.size(); i++) {
		this->_layers[i]->reshape();
	}
}

template <typename Dtype>
vector<map<string, Data<Dtype>*>> ArtisticStyle<Dtype>::compute_reprs(Data<Dtype>* net_in,
		const vector<string>& layers_style, const vector<string>& layers_content,
		const Dtype gram_scale) {
	vector<map<string, Data<Dtype>*>> repr(2);

	net_forward(net_in);

	set<string> layers;
	layers.insert(layers_style.begin(), layers_style.end());
	layers.insert(layers_content.begin(), layers_content.end());

	set<string>::iterator itr;
	for (itr = layers.begin(); itr != layers.end(); itr++) {
		const string& layer = *itr;

		Data<Dtype>* F = new Data<Dtype>(this->_layerDataMap[layer]);

		int M = F->getShape(1);
		int K = F->getCountByAxis(2);
		int N = M;

		F->reshape({1, 1, (uint32_t)M, (uint32_t)K});
		repr[1][layer] = F;

		//_on();
		repr[1][layer]->print_data({}, false);
		//_off();

		if (find(layers_style.begin(), layers_style.end(), layer) != layers_style.end()) {
			Data<Dtype>* G = new Data<Dtype>("G");
			G->reshape({1, 1, (uint32_t)M, (uint32_t)N});
			soooa_gpu_gemm(CblasNoTrans, CblasTrans,
				M, N, K,
				Cuda::alpha, F->device_data(), F->device_data(),
				Cuda::beta, G->mutable_device_data());;
			repr[0][layer] = G;

			//_on();
			repr[0][layer]->print_data({}, false);
			//_off();
		}
	}
	return repr;
}

template <typename Dtype>
Dtype ArtisticStyle<Dtype>::compute_style_grad(map<string, Data<Dtype>*>& F,
		map<string, Data<Dtype>*>& G, const string& layer, Data<Dtype>* grad) {
	// compute loss and gradient
	Data<Dtype>* Fl = F[layer];
	Data<Dtype>* Gl = G[layer];
	Dtype c = pow(Fl->getShape(2), -2.0) * pow(Fl->getShape(3), -2.0);
	Data<Dtype>* El = new Data<Dtype>(Gl);
	El->sub_device_data(this->G_style[layer]);
	Dtype loss = c / 4 * El->sumsq_device_data();

	int M = El->getShape(2);
	int N = Fl->getShape(3);
	int K = El->getShape(3);

	grad->reshape({1, 1, (uint32_t)M, (uint32_t)N});
	soooa_gpu_gemm(CblasNoTrans, CblasNoTrans,
			M, N, K,
			Cuda::alpha, El->device_data(), Fl->device_data(),
			Cuda::beta, grad->mutable_device_data());

	grad->scale_device_data(c);


	/*
	_on();
	Fl->print_data({}, false);
	grad->print_data({}, false);
	_off();
	*/

	reset_when_condition_le_0(grad->getCount(), Fl->device_data(),
			grad->mutable_device_data());

	/*
	_on();
	grad->print_data({}, false);
	_off();
	*/

	/*
	_on();
	Fl->print_data({}, false);
	Gl->print_data({}, false);
	El->print_data({}, false);
	grad->print_data({}, false);
	_off();
	exit(1);
	*/
	//cout << "loss: " << loss << endl;
	//exit(1);
	delete El;

	return loss;
}

template <typename Dtype>
Dtype ArtisticStyle<Dtype>::compute_content_grad(map<string, Data<Dtype>*>& F,
		const string& layer, Data<Dtype>* grad) {
	// compute loss and gradient
	Data<Dtype>* Fl = F[layer];
	Data<Dtype>* El = new Data<Dtype>(Fl);
	El->sub_device_data(this->F_content[layer]);
	Dtype loss = El->sumsq_device_data() / Dtype(2.0);
	grad->reshapeLike(El);
	grad->set_device_data(El);
	reset_when_condition_le_0(grad->getCount(), Fl->device_data(),
			grad->mutable_device_data());
	/*
	_on();
	Fl->print_data({}, false);
	El->print_data({}, false);
	grad->print_data({}, false);
	_off();
	*/
	delete El;

	return loss;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::transformer_preprocess(Data<Dtype>* data) {
	// 1. Network 입력에 맞게 reshape
	// 		-> 필요하면 넣어야 함

	// 2. transpose
	data->transpose({0, 3, 1, 2});

	// 3. channel swap
	//		-> BGR로 테스트, 현재 필요 없음

	// 4. raw_scale
	//		-> 이미 255 LEVEL, 필요없음

	// 5. mean subtraction
	soooa_sub_channel_mean((int)data->getCount(), (uint32_t)data->getCountByAxis(2),
			this->mean->device_data(), data->mutable_device_data());

	// 6. input scale
	//		-> NOT USED
}

template <typename Dtype>
void ArtisticStyle<Dtype>::transformer_deprocess(Data<Dtype>* data) {
	// 6. input scale
	//		-> NOT USED

	// 5. mean addition
	soooa_add_channel_mean((int)data->getCount(), (uint32_t)data->getCountByAxis(2),
			this->mean->device_data(), data->mutable_device_data());

	// 4. raw_scale
	//		-> 이미 255 LEVEL, 필요없음
	// Array로부터 cv::Mat 생성할 때 255scale float로 만드는 법을 모르겠어서
	// 일단 1.0scale로 변환
	data->scale_device_data(1.0/255.0);

	// 3. channel swap
	//		-> BGR로 테스트, 현재 필요 없음

	// 2. transpose
	data->transpose({0, 2, 3, 1});

	// 1. Network 입력에 맞게 reshape
	// 		-> 필요하면 넣어야 함
}

template <typename Dtype>
void ArtisticStyle<Dtype>::net_forward(Data<Dtype>* net_in) {
	this->_layers[0]->_outputData[0]->reshapeLike(net_in);
	this->_layers[0]->_outputData[0]->set_device_data(net_in);
	for (int i = 0; i < this->_layers.size(); i++) {
		this->_layers[i]->feedforward();
		if (this->_layers[i]->name == this->end)
			break;
	}
}

template <typename Dtype>
Dtype ArtisticStyle<Dtype>::style_optfn(Data<Dtype>* net_in) {
	// update params
	const vector<string>& layers_style = _map_keys(this->weights["style"]);
	const vector<string>& layers_content = _map_keys(this->weights["content"]);

	// compute representations
	vector<map<string, Data<Dtype>*>> GF = compute_reprs(net_in, layers_style,
			layers_content);
	map<string, Data<Dtype>*> G = GF[0];
	map<string, Data<Dtype>*> F = GF[1];

	/*
	_on();
	net_in->print_data({}, false);
	G["conv1_1"]->print_data({}, false);
	G["conv5_1"]->print_data({}, false);
	_off();
	exit(1);
	*/
	// backprop by layer
	Data<Dtype>* grad = 0;
	Dtype loss = Dtype(0.0);

	int endLayerIndex = this->layers.size() - 1;
	//this->_layers[endLayerIndex]->_outputData[0]->reset_device_data();
	this->_layerDataMap[this->layers[endLayerIndex]]->reset_device_data();
	for (int i = endLayerIndex; i >= 0; i--) {
		const string& layer = this->layers[i];
		const string& next_layer = (i == 0)?"":this->layers[i-1];
		//cout << "layer: " << layer << ", next_layer: " << next_layer << endl;

		grad = this->_layerDataMap[this->layers[i]];
		/*
		_on();
		grad->print_grad({}, false);
		_off();
		*/

		// style contribution
		if (find(layers_style.begin(), layers_style.end(), layer) != layers_style.end()) {
			Dtype wl = this->weights["style"][layer];
			Data<Dtype>* g = new Data<Dtype>("g");
			g->_data = g->_grad;

			Dtype l = compute_style_grad(F, G, layer, g);
			loss += wl * l * this->ratio;
			g->reshapeLike(grad);
			g->scale_device_data(wl * this->ratio);
			grad->add_device_grad(g);
			/*
			_on();
			grad->print_grad({}, false);
			_off();
			*/
			delete g;
		}

		// content contribution
		if (find(layers_content.begin(), layers_content.end(), layer) != layers_content.end()) {
			Dtype wl = this->weights["content"][layer];
			Data<Dtype>* g = new Data<Dtype>("g");
			g->_data = g->_grad;

			Dtype l = compute_content_grad(F, layer, g);
			loss += wl * l;
			g->reshapeLike(grad);
			g->scale_device_data(wl);
			grad->add_device_grad(g);
			/*
			_on();
			grad->print_grad({}, false);
			_off();
			*/
			delete g;
		}

		// compute gradient
		this->network->_backpropagationFromTo(layer, next_layer);
		if (next_layer == "") {
			grad = this->_layerDataMap["data"];
		} else {
			grad = this->_layerDataMap[next_layer];
		}
	}


	_clearNameDataMapVector(GF);

	/*
	_on();
	grad->print_grad({}, false);
	_off();
	cout << "loss: " << loss << endl;
	exit(1);
	*/

	// format gradient for minimize() function
	// grad->reshape({1, 1, 1, (uint32_t)grad->getCount()});

	return loss;
}






template <typename Dtype>
void ArtisticStyle<Dtype>::_on() {
	Data<Dtype>::printConfig = true;
	SyncMem<Dtype>::printConfig = true;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::_off() {
	Data<Dtype>::printConfig = false;
	SyncMem<Dtype>::printConfig = false;
}

template <typename Dtype>
Data<Dtype>* ArtisticStyle<Dtype>::_from_mat_to_data(const cv::Mat& mat) {
	Data<Dtype>* data = new Data<Dtype>("data");
	data->reshape({1, (uint32_t)mat.rows, (uint32_t)mat.cols, (uint32_t)mat.channels()});
	data->set_host_data((Dtype*)mat.data);
	return data;
}

template <typename Dtype>
uint32_t ArtisticStyle<Dtype>::_max_from_shape(const std::vector<uint32_t>& shape,
		int from, int to) {
	assert(from >= 0 && to >= from && to < shape.size());

	uint32_t max = 0;
	for (int i = from; i < to; i++) {
		if (shape[i] > max)
			max = shape[i];
	}
	return max;
}

template <typename Dtype>
uint32_t ArtisticStyle<Dtype>::_min_from_shape(const std::vector<uint32_t>& shape,
		int from, int to) {
	assert(from >= 0 && to >= from && to < shape.size());

	uint32_t min = UINT32_MAX;
	for (int i = from; i < to; i++) {
		if (shape[i] < min)
			min = shape[i];
	}
	return min;
}

template <typename Dtype>
uint32_t ArtisticStyle<Dtype>::_max_dim_from_mat(const cv::Mat& mat) {
	return uint32_t(max(mat.rows, mat.cols));
}

template <typename Dtype>
uint32_t ArtisticStyle<Dtype>::_min_dim_from_mat(const cv::Mat& mat) {
	return uint32_t(min(mat.rows, mat.cols));
}

template <typename Dtype>
const vector<string> ArtisticStyle<Dtype>::_map_keys(map<string, Dtype>& arg) {
	vector<string> keys;

	for (typename map<string, Dtype>::iterator it = arg.begin(); it != arg.end(); it++) {
		keys.push_back(it->first);
	}
	return keys;
}

template <typename Dtype>
Data<Dtype>* ArtisticStyle<Dtype>::_generateInitialInput() {
	Data<Dtype>* img0 = new Data<Dtype>("img0");

	if (this->content_type == "noise") {
		// ***********************************************************************************
		// PYTHON 생성 NOISE를 INPUT으로 이용하는 코드
	#if SMALL_TEST
		npz_t cnpy_npz = npz_load("/home/jkim/input_noise_64.npz");
	#else
		npz_t cnpy_npz = npz_load("/home/jkim/input_noise_512.npz");
	#endif
		NpyArray input_noise = cnpy_npz["input_noise"];

		vector<uint32_t> shape(4);
		int gap = shape.size() - input_noise.shape.size();
		for (int i = 0; i < 4; i++) {
			if (i < gap)
				shape[i] = 1;
			else
				shape[i] = input_noise.shape[i - gap];
		}

		img0->reshape(shape);
		img0->set_host_data((Dtype*)input_noise.data);

	#if 0
		_on();
		img0->print_data({}, false);
		_off();
	#endif
		// ***********************************************************************************
	} else if (this->content_type == "content") {
		img0->set(this->img_content);
	} else if (this->content_type == "mixed") {
		img0->set(this->img_content);
		img0->scale_device_data(0.95f);
		Data<Dtype>* img0_style = new Data<Dtype>(this->img_style);
		img0_style->scale_device_data(0.05f);
		img0->add_device_data(img0_style);
		delete img0_style;
	} else {
		cout << "invalid content type: " << this->content_type << endl;
		exit(1);
	}

	return img0;
}


template class ArtisticStyle<float>;

























#endif
