/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "Util.h"
#include "Network.h"
#include "cuda_runtime.h"
#include "MathFunctions.h"
#include <algorithm>
#include "PropMgmt.h"
#include "Update.h"
#include "Updater.h"
#include "Donator.h"
#include "MemoryMgmt.h"
#include "StdOutLog.h"

#define CONVLAYER_LOG 0

using namespace std;

/**
 * dst array에 src array를 더한다.
 *
 * @param dst dst array, dst + src가 저장이 될 장소
 * @param src src array
 * @param N The number of elements in the array.
 */
template <typename Dtype>
__global__ void AddArrayOfConvLayer(Dtype* dst, const Dtype* src, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	dst[idx] = dst[idx] + src[idx];
}

template<typename Dtype>
ConvLayer<Dtype>::ConvLayer()
: LearnableLayer<Dtype>() {
	this->type = Layer<Dtype>::Conv;

	const string& name = SLPROP_BASE(name);
	filter_dim& filterDim = SLPROP(Conv, filterDim);
	const bool deconv = SLPROP(Conv, deconv);
	this->biasTerm = SLPROP(Conv, biasTerm);

	if (filterDim.pad > 0) {
		SASSERT0(filterDim.pad_h == 0 && filterDim.pad_w == 0);
		filterDim.pad_h = filterDim.pad;
		filterDim.pad_w = filterDim.pad;
	}


	if (!deconv)
		this->type = Layer<Dtype>::Conv;
	else
		// HEIM : Deconv Debug 용. conv -> deconv layer 로 donate 할 때, 에러 무시하기 위한 수정. 해결책 마련 후 수정 필요.
		// this->type = Layer<Dtype>::Deconv;
		this->type = Layer<Dtype>::Conv;

	Optimizer opt = (Optimizer)SNPROP(optimizer);
	int paramHistoryDataCount = Update<Dtype>::getParamHistoryDataCount(opt);

	this->_params.resize(1);
	this->_params[Filter] = NULL;
	SNEW(this->_params[Filter], Data<Dtype>, name + "_filter");
	SASSUME0(this->_params[Filter] != NULL);

#if 0
	this->_params[Filter]->reshape(
			{filterDim.filters, filterDim.channels, filterDim.rows, filterDim.cols});

	param_filler<Dtype>& weightFiller = SLPROP(Conv, weightFiller);
	weightFiller.fill(this->_params[Filter]);
#endif

	this->_paramsHistory.resize(1);
	this->_paramsHistory[Filter] = NULL;

	if (paramHistoryDataCount >= 1) {
		SNEW(this->_paramsHistory[Filter], Data<Dtype>, name + "_filter_history");
		SASSUME0(this->_paramsHistory[Filter] != NULL);

#if 0
		this->_paramsHistory[Filter]->reshape(
				{filterDim.filters, filterDim.channels, filterDim.rows, filterDim.cols});
#endif
	}

	this->_paramsHistory2.resize(1);
	this->_paramsHistory2[Filter] = NULL;

	if (paramHistoryDataCount >= 2) {
		SNEW(this->_paramsHistory2[Filter], Data<Dtype>, name + "_filter_history2");
		SASSUME0(this->_paramsHistory2[Filter] != NULL);

#if 0
		this->_paramsHistory2[Filter]->reshape(
				{filterDim.filters, filterDim.channels, filterDim.rows, filterDim.cols});
#endif
	}

	this->_paramsInitialized.resize(1);
	this->_paramsInitialized[Filter] = false;


    if (this->updateParams.size() == 0) {
        UpdateParam upFilter;
        upFilter.paramType = Filter;
        upFilter.paramDataPtr = (void*)this->_params[Filter];
        upFilter.paramHis1Ptr = (void*)this->_paramsHistory[Filter];
        upFilter.paramHis2Ptr = (void*)this->_paramsHistory2[Filter];
        this->updateParams.push_back(upFilter);
    }

	checkCUDNN(cudnnCreateTensorDescriptor(&this->inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&this->filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&this->convDesc));

#if 0
	if (!deconv) {
		checkCUDNN(cudnnSetFilter4dDescriptor(this->filterDesc,
			CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			filterDim.filters, filterDim.channels, filterDim.rows, filterDim.cols));
	} else {
		checkCUDNN(cudnnSetFilter4dDescriptor(this->filterDesc,
			CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			filterDim.channels, filterDim.filters, filterDim.rows, filterDim.cols));
	}
#endif

	//int pad = (filterDim.rows-1)/2;
	checkCUDNN(cudnnSetConvolution2dDescriptor(this->convDesc,
			filterDim.pad_h, filterDim.pad_w, filterDim.stride, filterDim.stride,
			filterDim.dilation, filterDim.dilation,
			CUDNN_CROSS_CORRELATION
			,CUDNN_DATA_FLOAT));

	this->d_workspace = 0;


	if (this->biasTerm) {
		this->_params.resize(2);
		this->_params[Bias] = NULL;
		SNEW(this->_params[Bias], Data<Dtype>, name + "_bias");
		SASSUME0(this->_params[Bias] != NULL);
		this->_params[Bias]->reshape({filterDim.filters, 1, 1, 1});
		param_filler<Dtype>& biasFiller = SLPROP(Conv, biasFiller);
		biasFiller.fill(this->_params[Bias]);

		this->_paramsHistory.resize(2);
		this->_paramsHistory[Bias] = NULL;

		if (paramHistoryDataCount >= 1) {
			SNEW(this->_paramsHistory[Bias], Data<Dtype>, name + "_bias_history");
			SASSUME0(this->_paramsHistory[Bias] != NULL);
			this->_paramsHistory[Bias]->reshape({filterDim.filters, 1, 1, 1});
		}

		this->_paramsHistory2.resize(2);
		this->_paramsHistory2[Bias] = NULL;

		if (paramHistoryDataCount >= 2) {
			SNEW(this->_paramsHistory2[Bias], Data<Dtype>, name + "_bias_history2");
			SASSUME0(this->_paramsHistory2[Bias] != NULL);
			this->_paramsHistory2[Bias]->reshape({filterDim.filters, 1, 1, 1});
		}

		this->_paramsInitialized.resize(2);
		this->_paramsInitialized[Bias] = false;

		if (this->updateParams.size() == 1) {
			UpdateParam upBias;
			upBias.paramType = Bias;
			upBias.paramDataPtr = (void*)this->_params[Bias];
			upBias.paramHis1Ptr = (void*)this->_paramsHistory[Bias];
			upBias.paramHis2Ptr = (void*)this->_paramsHistory2[Bias];
			this->updateParams.push_back(upBias);
		}

		checkCUDNN(cudnnCreateTensorDescriptor(&this->biasTensorDesc));
		checkCUDNN(cudnnSetTensor4dDescriptor(this->biasTensorDesc,
				CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filterDim.filters, 1, 1));
	}
}

template <typename Dtype>
ConvLayer<Dtype>::~ConvLayer() {
    if (SLPROP_BASE(receive)) {
        Donator<Dtype>::releaseReceiver(SLPROP_BASE(donatorID));
    } else {
    	SDELETE(this->_params[ParamType::Filter]);
		if (this->_paramsHistory[ParamType::Filter] != NULL)
            SDELETE(this->_paramsHistory[ParamType::Filter]);
		if (this->_paramsHistory2[ParamType::Filter] != NULL)
			SDELETE(this->_paramsHistory2[ParamType::Filter]);

        if (biasTerm) {
			SDELETE(this->_params[ParamType::Bias]);
	        if (this->_paramsHistory[ParamType::Bias] != NULL)
	            SDELETE(this->_paramsHistory[ParamType::Bias]);
	        if (this->_paramsHistory2[ParamType::Bias] != NULL)
	            SDELETE(this->_paramsHistory2[ParamType::Bias]);
        }
        this->_params.clear();
        this->_paramsHistory.clear();
        this->_paramsHistory2.clear();
    }

	if(this->d_workspace) 
        CUDAFREE(this->d_workspace);

	checkCUDNN(cudnnDestroyTensorDescriptor(this->inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputTensorDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(this->filterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(this->convDesc));

	if (this->biasTerm) {
		checkCUDNN(cudnnDestroyTensorDescriptor(this->biasTensorDesc));
	}

    this->updateParams.clear();
}


template <typename Dtype>
void ConvLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();

	//
	if (adjusted) {
		Optimizer opt = (Optimizer)SNPROP(optimizer);
		int paramHistoryDataCount = Update<Dtype>::getParamHistoryDataCount(opt);

		filter_dim& filterDim = SLPROP(Conv, filterDim);
		filterDim.channels = this->_inputData[0]->getShape(1);

		vector<uint32_t> filterShape;
		const bool deconv = SLPROP(Conv, deconv);

        if (!deconv) {
            filterShape.push_back(filterDim.filters);
            filterShape.push_back(filterDim.channels);
            filterShape.push_back(filterDim.rows);
            filterShape.push_back(filterDim.cols);
        } else {
            filterShape.push_back(filterDim.channels);
            filterShape.push_back(filterDim.filters);
            filterShape.push_back(filterDim.rows);
            filterShape.push_back(filterDim.cols);
        }

		this->_params[Filter]->reshape(filterShape);

		param_filler<Dtype>& weightFiller = SLPROP(Conv, weightFiller);
		weightFiller.fill(this->_params[Filter]);


		if (paramHistoryDataCount >= 1) {
			this->_paramsHistory[Filter]->reshape(filterShape);
		}

		if (paramHistoryDataCount >= 2) {
			this->_paramsHistory2[Filter]->reshape(filterShape);
		}

		if (!deconv) {
			checkCUDNN(cudnnSetFilter4dDescriptor(this->filterDesc,
				CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
				filterDim.filters, filterDim.channels, filterDim.rows, filterDim.cols));
		} else {
			checkCUDNN(cudnnSetFilter4dDescriptor(this->filterDesc,
				CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
				filterDim.channels, filterDim.filters, filterDim.rows, filterDim.cols));
		}

	}


	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const bool deconv = SLPROP(Conv, deconv);
	const filter_dim& filterDim = SLPROP(Conv, filterDim);
	const int deconvExtraCell = SLPROP(Conv, deconvExtraCell);
	const string& name = SLPROP_BASE(name);

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	int n = 0, c = 0, h = 0, w = 0;
    if (!deconv) {
	    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
			this->convDesc,
			this->inputTensorDesc,
			this->filterDesc,
			&n, &c, &h, &w));
    } else {
        SASSERT0(rows == cols);
        SASSERT0(filterDim.rows == filterDim.cols);
        SASSERT0(deconvExtraCell < filterDim.stride);




        // See ConvLayer.h
        n = batches;
        //c = filterDim.filters * filterDim.channels;
        c = filterDim.filters;
        h = filterDim.stride * (cols - 1) + filterDim.cols -
            2 * filterDim.pad + deconvExtraCell;
        w = filterDim.stride * (rows - 1) + filterDim.rows -
            2 * filterDim.pad + deconvExtraCell;
    }

    checkCUDNN(cudnnSetTensor4dDescriptor(
        this->outputTensorDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        n, c, h, w));

	const uint32_t obatches = static_cast<uint32_t>(n);
	const uint32_t ochannels = static_cast<uint32_t>(c);
	const uint32_t orows = static_cast<uint32_t>(h);
	const uint32_t ocols = static_cast<uint32_t>(w);

#if CONVLAYER_LOG
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			name.c_str(), obatches, ochannels, orows, ocols);
#endif

	this->_inputShape[0] = inputShape;
	//this->_preActivation->reshape({obatches, ochannels, orows, ocols});
	this->_outputData[0]->reshape({obatches, ochannels, orows, ocols});


	const int u_in = channels * rows * cols;
	const int u_out = c * h * w;
	const int b_in = batches * channels * rows * cols;
	const int b_out = n * c * h * w;

	const size_t memoryLimitInBytes = 8 << 20;
	//const size_t memoryLimitInBytes = 0;

	size_t convFwdWorkspaceSize;
	size_t convBwdFilterWorkspaceSize;
	size_t convBwdDataWorkspaceSize;

	// forward algorithm
    if (!deconv) {
	    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
			//CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
			memoryLimitInBytes,
			&this->convFwdAlgo));

		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->convFwdAlgo,
			&convFwdWorkspaceSize));
    } else {
	    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
			Cuda::cudnnHandle,
			this->outputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->inputTensorDesc,
			//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
			//CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
			8<<20,
			&this->convFwdAlgo));

		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
			Cuda::cudnnHandle,
			this->outputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->convFwdAlgo,
			&convFwdWorkspaceSize));
    }

	// backward filter algorithm
    if (!deconv) {
	    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			//CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
			//CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
			memoryLimitInBytes,
			&this->convBwdFilterAlgo));

	    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			this->convBwdFilterAlgo,
			&convBwdFilterWorkspaceSize));
    } else {
	    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
			Cuda::cudnnHandle,
			this->outputTensorDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->filterDesc,
			//CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
			//CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
			8<<20,
			&this->convBwdFilterAlgo));

	    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
			Cuda::cudnnHandle,
			this->outputTensorDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->filterDesc,
			this->convBwdFilterAlgo,
			&convBwdFilterWorkspaceSize));
    }

	// backward data algorithm
    if (!deconv) {
	    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			//CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            //CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
			memoryLimitInBytes,
			&this->convBwdDataAlgo));

	    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->convBwdDataAlgo,
			&convBwdDataWorkspaceSize));
    } else {
	    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->outputTensorDesc,
			//CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            //CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
			8<<20,
			&this->convBwdDataAlgo));

	    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->convBwdDataAlgo,
			&convBwdDataWorkspaceSize));
    }

	this->workspaceSize = 0;
	this->workspaceSize = max(this->workspaceSize, convFwdWorkspaceSize);
	this->workspaceSize = max(this->workspaceSize, convBwdFilterWorkspaceSize);
	this->workspaceSize = max(this->workspaceSize, convBwdDataWorkspaceSize);


	if (this->workspaceSize > 0) {
#if CONVLAYER_LOG
		cout << name << "'s workspace: " << this->workspaceSize << endl;
#endif
		if (this->d_workspace) {
            CUDAFREE(this->d_workspace);
			this->d_workspace = 0;
		}
        CUDAMALLOC(&this->d_workspace, this->workspaceSize);
	}
}

template <typename Dtype>
void ConvLayer<Dtype>::update() {
	const filter_dim& filterDim = SLPROP(Conv, filterDim);
	const update_param& weightUpdateParam = SLPROP(Conv, weightUpdateParam);
	const float weightDecay = SNPROP(weightDecay);
	const float beta1 = SNPROP(beta1);
	const float beta2 = SNPROP(beta2);

	// update filters ...
	const uint32_t weightSize = filterDim.size();
	const Dtype regScale = weightDecay * weightUpdateParam.decay_mult;
	const Dtype learnScale = 
			Update<Dtype>::calcLearningRate() * weightUpdateParam.lr_mult;
    SLPROP(Conv, decayedBeta1) *= beta1;
    SLPROP(Conv, decayedBeta2) *= beta2;

    UpdateContext contextFilter = 
        Update<Dtype>::makeContext(weightSize, regScale, learnScale);

    SASSUME0(this->updateParams.size() > 0);
    this->updateParams[Filter].context = contextFilter;

    if (this->biasTerm) {
		// update biases ...
		const update_param& biasUpdateParam = SLPROP(Conv, biasUpdateParam);
		const uint32_t biasSize = filterDim.filters;
		const Dtype regScale_b = weightDecay * biasUpdateParam.decay_mult;
		const Dtype learnScale_b =
				Update<Dtype>::calcLearningRate() * biasUpdateParam.lr_mult;

		UpdateContext contextBias =
			Update<Dtype>::makeContext(biasSize, regScale_b, learnScale_b);

		SASSUME0(this->updateParams.size() == 2);
		this->updateParams[Bias].context = contextBias;
    }

    Updater::updateParams(this->updateParams);
}

template <typename Dtype>
void ConvLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	const filter_dim& filterDim = SLPROP(Conv, filterDim);
	const uint32_t weightSize = filterDim.size();
	const uint32_t biasSize = filterDim.filters;
    ConvLayer<Dtype>* _targetLayer = (ConvLayer<Dtype>*)targetLayer;

    //int blockSize = BW;
    int blockSize = SOOOA_CUDA_NUM_THREADS;
    int gridSize = (weightSize + blockSize -1) / blockSize;

    AddArrayOfConvLayer<<<gridSize, blockSize>>>(
        _targetLayer->_params[Filter]->mutable_device_grad(),
        this->_params[Filter]->device_grad(), weightSize);

    if (this->biasTerm) {
		gridSize = (biasSize + blockSize -1) / blockSize;
		AddArrayOfConvLayer<<<gridSize, blockSize>>>(
			_targetLayer->_params[Bias]->mutable_device_grad(),
			this->_params[Bias]->device_grad(), biasSize);
    }
}

template <typename Dtype>
void ConvLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	const filter_dim& filterDim = SLPROP(Conv, filterDim);
	const uint32_t weightSize = filterDim.size();
    ConvLayer<Dtype>* _targetLayer = (ConvLayer<Dtype>*)targetLayer;

    memcpy(this->_params[Filter]->mutable_host_grad(), _targetLayer->_params[Filter]->host_grad(),
        weightSize);

    if (this->biasTerm) {
		const uint32_t biasSize = filterDim.filters;
		memcpy(this->_params[Bias]->mutable_host_grad(), _targetLayer->_params[Bias]->host_grad(),
			biasSize);
    }
}

template <typename Dtype>
void ConvLayer<Dtype>::feedforward() {
	reshape();
	_computeFiltersConvolutionData();
}

template <typename Dtype>
void ConvLayer<Dtype>::_computeFiltersConvolutionData() {
	// Apply filters to input data
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	const Dtype* d_filtersData = this->_params[Filter]->device_data();
	//Dtype* d_preActivationData = _preActivation->mutable_device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

	const bool deconv = SLPROP(Conv, deconv);
    if (!deconv) {
	    checkCUDNN(cudnnConvolutionForward(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, this->filterDesc, d_filtersData,
            this->convDesc, this->convFwdAlgo, this->d_workspace, this->workspaceSize,
			&Cuda::beta, this->outputTensorDesc, d_outputData));
    } else {
	    checkCUDNN(cudnnConvolutionBackwardData(Cuda::cudnnHandle,
			&Cuda::alpha, this->filterDesc, d_filtersData, this->inputTensorDesc,
            d_inputData,
            this->convDesc, this->convBwdDataAlgo, this->d_workspace, this->workspaceSize,
			&Cuda::beta, this->outputTensorDesc, d_outputData));
    }

    if (this->biasTerm) {
		const Dtype* d_biasesData = this->_params[Bias]->device_data();
		checkCUDNN(cudnnAddTensor(Cuda::cudnnHandle,
				&Cuda::alpha, this->biasTensorDesc, d_biasesData,
				&Cuda::alpha, this->outputTensorDesc, d_outputData));
    }
}


template <typename Dtype>
void ConvLayer<Dtype>::backpropagation() {
	if (SLPROP_BASE(propDown)[0]) {
		//_computePreActivationGrad();
		_computeFiltersGrad();
		if (this->biasTerm) {
			_computeBiasesGrad();
		}
		_computeInputGrad();
	}
}


template <typename Dtype>
void ConvLayer<Dtype>::_computeFiltersGrad() {

	// d(Cost)/d(Filters)
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_filtersGrad = this->_params[Filter]->mutable_device_grad();

	const bool deconv = SLPROP(Conv, deconv);
    if (!deconv) {
	    checkCUDNN(cudnnConvolutionBackwardFilter(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, this->outputTensorDesc,
            d_outputGrad, this->convDesc, this->convBwdFilterAlgo, this->d_workspace,
            this->workspaceSize, &Cuda::beta, this->filterDesc, d_filtersGrad));
    } else {
	    checkCUDNN(cudnnConvolutionBackwardFilter(Cuda::cudnnHandle,
			&Cuda::alpha, this->outputTensorDesc, d_outputGrad, this->inputTensorDesc,
            d_inputData, this->convDesc, this->convBwdFilterAlgo, this->d_workspace,
            this->workspaceSize, &Cuda::beta, this->filterDesc, d_filtersGrad));
    }

}

template <typename Dtype>
void ConvLayer<Dtype>::_computeBiasesGrad() {
	// d(Cost)/d(Biases)
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_biasGrad = this->_params[Bias]->mutable_device_grad();

	checkCUDNN(cudnnConvolutionBackwardBias(Cuda::cudnnHandle,
			&Cuda::alpha, this->outputTensorDesc, d_outputGrad,
			&Cuda::beta, this->biasTensorDesc, d_biasGrad));


}

template <typename Dtype>
void ConvLayer<Dtype>::_computeInputGrad() {
	// d(Cost)/d(Input)
	const Dtype* d_filtersData = this->_params[Filter]->device_data();
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
    Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();

    const bool deconv = SLPROP(Conv, deconv);
    if (!deconv) {
        checkCUDNN(cudnnConvolutionBackwardData(Cuda::cudnnHandle,
			&Cuda::alpha, this->filterDesc, d_filtersData, this->outputTensorDesc,
            d_outputGrad, this->convDesc, this->convBwdDataAlgo, this->d_workspace,
            this->workspaceSize, &Cuda::beta, this->inputTensorDesc, d_inputGrad));
    } else {
	    checkCUDNN(cudnnConvolutionForward(Cuda::cudnnHandle, &Cuda::alpha,
            this->outputTensorDesc, d_outputGrad, this->filterDesc, d_filtersData,
            this->convDesc, this->convFwdAlgo, this->d_workspace, this->workspaceSize,
			&Cuda::beta, this->inputTensorDesc, d_inputGrad));
    }

}

template ConvLayer<float>::ConvLayer();
template ConvLayer<float>::~ConvLayer();
template void ConvLayer<float>::reshape();
template void ConvLayer<float>::update();
template void ConvLayer<float>::feedforward();
template void ConvLayer<float>::backpropagation();
