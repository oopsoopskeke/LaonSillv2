/**
 * @file BatchNormLayer_device.cu
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "BatchNormLayer.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "ColdLog.h"
#include "Perf.h"
#include "MathFunctions.h"
#include "PropMgmt.h"
#include "Update.h"
#include "Updater.h"
#include "Donator.h"

#define BATCHCONDLAYER_LOG  0

using namespace std;

// FIXME: 커널함수들 더 빨리 동작시킬 수 있게 수정 필요 
//        ex. 중간 계산값을 메모리로 들고 있는 방식

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

template <typename Dtype>
__global__ void FillValues(Dtype *vec, int size, Dtype value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	vec[idx] = value;
}

template <typename Dtype>
__global__ void CalcMean(const Dtype *input, int depth, int batchCount, Dtype *mean)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth) 
		return;

    for (int i = 0 ; i < batchCount; i++) {
        int index = i * depth + idx;
        mean[idx] += input[index];
    }


    mean[idx] = mean[idx] / (Dtype)batchCount;
}

template <typename Dtype>
__global__ void CalcVariance(const Dtype *input, const Dtype* mean, int depth, int batchCount,
    Dtype *variance)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth) 
		return;

    for (int i = 0 ; i < batchCount; i++) {
        int index = i * depth + idx;
        variance[idx] += (input[index] - mean[idx]) * (input[index] - mean[idx]);
    }

    variance[idx] = variance[idx] / (Dtype)batchCount;
}


template <typename Dtype>
__global__ void Normalize(const Dtype *input, const Dtype* mean, const Dtype* variance,
    const Dtype* gamma, const Dtype* beta, int depth, int batchCount, Dtype epsilon,
    Dtype* normInput, Dtype* output)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = depth * batchCount;
	if (idx >= count) 
		return;

    int curDepth = idx % depth;
    Dtype denominator = sqrtf(variance[curDepth] + epsilon);

    normInput[idx] = (input[idx] - mean[curDepth]) / denominator;
    output[idx] = normInput[idx] * gamma[curDepth] + beta[curDepth];
}

#define USE_SIMPLE_MOVING_AVERAGE       1
template <typename Dtype>
__global__ void IncrementalMean(const Dtype *input, int depth, const Dtype counter,
    Dtype* output)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth) 
		return;
#if USE_SIMPLE_MOVING_AVERAGE
    output[idx] = 0.99 * output[idx] + 0.01 * input[idx];
#else
    output[idx] = ((counter - 1.0) * output[idx] + input[idx]) / counter;
#endif
}

template <typename Dtype>
__global__ void Inference(const Dtype *input, const Dtype *globalMean,
    const Dtype *globalVar, const Dtype *gamma, const Dtype *beta, int depth,
    int batchCount, const Dtype counter, Dtype epsilon, Dtype* output)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth)
		return;

    Dtype varFactor = 1.0;

    if (counter > 1.1) {
        varFactor = (Dtype)counter / ((Dtype)counter - 1.0);
    }

    Dtype sqrtVariance = sqrtf(globalVar[idx] * varFactor + epsilon);

    for (int i = 0 ; i < batchCount; i++) {
        int index = i * depth + idx;

        output[index] = input[index] * gamma[idx] / sqrtVariance + beta[idx] - 
            gamma[idx] * globalMean[idx] / sqrtVariance;
    }
}

template <typename Dtype>
__global__ void ComputeNormInputGrad(const Dtype *outputGrads, const Dtype *gammas, int depth,
    int batchCount, Dtype* normInputGrads)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = depth * batchCount;
	if (idx >= count) 
		return;
    int curDepth = idx % depth;

    normInputGrads[idx] = outputGrads[idx] * gammas[curDepth];
}

template <typename Dtype>
__global__ void ComputeVarianceGrad(const Dtype* normInputGrad, const Dtype *inputData, 
    const Dtype *mean, const Dtype *variance, Dtype epsilon, int depth, int batchCount,
    Dtype* varianceGrad)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth) 
		return;

    varianceGrad[idx] = 0;
    Dtype poweredVar = (-0.5) * pow((variance[idx] + epsilon), -1.5);
    for (int i = 0; i < batchCount; i++) {
        int index = i * depth + idx;
        varianceGrad[idx] += normInputGrad[index] * (inputData[index] - mean[idx]) * 
            poweredVar;
    }
}

template <typename Dtype>
__global__ void ComputeMeanGrad(const Dtype *normInputGrads, const Dtype *vars,
    const Dtype *varGrads, const Dtype* inputData, const Dtype* means, int depth,
    int batchCount, Dtype epsilon, Dtype* meanGrads)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth) 
		return;

    meanGrads[idx] = 0;
    Dtype sqrtVar = (-1) / sqrtf(vars[idx] + epsilon);
    Dtype varGradFactor = varGrads[idx] * (-2) / (Dtype)batchCount;
    for (int i = 0; i < batchCount; i++) {
        int index = i * depth + idx;
        meanGrads[idx] += normInputGrads[index] * sqrtVar +
            varGradFactor * (inputData[index] - means[idx]);
    }
}

template <typename Dtype>
__global__ void ComputeInputGrad(const Dtype *normInputGrads, const Dtype *vars,
    const Dtype *varGrads, const Dtype* inputData, const Dtype* means, const Dtype* meanGrads,
    int depth, int batchCount, Dtype epsilon, Dtype* inputGrads)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth) 
		return;

    Dtype sqrtVar = sqrtf(vars[idx] + epsilon);
    Dtype varGradFactor = varGrads[idx] * 2 / (Dtype)batchCount;
    Dtype meanFactor = meanGrads[idx] / (Dtype)batchCount;
    for (int i = 0; i < batchCount; i++) {
        int index = i * depth + idx;
        inputGrads[index] = normInputGrads[index] / sqrtVar +
            varGradFactor * (inputData[index] - means[idx]) + meanFactor;
    }
}

template <typename Dtype>
__global__ void ComputeScaleGrad(const Dtype *normInputs, const Dtype *outputGrads,
    int depth, int batchCount, Dtype* gammaGrads)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth) 
		return;

    gammaGrads[idx] = 0;
    for (int i = 0; i < batchCount; i++) {
        int index = i * depth + idx;
        gammaGrads[idx] += outputGrads[index] * normInputs[index];
    }
}

template <typename Dtype>
__global__ void ComputeShiftGrad(const Dtype *outputGrads, int depth, int batchCount,
    Dtype* betaGrads)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth) 
		return;

    betaGrads[idx] = 0;
    for (int i = 0; i < batchCount; i++) {
        int index = i * depth + idx;
        betaGrads[idx] += outputGrads[index];
    }
}

template<typename Dtype>
BatchNormLayer<Dtype>::~BatchNormLayer() {
    if (SLPROP_BASE(receive)) {
        Donator<Dtype>::releaseReceiver(SLPROP_BASE(donatorID));
    } else {
        Util::clearVector(this->_params);
        Util::clearVector(this->_paramsHistory);
        Util::clearVector(this->_paramsHistory2);
    }

    SASSERT0(this->normInputSet != NULL);
    SFREE(this->normInputSet);

    this->updateParams.clear();
}

template <typename Dtype>
void BatchNormLayer<Dtype>::update() {
    const uint32_t size = this->depth;
	const Dtype regScale = SNPROP(weightDecay);
	const Dtype learnScale = Update<float>::calcLearningRate();
    const Dtype beta1 = SNPROP(beta1);
    const Dtype beta2 = SNPROP(beta2);

    SLPROP(BatchNorm, decayedBeta1) *= beta1;
    SLPROP(BatchNorm, decayedBeta2) *= beta2;

    UpdateContext context = Update<Dtype>::makeContext(size, regScale, learnScale);

    SASSUME0(this->updateParams.size() == 2);
    this->updateParams[Gamma].context = context;
    this->updateParams[Beta].context = context;

    Updater::updateParams(this->updateParams);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::feedforward() {
    reshape();
    struct timespec startTime;
    SPERF_START(BATCHNORM_LAYER_FWTIME, &startTime);

    // FIXME: 현재 CPU 코드로 구현이 되어 있다. GPU 코드로 변경하자.
    // (1) mini-batch mean 값을 구한다.
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];

    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* outputData = this->_outputData[0]->mutable_device_data();

	if (SLPROP(BatchNorm, train)) {
        Dtype* means = this->meanSet->mutable_device_data();
        Dtype* vars = this->varSet->mutable_device_data();

        // (1) mini-batch에 사용하는 mean, variance를 초기화 한다.
        FillValues<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            means, this->depth, 0.0f);
        FillValues<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            vars, this->depth, 0.0f);

        // (2) mini-batch mean 값을 구한다.
        CalcMean<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            inputData, this->depth, batchCount, means);

        // (3) mini-batch variance 값을 구한다.
        CalcVariance<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            inputData, means, this->depth, batchCount, vars);

        // (4) normalize 
        Dtype* normInputs = this->normInputSet->mutable_device_data();
        const Dtype* gammas = this->_params[ParamType::Gamma]->device_data();
        const Dtype* betas = this->_params[ParamType::Beta]->device_data();
        Normalize<<<SOOOA_GET_BLOCKS(this->depth * batchCount), SOOOA_CUDA_NUM_THREADS>>>(
            inputData, means, vars, gammas, betas, this->depth, batchCount,
            (Dtype)SLPROP(BatchNorm, epsilon), normInputs, outputData);

        // (5) global meanSets과 varianceSets를 갱신한다.
        Dtype* counter = this->_params[ParamType::GlobalCount]->mutable_host_data();
        counter[0] += 1;

        Dtype* globalMeans = this->_params[ParamType::GlobalMean]->mutable_device_data();
        Dtype* globalVars = this->_params[ParamType::GlobalVar]->mutable_device_data();
        IncrementalMean<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            means, this->depth, counter[0], globalMeans);
        IncrementalMean<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            vars, this->depth, counter[0], globalVars);
	} else {
        const Dtype* counter = this->_params[ParamType::GlobalCount]->host_data();
        SASSERT((counter[0] > 0), "need train before inference");

        const Dtype* globalMeans = this->_params[ParamType::GlobalMean]->device_data();
        const Dtype* globalVars = this->_params[ParamType::GlobalVar]->device_data();
        const Dtype* gammas = this->_params[ParamType::Gamma]->device_data();
        const Dtype* betas = this->_params[ParamType::Beta]->device_data();

        Inference<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            inputData, globalMeans, globalVars, gammas, betas, this->depth, batchCount,
            counter[0], (Dtype)SLPROP(BatchNorm, epsilon), outputData);
    }

    SPERF_END(BATCHNORM_LAYER_FWTIME, startTime);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
		const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
		const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
		assert(count == inputDataCount);
	}

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();

	uint32_t batches = inputShape[0];
	uint32_t channels = inputShape[1];
	uint32_t rows = inputShape[2];
	uint32_t cols = inputShape[3];
    uint32_t depth = this->_inputData[0]->getCountByAxis(1);

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({batches, channels, rows, cols});

	STDOUT_COND_LOG(BATCHCONDLAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);
	STDOUT_COND_LOG(BATCHCONDLAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), batches, channels, rows, cols);

    // Batch Normalization 과정에 필요한 구조체들의 메모리를 할당한다.
    if (this->depth == 0) {
        this->depth = depth;

        Optimizer opt = (Optimizer)SNPROP(optimizer);
        int paramHistoryDataCount = Update<Dtype>::getParamHistoryDataCount(opt);

        this->_params[ParamType::Gamma]->reshape({1, channels, rows, cols});
        this->_params[ParamType::Beta]->reshape({1, channels, rows, cols});
        this->_params[ParamType::GlobalMean]->reshape({1, channels, rows, cols});
        this->_params[ParamType::GlobalVar]->reshape({1, channels, rows, cols});
        this->_params[ParamType::GlobalCount]->reshape({1, 1, 1, 1});

        if (paramHistoryDataCount >= 1) {
            this->_paramsHistory[ParamType::Gamma]->reshape({1, channels, rows, cols});
            this->_paramsHistory[ParamType::Beta]->reshape({1, channels, rows, cols});
            this->_paramsHistory[ParamType::GlobalMean]->reshape({1, channels, rows, cols});
            this->_paramsHistory[ParamType::GlobalVar]->reshape({1, channels, rows, cols});
            this->_paramsHistory[ParamType::GlobalCount]->reshape({1, 1, 1, 1});
        }

        if (paramHistoryDataCount >= 2) {
            this->_paramsHistory2[ParamType::Gamma]->reshape({1, channels, rows, cols});
            this->_paramsHistory2[ParamType::Beta]->reshape({1, channels, rows, cols});
            this->_paramsHistory2[ParamType::GlobalMean]->reshape({1, channels, rows, cols});
            this->_paramsHistory2[ParamType::GlobalVar]->reshape({1, channels, rows, cols});
            this->_paramsHistory2[ParamType::GlobalCount]->reshape({1, 1, 1, 1});
        }

        this->meanSet->reshape({1, channels, rows, cols});
        this->varSet->reshape({1, channels, rows, cols});

        this->normInputSet->reshape({batches, channels, rows, cols});

        // FIXME: 더 좋은 초기화 방법이 있을지도 모른다..
        Dtype* gammas = this->_params[ParamType::Gamma]->mutable_device_data();
        FillValues<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            gammas, this->depth, 1.0f);
        this->_paramsInitialized[ParamType::Gamma] = true;

        Dtype* betas = this->_params[ParamType::Beta]->mutable_device_data();
        FillValues<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            betas, this->depth, 0.0f);
        this->_paramsInitialized[ParamType::Beta] = true;

        Dtype* globalMeans = this->_params[ParamType::GlobalMean]->mutable_device_data();
        FillValues<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            globalMeans, this->depth, 0.0f);
        this->_paramsInitialized[ParamType::GlobalMean] = true;

        Dtype* globalVars = this->_params[ParamType::GlobalVar]->mutable_device_data();
        FillValues<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            globalVars, this->depth, 1.0f);
        this->_paramsInitialized[ParamType::GlobalVar] = true;

        Dtype* globalCounts = this->_params[ParamType::GlobalCount]->mutable_device_data();
        FillValues<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
            globalCounts, this->depth, 0.0f);
        this->_paramsInitialized[ParamType::GlobalCount] = true;
    } else {
        SASSERT0(this->depth == depth);
    }

    if (this->updateParams.size() == 0) {
        UpdateParam upGamma;
        upGamma.paramType = Gamma;
        upGamma.paramDataPtr = (void*)this->_params[Gamma];
        upGamma.paramHis1Ptr = (void*)this->_paramsHistory[Gamma];
        upGamma.paramHis2Ptr = (void*)this->_paramsHistory2[Gamma];
        this->updateParams.push_back(upGamma);

        UpdateParam upBeta;
        upBeta.paramType = Beta;
        upBeta.paramDataPtr = (void*)this->_params[Beta];
        upBeta.paramHis1Ptr = (void*)this->_paramsHistory[Beta];
        upBeta.paramHis2Ptr = (void*)this->_paramsHistory2[Beta];
        this->updateParams.push_back(upBeta);
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeNormInputGrad() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];

    const Dtype* outputGrads = this->_outputData[0]->device_grad();
    Dtype* normInputGrads = this->normInputSet->mutable_device_grad();
    const Dtype* gammas = this->_params[ParamType::Gamma]->device_data();

    ComputeNormInputGrad<<<SOOOA_GET_BLOCKS(this->depth * batchCount),
        SOOOA_CUDA_NUM_THREADS>>>(
        outputGrads, gammas, this->depth, batchCount, normInputGrads);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeVarianceGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* varGrads = this->varSet->mutable_device_grad();
    const Dtype* normInputGrads = this->normInputSet->device_grad();
    const Dtype* means = this->meanSet->device_data();
    const Dtype* vars = this->varSet->device_data();

    ComputeVarianceGrad<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
        normInputGrads, inputData, means, vars, (Dtype)SLPROP(BatchNorm, epsilon), depth, batchCount,
        varGrads);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeMeanGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* meanGrads = this->meanSet->mutable_device_grad();
    const Dtype* normInputGrads = this->normInputSet->device_grad();
    const Dtype* vars = this->varSet->device_data();
    const Dtype* varGrads = this->varSet->device_grad();
    const Dtype* means = this->meanSet->device_data();

    ComputeMeanGrad<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
        normInputGrads, vars, varGrads, inputData, means, depth, batchCount,
        (Dtype)SLPROP(BatchNorm, epsilon), meanGrads);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeInputGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* inputData = this->_inputData[0]->device_data();
	Dtype* inputGrads = this->_inputData[0]->mutable_device_grad();
    const Dtype* normInputGrads = this->normInputSet->device_grad();
    const Dtype* vars = this->varSet->device_data();
    const Dtype* varGrads = this->varSet->device_grad();
    const Dtype* means = this->meanSet->device_data();
    const Dtype* meanGrads = this->meanSet->device_grad();

    ComputeInputGrad<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
        normInputGrads, vars, varGrads, inputData, means, meanGrads, depth, batchCount,
        (Dtype)SLPROP(BatchNorm, epsilon), inputGrads);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeScaleGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* outputGrads = this->_outputData[0]->device_grad();;
    Dtype* gammaGrads = this->_params[ParamType::Gamma]->mutable_device_grad();
    const Dtype* normInputs = this->normInputSet->device_data();

    ComputeScaleGrad<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
        normInputs, outputGrads, depth, batchCount, gammaGrads);
   
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeShiftGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* outputGrads = this->_outputData[0]->device_grad();
    Dtype* betaGrads = this->_params[ParamType::Beta]->mutable_device_grad();

    ComputeShiftGrad<<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
        outputGrads, depth, batchCount, betaGrads);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::backpropagation() {
    struct timespec startTime;
    SPERF_START(BATCHNORM_LAYER_BWTIME, &startTime);
    /*
     * 아래와 같은 simple한 network layer가 있다고 가정하자.
     *
     *               <<<< ith layer >>>>                        <<<< i+1th layer >>>>
     *   .....    Xi  Norm    ^Xi   γi * ^Xi + βi      Yi (=Xi+1)  ........
     *   .....    O ---------  O  ---------------------  O         ........
     *                                                     dL/dYi is already computed
     *
     *  (※  Xi = i번째 layer의 input 값, Norm = normaliztion
     *      ^Xi = i번째 layer의 중간 값, γi = scale factor, βi = shift factor
     *      Yi = i번째 layer의 ouput 값, i+1 번째 layer의 input 값이기도 함
     *      L = loss, dL/dYi = i+1번째 layer에서 계산되었던 gradient 값)
     *
     *  BatchNormLayer에서는 γi, βi를 학습해야 하는데 그것을 위해서 dL/dγi, dL/dβi를 계산해야
     *  한다. 또한, 하위 layer에 전달할 dL/dXi이 필요하다.
     *
     *  논문(https://arxiv.org/abs/1502.03167)에서 각각의 계산식이 있기 때문에 그것을 이용하여
     *  연산을 하도록 하자.)
     */

    // (1) dL/d^Xi = dL/dYi * γi
    computeNormInputGrad();

    // (2) dL/dSquaredSigma
    computeVarianceGrad();

    // (3) dL/dMean
    computeMeanGrad();

    // (4) dL/dXi
    computeInputGrad();

    // (5) dL/dγi
    computeScaleGrad();

    // (6) dL/dβi
    computeShiftGrad();

    SPERF_END(BATCHNORM_LAYER_BWTIME, startTime);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
    return;
}

template <typename Dtype>
void BatchNormLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
    return;
}

template BatchNormLayer<float>::~BatchNormLayer();
template void BatchNormLayer<float>::reshape();
template void BatchNormLayer<float>::update();
template void BatchNormLayer<float>::feedforward();
template void BatchNormLayer<float>::backpropagation();
template void BatchNormLayer<float>::applyChanges(LearnableLayer<float> *targetLayer);
template void BatchNormLayer<float>::syncParams(LearnableLayer<float> *targetLayer);

