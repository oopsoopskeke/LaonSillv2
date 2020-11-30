/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "cuda_runtime.h"
#include <algorithm>

#include "FullyConnectedLayer.h"
#include "MathFunctions.h"
#include "Util.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "PropMgmt.h"
#include "Update.h"
#include "Updater.h"
#include "Donator.h"
#include "frcnn_common.h"
#include "MemoryMgmt.h"

#define FULLYCONNECTEDLAYER_LOG 0

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
template <typename Dtype>
__global__ void FillValues(Dtype *vec, int size, Dtype value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	vec[idx] = value;
}

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
template <typename Dtype>
__global__ void Dropout(const int n, const Dtype* in, const Dtype* mask,
		const unsigned int threashold, const float scale, Dtype *out)
{

	CUDA_KERNEL_LOOP(index, n) {
		//out[index] = in[index] * (mask[index] > threshold) * scale;
		out[index] = in[index] * (mask[index]) * scale;
	}
}

/**
 * dst array에 src array를 더한다.
 *
 * @param dst dst array, dst + src가 저장이 될 장소
 * @param src src array
 * @param N The number of elements in the array.
 */
template <typename Dtype>
__global__ void AddData(Dtype* dst, const Dtype* src, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	dst[idx] = dst[idx] + src[idx]; 
}

template <typename Dtype>
FullyConnectedLayer<Dtype>::~FullyConnectedLayer() {
    if (SLPROP(FullyConnected, receive)) {
        Donator<Dtype>::releaseReceiver(SLPROP(FullyConnected, donatorID));
    } else {
        Util::clearVector(this->_params);
        Util::clearVector(this->_paramsHistory);
        Util::clearVector(this->_paramsHistory2);
    }
    this->updateParams.clear();
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
		const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
		const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
		SASSERT0(count == inputDataCount);
	}

	/*
	// 배치수가 변경되는 경우는 허용하도록 하자.
	const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
	const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
	if (inputDataCount == count)
		return;
		*/

	// XXX: 주의

    // 여기에서는 batch 개수만 변경이 될 수 있다고 가정하였다.
    // 따라서 batch 개수에 대한 변경만 체크한다.
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	this->batches = this->_inputData[0]->getShape(0);
	this->in_rows = this->_inputData[0]->getCountByAxis(SLPROP(FullyConnected, axis));
	this->out_rows = SLPROP(FullyConnected, nOut);

	const uint32_t channels = 1;
	const uint32_t cols = 1;

	//this->_inputShape[0] = {batches, channels, in_rows, cols};
	this->_inputShape[0] = this->_inputData[0]->getShape();
	this->_outputData[0]->reshape({this->batches, channels, this->out_rows, cols});

	/*
	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			this->batches, channels, this->in_rows, cols));

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			this->batches, channels, this->out_rows, cols));
			*/

	STDOUT_COND_LOG(FULLYCONNECTEDLAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), this->batches, channels, this->in_rows, cols);
	STDOUT_COND_LOG(FULLYCONNECTEDLAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), this->batches, channels, this->out_rows, cols);

	const uint32_t u_in = in_rows;
	const uint32_t u_out = out_rows;
	const uint32_t b_in = batches * in_rows;
	const uint32_t b_out = batches * out_rows;

	STDOUT_COND_LOG(FULLYCONNECTEDLAYER_LOG,
	    "<%s> layer reshape info (u_in, u_out, b_in, b_out) : %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), u_in, u_out, b_in, b_out);

	this->_params[ParamType::Weight]->reshape({1, 1, u_out, u_in});
	this->_params[ParamType::Bias]->reshape({1, u_out, 1, 1});

    if (this->_paramsHistory[ParamType::Weight] != NULL)
	    this->_paramsHistory[ParamType::Weight]->reshape({1, 1, u_out, u_in});
	if (this->_paramsHistory[ParamType::Bias] != NULL)
	    this->_paramsHistory[ParamType::Bias]->reshape({1, u_out, 1, 1});
	if (this->_paramsHistory2[ParamType::Weight] != NULL)
	    this->_paramsHistory2[ParamType::Weight]->reshape({1, 1, u_out, u_in});
	if (this->_paramsHistory2[ParamType::Bias] != NULL)
	    this->_paramsHistory2[ParamType::Bias]->reshape({1, u_out, 1, 1});

	if (!this->_paramsInitialized[Weight]) {
        SLPROP(FullyConnected, weightFiller).fill(this->_params[ParamType::Weight]);
		this->_paramsInitialized[Weight] = true;
	}
	if (!this->_paramsInitialized[Bias]) {
        SLPROP(FullyConnected, weightFiller).fill(this->_params[ParamType::Bias]);
		this->_paramsInitialized[Bias] = true;
	}

    if (this->updateParams.size() == 0) {
        UpdateParam upWeight;
        upWeight.paramType = Weight;
        upWeight.paramDataPtr = (void*)this->_params[Weight];
        upWeight.paramHis1Ptr = (void*)this->_paramsHistory[Weight];
        upWeight.paramHis2Ptr = (void*)this->_paramsHistory2[Weight];
        this->updateParams.push_back(upWeight);

        UpdateParam upBias;
        upBias.paramType = Bias;
        upBias.paramDataPtr = (void*)this->_params[Bias];
        upBias.paramHis1Ptr = (void*)this->_paramsHistory[Bias];
        upBias.paramHis2Ptr = (void*)this->_paramsHistory2[Bias];
        this->updateParams.push_back(upBias);
    }

    this->_onevec.reshape(this->batches);
    this->_onevec.reset_host_mem(false, 1.0);
	//checkCudaErrors(Util::ucudaMalloc(&this->d_onevec, sizeof(Dtype)*batches));
	//FillValues<<<SOOOA_GET_BLOCKS(batches), SOOOA_CUDA_NUM_THREADS>>>(
	//		this->d_onevec, batches, 1.0f);

	this->_mask.reshape(b_out);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::update() {
	const uint32_t weightSize = this->in_rows * this->out_rows;
	const Dtype regScale =
        SNPROP(weightDecay) * SLPROP(FullyConnected, weightUpdateParam).decay_mult;
	const Dtype learnScale = Update<Dtype>::calcLearningRate() *
		SLPROP(FullyConnected, weightUpdateParam).lr_mult;

    const Dtype beta1 = SNPROP(beta1);
    const Dtype beta2 = SNPROP(beta2);

    SLPROP(FullyConnected, decayedBeta1) *= beta1;
    SLPROP(FullyConnected, decayedBeta2) *= beta2;

    UpdateContext contextWeight = 
        Update<Dtype>::makeContext(weightSize, regScale, learnScale);



	const uint32_t biasSize = out_rows;
	const Dtype regScale_b = 
        SNPROP(weightDecay) * SLPROP(FullyConnected, biasUpdateParam).decay_mult;
	const Dtype learnScale_b = Update<Dtype>::calcLearningRate() *
        SLPROP(FullyConnected, biasUpdateParam).lr_mult;

    UpdateContext contextBias = 
        Update<Dtype>::makeContext(biasSize, regScale_b, learnScale_b);

    SASSUME0(this->updateParams.size() == 2);
    this->updateParams[Weight].context = contextWeight;
    this->updateParams[Bias].context = contextBias;

	Updater::updateParams(this->updateParams);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	//const uint32_t in_rows = this->_inputShape[0][2];
	//const uint32_t out_rows = this->_outputData[0]->getShape(2);

    const uint32_t weightSize = this->in_rows * this->out_rows;
    const uint32_t biasSize = this->out_rows;
    FullyConnectedLayer<Dtype>* _targetLayer = (FullyConnectedLayer<Dtype>*)targetLayer;

    //int blockSize = BW;
    int blockSize = SOOOA_CUDA_NUM_THREADS;
    int gridSize;

    gridSize = (weightSize + blockSize -1) / blockSize;

    AddData<<<gridSize, blockSize>>>(
        _targetLayer->_params[Weight]->mutable_device_grad(),
        this->_params[Weight]->device_grad(), weightSize);

    gridSize = (biasSize + blockSize -1) / blockSize;

    AddData<<<gridSize, blockSize>>>(
        _targetLayer->_params[Bias]->mutable_device_grad(),
        this->_params[Bias]->device_grad(), biasSize);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	//const uint32_t in_rows = this->_inputShape[0][2];
	//const uint32_t out_rows = this->_outputData[0]->getShape(2);

    const uint32_t weightSize = this->in_rows * this->out_rows;
    const uint32_t biasSize = this->out_rows;
    FullyConnectedLayer<Dtype>* _targetLayer = (FullyConnectedLayer<Dtype>*)targetLayer;

    memcpy(this->_params[Weight]->mutable_host_grad(), _targetLayer->_params[Weight]->host_grad(),
        weightSize);
    memcpy(this->_params[Bias]->mutable_host_grad(), _targetLayer->_params[Bias]->host_grad(),
        biasSize);
}




template <typename Dtype>
void FullyConnectedLayer<Dtype>::saveParams(ofstream& ofs) {
	LearnableLayer<Dtype>::saveParams(ofs);
	/*
	if (this->_inputData.size() == 1) {
		cout << SLPROP_BASE(name) << " saves as usual ... " << endl;
		LearnableLayer<Dtype>::saveParams(ofs);
	} else {
		cout << SLPROP_BASE(name) << " saves as special ... " << endl;
		uint32_t numParams = this->_params.size();

		vector<vector<float>> bboxMeans;
		vector<vector<float>> bboxStds;
		fill2dVecWithData(this->_inputData[1], bboxMeans);
		fill2dVecWithData(this->_inputData[2], bboxStds);


#if 0
		this->_inputData[1]->print_shape();
		this->_inputData[2]->print_shape();
		this->_params[0]->print_shape();
		this->_params[1]->print_shape();
		exit(1);
#endif



		Data<Dtype>* param0 = this->_params[0];
		Data<Dtype> orig0(param0->_name, true);
		orig0.reshapeLike(param0);

		const Dtype* srcPtr0 = param0->host_data();
		Dtype* dstPtr0 = orig0.mutable_host_data();

		const int numRows0 = param0->getShape(2);
		const int numCols0 = param0->getShape(3);
		int index;
		int id1, id2;
		for (int row = 0; row < numRows0; row++) {
			id2 = row / 4;
			id1 = row % 4;
			for (int col = 0; col < numCols0; col++) {
				index = row * numCols0 + col;
				dstPtr0[index] = srcPtr0[index] * bboxStds[id2][id1];
			}
		}



		Data<Dtype>* param1 = this->_params[1];
		Data<Dtype> orig1(param1->_name, true);
		orig1.reshapeLike(param1);

		const Dtype* srcPtr1 = param1->host_data();
		Dtype* dstPtr1 = orig1.mutable_host_data();

		const int numRows1 = param1->getShape(1);
		for (int row = 0; row < numRows1; row++) {
			id2 = row / 4;
			id1 = row % 4;
			dstPtr1[row] = srcPtr1[row] * bboxStds[id2][id1] + bboxMeans[id2][id1];
		}
		orig0.save(ofs);
		orig1.save(ofs);
	}
	*/
}













template <typename Dtype>
void FullyConnectedLayer<Dtype>::feedforward() {
	reshape();

	_computeWeightedData();
	_computeWeightBiasedData();
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightedData() {
	// Apply weight to input data
	const Dtype* d_weightData = this->_params[Weight]->device_data();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	//Dtype* d_preActivationData = _preActivation->mutable_device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

    /**
     * [cublasSgemm() 함수 설명 (from cuBlas User Documentation)]
     *
     * cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
     *                            cublasOperation_t transb, int m, int n, int k, 
     *                            const float *alpha, const float *A, int * lda, 
     *                            const float *B, int ldb, const float *beta, float *C, 
     *                            int ldc)
     *
     * C = α op ( A ) op ( B ) + β C
     *
     * where α and β are scalars, and A , B and C are matrices stored in column-major format
     * with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively. Also, for
     * matrix A 
     *
     * op ( A ) = A if  transa == CUBLAS_OP_N A T if  transa == CUBLAS_OP_T A H if  transa ==
     * CUBLAS_OP_C
     *
     * and op ( B ) is defined similarly for matrix B .
     *
     * cublasOperation_t option
     *  (1) CUBLAS_OP_N => the non-transpose operation is selected.
     *  (2) CUBLAS_OP_T => the transpose operation is selected.
     *  (3) CUBLAS_OP_C => the conjugate transpose operation is selected.
     *
     * lda,ldb,ldc => leading dimension of two-dimensional array used to store the matrix A,
     *                B, C
     */

	if (this->batches == 1) {
		soooa_gpu_gemv(CblasNoTrans,
				this->out_rows, this->in_rows,
				Cuda::alpha, d_weightData, d_inputData,
				Cuda::beta, d_outputData);

	} else {
		soooa_gpu_gemm(CblasNoTrans, CblasTrans,
				this->batches, this->out_rows, this->in_rows,
				Cuda::alpha, d_inputData, d_weightData,
				Cuda::beta, d_outputData);
	}
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightBiasedData() {
	// Add bias to weighted input data
	const Dtype* d_biasData = this->_params[Bias]->device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

	if (this->batches == 1) {
		soooa_gpu_axpy(this->out_rows, 1.0f,  d_biasData, d_outputData);
	} else {
		soooa_gpu_gemm(CblasNoTrans, CblasNoTrans,
				this->batches, this->out_rows,	1,
				Cuda::alpha, this->_onevec.device_mem(), d_biasData,
				Cuda::alpha, d_outputData);
	}
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::backpropagation() {
    /*
     * 아래와 같은 simple한 network layer가 있다고 가정하자.
     *
     *               <<<< ith layer >>>>            <<<< i+1th layer >>>>
     *   .....    Xi    Wi     Ai     Fi       Yi (=Xi+1)   ........
     *                  Bi
     *   .....    O ---------  O  ------------  O            ........
     *                                                     dL/dYi is already computed
     *
     *  (※  Xi = i번째 layer의 input 값, Wi = i번째 layer의 weight, 
     *      Bi = i번째 layer의 bias 값,  Ai = i번째 layer의 중간 값
     *      Fi = i번째 layer의 activation function
     *      Yi = i번째 layer의 ouput 값, i+1 번째 layer의 input 값이기도 함
     *      L = loss, dL/dYi = i+1번째 layer에서 계산되었던 gradient 값)
     *
     *  gradient descent 방식으로 학습을 하기 위해서는 dL/dWi & dL/dBi가 필요하다.
     *  체인 룰에 의하여 아래와 같은 식으로 표현이 된다:
     *  (가) dYi/dWi = dL/dYi * dYi/dAi * dAi/dWi
     *  (나) dYi/dBi = dL/dYi * dYi/dAi * dAi/dBi
     *
     *  (가),(나)를 계산하기 위해서는 아래와 같이 4가지 계산이 필요하다.
     *
     *  (A) dL/dYi : i+1번째 layer의 backward 과정에서 _outputData[0]의 grad에 값을 저장해
     *                두었다.
     *
     *  (B) dYi/dAi : _computePreActivationGrad() 에서 dL/dYi * dYi/dAi의 계산을  수행 한다. 
     *                dL/dYi는 구해져 있기 때문에 Yi, Ai 값이 필요하다. 이 값들은 forward시에
     *                각각 _outputData[0]의 data와 _preActivation의 data에 저장이 되어 있다.
     *                activation function에 맞게 Yi, Ai, dL/dYi를 입력값으로 하여 dL/dYi * 
     *                dYi/dAi 값이 계산이 되고, 결과값은 this->_preActivation의 grad에 담는다.
     *
     *  (C) dAi/dWi : _computeWeightGrad()에서 (A), (B)의 결과를 조합하여 weight Grad를
     *               계산한다. dAi/dWi는 실제로 transpose Xi이기 때문에 GEMM 연산만 진행
     *               한다. 결과값은 _params[Weight]의 grad에 저장된다.
     *
     *  (D) dAi/dBi : (C)과정과 동일하다. _computeBiasGrad()에서 bias를 계산하고, 그 결과 값을
     *                _params[Bias]의 grad에 저장을 하는 것만 다르다.
     *
     *  마지막으로 i-1 layer에게 dL/dYi-1값을 전달해야 한다. 이 과정은 _computeInputGrad()
     *  에서 수행이 된다. 결과값을 _inputData의 grad에 저장한다. dL/dYi-1 = dL/dXi =
     *   dL/dAi * dAi/dXi가 된다. dL/dAi는 _preAcitvation의 grad에 저장이 되어 있고, dAi/dXi는
     *  Wi의 transpose 이기 때문에 계산가능하다.
     */

	_computeWeightGrad();
	_computeBiasGrad();
	_computeInputGrad();
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightGrad() {
	// d(Cost)/d(Weight)
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_weightGrad = this->_params[Weight]->mutable_device_grad();

	// d_weightGrad에 Cuda::alpha를 적용하는 것은 아마도 snapshot diff 기능 때문인듯 보임.
	// SoooA에서는 일부러 reset하고 아래 Cuda::alpha를 적용하는 것보다는
	// reset없이 Cuda::beta를 적용하는 것이 나아 보임 ...
	// _computeBiasGrad에도 동일.
	soooa_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			this->out_rows, this->in_rows, this->batches,
			Cuda::alpha, d_outputGrad, d_inputData,
			Cuda::alpha, d_weightGrad);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeBiasGrad() {
	// d(Cost)/d(Bias) (same as d_preActivationGrad)
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_biasGrad = this->_params[Bias]->mutable_device_grad();

	soooa_gpu_gemv<Dtype>(CblasTrans,
			this->batches, this->out_rows,
			Cuda::alpha, d_outputGrad, this->_onevec.device_mem(),
			Cuda::alpha, d_biasGrad);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeInputGrad() {
	//const uint32_t batches = this->_inputShape[0][0];
	//const uint32_t in_rows = this->_inputShape[0][2];
	//const uint32_t out_rows = this->_outputData[0]->getShape(2);

	// d(Cost)/d(Input)
	const Dtype* d_weightData = this->_params[Weight]->device_data();
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();

	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			this->batches, this->in_rows, this->out_rows,
			Cuda::alpha, d_outputGrad, d_weightData,
			Cuda::beta, d_inputGrad);

	this->_inputData[0]->print_grad("inputGrad:");
}

template FullyConnectedLayer<float>::~FullyConnectedLayer();
template void FullyConnectedLayer<float>::reshape();
template void FullyConnectedLayer<float>::update();
template void FullyConnectedLayer<float>::feedforward();
template void FullyConnectedLayer<float>::backpropagation();


/*
template void* FullyConnectedLayer<float>::initLayer();
template void FullyConnectedLayer<float>::destroyLayer(void* instancePtr);
template void FullyConnectedLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool FullyConnectedLayer<float>::allocLayerTensors(void* instancePtr);
template void FullyConnectedLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void FullyConnectedLayer<float>::backwardTensor(void* instancePtr);
template void FullyConnectedLayer<float>::learnTensor(void* instancePtr);
*/
