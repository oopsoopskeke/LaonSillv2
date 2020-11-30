/*
 * ScaleLayer.cpp
 *
 *  Created on: Jan 6, 2018
 *      Author: jkim
 */

#include "ScaleLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"
#include "Updater.h"
#include "PlanParser.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
ScaleLayer<Dtype>::ScaleLayer()
: LearnableLayer<Dtype>(),
  sumMultiplier("sumMultiplier"), sumResult("sumResult"), temp("temp") {
	this->type = Layer<Dtype>::Scale;

	this->biasLayer = NULL;
	// assume that axis is always canonical
	this->axis = SLPROP(Scale, axis);
	const int numAxes = SLPROP(Scale, numAxes);
	SASSERT(numAxes >= -1, "numAxes must be non-negative,"
			"or -1 to extend to the end of input[0]");

	this->biasTerm = SLPROP(Scale, biasTerm);

	this->_params.resize(1);
	this->_paramsHistory.resize(1);
	this->_paramsHistory2.resize(1);
	this->_paramsInitialized.resize(1);

	LearnableLayer<Dtype>::initParam(0, "scale");
	this->updatePolicies.resize(this->biasTerm ? 2 : 1);

	// XXX: updateParams.size()가 1로 시작,
	// clear해준다.
	this->updateParams.clear();
}

template <typename Dtype>
ScaleLayer<Dtype>::~ScaleLayer() {
	if (this->biasLayer != NULL) {
		SDELETE(this->biasLayer);
	}

	//LearnableLayer<Dtype>::releaseParam(0);

	//this->_params.resize(1);
	//this->_paramsHistory.resize(1);
	//this->_paramsHistory2.resize(1);
}


template <typename Dtype>
void ScaleLayer<Dtype>::reshape() {
	const int numAxes = SLPROP(Scale, numAxes);

	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		const vector<uint32_t>::const_iterator& shapeStart =
				this->_inputData[0]->getShape().begin() + this->axis;
		const vector<uint32_t>::const_iterator& shapeEnd =
				(numAxes == -1) ? this->_inputData[0]->getShape().end() : (shapeStart + numAxes);
		vector<uint32_t> scaleShape(shapeStart, shapeEnd);
		// restore scaleShape numAxes to 4
		// guaranttes scales shape numAxes is 4
		const int scaleShapeSize = scaleShape.size();
		for (int i = 0; i < Data<Dtype>::SHAPE_SIZE - scaleShapeSize; i++) {
			scaleShape.insert(scaleShape.begin(), 1);
		}
		LearnableLayer<Dtype>::reshapeParam(0, scaleShape);

		param_filler<Dtype>& filler =  SLPROP(Scale, filler);
		filler.fill(this->_params[0]);

		UpdateParam upScale;
		upScale.paramType = 0;
		upScale.paramDataPtr = (void*)this->_params[0];
		upScale.paramHis1Ptr = (void*)this->_paramsHistory[0];
		upScale.paramHis2Ptr = (void*)this->_paramsHistory2[0];
		this->updateParams.push_back(upScale);

		if (this->biasTerm) {
			this->biasLayer = buildBiasLayer();
			// case: params.size == 1 && inputData.size == 1
			// or params.size == 0 && inputData.size == 2
			SASSERT(this->_inputData.size() == 1 && this->_params.size() == 1,
					"support only 1 input and 1 param case.");

			this->biasParamId = 1;

			this->_params.resize(2);
			this->_paramsHistory.resize(2);
			this->_paramsHistory2.resize(2);
			this->_paramsInitialized.resize(2);

			LearnableLayer<Dtype>* learnableLayer = dynamic_cast<LearnableLayer<Dtype>*>(
					this->biasLayer);
			SASSERT(learnableLayer, "biasLayer should be learnable layer.");

			//LearnableLayer<Dtype>::initParam(1, "bias");
			//LearnableLayer<Dtype>::reshapeParam(1, scaleShape);
			// set bias layer params as secondary params of scale layer
			this->_params[this->biasParamId] = learnableLayer->_params[0];
			this->_paramsHistory[this->biasParamId] = learnableLayer->_paramsHistory[0];
			this->_paramsHistory2[this->biasParamId] = learnableLayer->_paramsHistory2[0];
			this->_paramsInitialized[this->biasParamId] = learnableLayer->_paramsInitialized[0];

	        UpdateParam upBias;
	        upBias.paramType = this->biasParamId;
	        upBias.paramDataPtr = (void*)this->_params[this->biasParamId];
	        upBias.paramHis1Ptr = (void*)this->_paramsHistory[this->biasParamId];
	        upBias.paramHis2Ptr = (void*)this->_paramsHistory2[this->biasParamId];
	        this->updateParams.push_back(upBias);


			//this->_params[1]->print_shape();
			//this->_paramsHistory[1]->print_shape();
			//this->_paramsHistory2[1]->print_shape();
			//cout << this->_params[1] << "," << this->_paramsHistory[1] << "," << this->_paramsHistory2[1] << endl;
			//exit(1);
		}
	}

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	SASSERT(this->_inputData[0]->numAxes() >= this->axis + numAxes, 
			"scale params's shape extends past input[0]'s shape when applied "
			"starting with input[0] axis = %d", this->axis);
	for (int i = 0; i < numAxes; i++) {
		SASSERT(this->_inputData[0]->getShape(this->axis + i) == 
				this->_params[0]->getShape(Data<Dtype>::SHAPE_SIZE - numAxes + i),
				"dimension mismatch between input[0]->getShape(%d) and params[0]->getShape(%d)",
				this->axis + i, Data<Dtype>::SHAPE_SIZE - numAxes + i);
	}

	this->outerDim = this->_inputData[0]->getCountByAxis(0, this->axis);
	this->scaleDim = this->_params[0]->getCount();
	this->innerDim = this->_inputData[0]->getCountByAxis(this->axis + numAxes);
	if (this->_inputData[0] == this->_outputData[0]) {		// in-place computation
		this->temp.reshapeLike(this->_inputData[0]);
	} else {
		this->_outputData[0]->reshapeLike(this->_inputData[0]);
	}
	this->sumResult.reshape({1, 1, 1, (uint32_t)(this->outerDim * this->scaleDim)});
	const int sumMultSize = std::max(this->outerDim, this->innerDim);
	this->sumMultiplier.reshape({1, 1, 1, (uint32_t)sumMultSize});
	if (this->sumMultiplier.host_data()[sumMultSize - 1] != Dtype(1)) {
		soooa_set(sumMultSize, Dtype(1), this->sumMultiplier.mutable_host_data());
	}

	if (this->biasLayer) {
		this->biasLayer->_inputData[0] = this->_outputData[0];
		this->biasLayer->_outputData[0] = this->_outputData[0];
		this->biasLayer->reshape();
	}
}

template<typename Dtype>
__global__ void ScaleForward(const int n, const Dtype* in, const Dtype* scale,
		const int scale_dim, const int inner_dim, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n) {
		const int scale_index = (index / inner_dim) % scale_dim;
		out[index] = in[index] * scale[scale_index];
	}
}

template<typename Dtype>
__global__ void ScaleBiasForward(const int n, const Dtype* in,
		const Dtype* scale, const Dtype* bias, const int scale_dim,
		const int inner_dim, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n) {
		const int scale_index = (index / inner_dim) % scale_dim;
		out[index] = in[index] * scale[scale_index] + bias[scale_index];
	}
}

template<typename Dtype>
void ScaleLayer<Dtype>::feedforward() {
	const int count = this->_outputData[0]->getCount();
	const Dtype* inputData = this->_inputData[0]->device_data();
	if (this->_inputData[0] == this->_outputData[0]) {
		// in-place computation; need to store bottom data before overwriting it.
		// Note that this is only necessary for Backward; we could skip this if not
		// doing Backward, but Caffe currently provides no way of knowing whether
		// we'll need to do Backward at the time of the Forward call.
		soooa_copy(this->_inputData[0]->getCount(), this->_inputData[0]->device_data(),
				this->temp.mutable_device_data());
	}
	const Dtype* scaleData = this->_params[0]->device_data();
	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	if (this->biasLayer) {
		//const Dtype* biasData = this->_params[this->biasParamId]->device_data();
		const Dtype* biasData = this->biasLayer->_params[0]->device_data();
		ScaleBiasForward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(count,
				inputData, scaleData, biasData, this->scaleDim, this->innerDim, outputData);
	} else {
		ScaleForward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(count,
				inputData, scaleData, this->scaleDim, this->innerDim, outputData);
	}

}

template<typename Dtype>
void ScaleLayer<Dtype>::backpropagation() {
	if (this->biasLayer) {
		this->biasLayer->backpropagation();
	}

	const bool scaleParam = (this->_inputData.size() == 1);
	Data<Dtype>* scale = this->_params[0];

	{
		const Dtype* outputGrad = this->_outputData[0]->device_grad();
		const bool inPlace = (this->_inputData[0] == this->_outputData[0]);
		const Dtype* inputData = (inPlace ? &this->temp : this->_inputData[0])->device_data();
		// Hack: store big eltwise product in this->_inputData[0] diff, except in the special
		// case where this layer itself does the eltwise product, in which case we
		// can store it directly in the scale diff, and we're done.
		// If we're computing in-place (and not doing eltwise computation), this
		// hack doesn't work and we store the product in this->temp.
		const bool isEltwise = (this->_inputData[0]->getCount() == scale->getCount());
		Dtype* product = (isEltwise ? scale->mutable_device_grad() :
				(inPlace ? this->temp.mutable_device_data() :
						this->_inputData[0]->mutable_device_grad()));

		soooa_gpu_mul(this->_outputData[0]->getCount(), outputGrad, inputData, product);
		if (!isEltwise) {
			Dtype* sumResult = NULL;
			if (this->innerDim == 1) {
				sumResult = product;
			} else if (this->sumResult.getCount() == 1) {
				const Dtype* sumMult = this->sumMultiplier.device_data();
				Dtype* scaleGrad = scale->mutable_device_grad();
				if (scaleParam) {
					Dtype result;
					soooa_gpu_dot(this->innerDim, product, sumMult, &result);
					*scaleGrad += result;
				} else {
					soooa_gpu_dot(this->innerDim, product, sumMult, scaleGrad);
				}
			} else {
				const Dtype* sumMult = this->sumMultiplier.device_data();
				sumResult =
						(this->outerDim == 1) ?
								scale->mutable_device_grad() :
								this->sumResult.mutable_device_data();
				soooa_gpu_gemv(CblasNoTrans, this->sumResult.getCount(), this->innerDim,
						Dtype(1), product, sumMult, Dtype(0), sumResult);
			}
			if (this->outerDim != 1) {
				const Dtype* sumMult = this->sumMultiplier.device_data();
				if (this->scaleDim == 1) {
					Dtype* scaleGrad = scale->mutable_device_grad();
					if (scaleParam) {
						Dtype result;
						soooa_gpu_dot(this->outerDim, sumMult, sumResult,
								&result);
						*scaleGrad += result;
					} else {
						soooa_gpu_dot(this->outerDim, sumMult, sumResult,
								scaleGrad);
					}
				} else {
					Dtype* scaleGrad = scale->mutable_device_grad();
					soooa_gpu_gemv(CblasTrans, this->outerDim, this->scaleDim, Dtype(1),
							sumResult, sumMult, Dtype(scaleParam),
							scaleGrad);
				}
			}
		}
	}

	{
		const int count = this->_outputData[0]->getCount();
		const Dtype* outputGrad = this->_outputData[0]->device_grad();
		const Dtype* scaleData = scale->device_data();
		Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
		ScaleForward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(count,
				outputGrad, scaleData, this->scaleDim, this->innerDim, inputGrad);
	}
}

template <typename Dtype>
void ScaleLayer<Dtype>::update() {
	//this->biasLayer->update();

	const Dtype weightDecay = SNPROP(weightDecay);
	const Dtype learningRate = Update<float>::calcLearningRate();
	const Dtype beta1 = SNPROP(beta1);
	const Dtype beta2 = SNPROP(beta2);

	SLPROP(Bias, decayedBeta1) *= beta1;
	SLPROP(Bias, decayedBeta2) *= beta2;

	for (int i = 0; i < this->_params.size(); i++) {
		int paramSize = this->_params[i]->getCount();
		Dtype regScale = weightDecay * this->updatePolicies[i].decay_mult;
		Dtype learnScale = learningRate * this->updatePolicies[i].lr_mult;
		UpdateContext context = Update<Dtype>::makeContext(paramSize, regScale, learnScale);
		this->updateParams[i].context = context;
	}
	Updater::updateParams(this->updateParams);
}



template <typename Dtype>
void ScaleLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}

template <typename Dtype>
void ScaleLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}




template <typename Dtype>
BiasLayer<Dtype>* ScaleLayer<Dtype>::buildBiasLayer() {
	int innerBiasId = BiasLayer<Dtype>::INNER_ID;
	BiasLayer<Dtype>::INNER_ID += 10;

	int axis = SLPROP(Scale, axis);
	int numAxes = SLPROP(Scale, numAxes);
	param_filler<Dtype> biasFiller = SLPROP(Scale, biasFiller);


	// XXX: 임시로 여기서 BiasFillerType을 string으로 변환
	string biasFillerType;
	switch(biasFiller.type) {
	case Constant: biasFillerType = "Constant"; break;
	case Xavier: biasFillerType = "Xavier"; break;
	case Gaussian: biasFillerType = "Gaussian"; break;
	case MSRA: biasFillerType = "MSRA"; break;
	default: SASSERT(false, "unsupported biasFillerType: %d", biasFiller.type);
	}



	// in-place bias layer definition
	stringstream biasDef;
	biasDef << "{\n";
	biasDef << "\t\"name\" : \"inner_bias\",\n";
	biasDef << "\t\"id\" : " << innerBiasId << ",\n";
	biasDef << "\t\"layer\" : \"Bias\",\n";
	biasDef << "\t\"input\" : [\"inner_bias_" << innerBiasId << "\"],\n";
	biasDef << "\t\"output\" : [\"inner_bias_" << innerBiasId << "\"],\n";
	biasDef << "\t\"axis\" : " << axis << ",\n";
	biasDef << "\t\"numAxes\" : " << numAxes << ",\n";
	biasDef << "\t\"filler.type\" : \"" << biasFillerType << "\",\n";
	biasDef << "\t\"filler.value\" : " << biasFiller.value << ",\n";
	biasDef << "\t\"filler.mean\" : " << biasFiller.mean << ",\n";
	biasDef << "\t\"filler.std\" : " << biasFiller.std << "\n";
	biasDef << "}\n";
	//cout << biasDef.str() << endl;

	_BiasPropLayer* innerProp = NULL;
	SNEW(innerProp, _BiasPropLayer);
	SASSUME0(innerProp != NULL);

	Json::Reader reader;
	Json::Value layer;
	reader.parse(biasDef, layer);

	vector<string> keys = layer.getMemberNames();
	string layerType = layer["layer"].asCString();

	for (int j = 0; j < keys.size(); j++) {
		string key = keys[j];
		Json::Value val = layer[key.c_str()];
		if (strcmp(key.c_str(), "layer") == 0) continue;
		if (strcmp(key.c_str(), "innerLayer") == 0) continue;

		PlanParser::setPropValue(val, true, layerType, key,  (void*)innerProp);
	}

	BiasLayer<Dtype>* biasLayer = NULL;
	SNEW(biasLayer, BiasLayer<Dtype>, innerProp);
	SASSUME0(biasLayer != NULL);

	SDELETE(innerProp);

	biasLayer->_inputData.push_back(this->_inputData[0]);
	biasLayer->_outputData.push_back(this->_outputData[0]);

	return biasLayer;
}



















/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* ScaleLayer<Dtype>::initLayer() {
	ScaleLayer* layer = NULL;
	SNEW(layer, ScaleLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ScaleLayer<Dtype>::destroyLayer(void* instancePtr) {
    ScaleLayer<Dtype>* layer = (ScaleLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ScaleLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    ScaleLayer<Dtype>* layer = (ScaleLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ScaleLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ScaleLayer<Dtype>* layer = (ScaleLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ScaleLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    ScaleLayer<Dtype>* layer = (ScaleLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void ScaleLayer<Dtype>::backwardTensor(void* instancePtr) {
    ScaleLayer<Dtype>* layer = (ScaleLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void ScaleLayer<Dtype>::learnTensor(void* instancePtr) {
    ScaleLayer<Dtype>* layer = (ScaleLayer<Dtype>*)instancePtr;
    layer->update();
}

template<typename Dtype>
bool ScaleLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 1)
        return false;

    TensorShape outputShape1 = inputShape[0];
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t ScaleLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
	Optimizer opt = (Optimizer)SNPROP(optimizer);
	int paramHistoryDataCount = Update<Dtype>::getParamHistoryDataCount(opt);

    vector<uint32_t> _inputShape;
    _inputShape.push_back(inputShape[0].N);
    _inputShape.push_back(inputShape[0].C);
    _inputShape.push_back(inputShape[0].H);
    _inputShape.push_back(inputShape[0].W);

    const int axis = SLPROP(Scale, axis);
	const int numAxes = SLPROP(Scale, numAxes);
    const vector<uint32_t>::const_iterator& shapeStart = _inputShape.begin() + axis;
    const vector<uint32_t>::const_iterator& shapeEnd = (numAxes == -1) ? _inputShape.end() : 
        (shapeStart + numAxes);

    vector<uint32_t> scaleShape(shapeStart, shapeEnd);
    // restore scaleShape numAxes to 4
    // guaranttes scales shape numAxes is 4
    const int scaleShapeSize = scaleShape.size();
    for (int i = 0; i < Data<Dtype>::SHAPE_SIZE - scaleShapeSize; i++) {
        scaleShape.insert(scaleShape.begin(), 1);
    }

    size_t scaleCount = 1;
    for (int i = 0; i < scaleShape.size(); i++) {
        scaleCount *= scaleShape[i];
    }

    size_t size = 0;
    size += ALIGNUP(sizeof(Dtype) * scaleCount, SPARAM(CUDA_MEMPAGE_SIZE)) * 
        paramHistoryDataCount * 2UL;

    SASSERT(false, "not supported yet");


    // sumMultiplier
    // sumResult
    // temp
    // biasInput
    // biasOutput
    return size;
}

template class ScaleLayer<float>;
