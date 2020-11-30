/*
 * PriorBoxLayer.cpp
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#include "PriorBoxLayer.h"
#include "SysLog.h"
#include "MathFunctions.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

template <typename Dtype>
PriorBoxLayer<Dtype>::PriorBoxLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::PriorBox;

	vector<Dtype> minSize;
	vector<Dtype> maxSize;
	vector<Dtype> aspectRatio;
	vector<Dtype> variance;

	SASSERT(SLPROP(PriorBox, minSize).size() > 0, "must provide minSizes.");
	for (int i = 0; i < SLPROP(PriorBox, minSize).size(); i++) {
		minSize.push_back(SLPROP(PriorBox, minSize)[i]);
		SASSERT(minSize.back() > 0, "minSize must be positive.");
	}
	aspectRatio.clear();
	aspectRatio.push_back(Dtype(1));
	for (int i = 0; i < SLPROP(PriorBox, aspectRatio).size(); i++) {
		Dtype ar = SLPROP(PriorBox, aspectRatio)[i];
		bool alreadyExsit = false;
		for (int j = 0; j < aspectRatio.size(); j++) {
			if (fabs(ar - aspectRatio[j]) < 1e-6) {
				alreadyExsit = true;
				break;
			}
		}
		if (!alreadyExsit) {
			aspectRatio.push_back(ar);
			if (SLPROP(PriorBox, flip)) {
				aspectRatio.push_back(1 / ar);
			}
		}
	}
	this->numPriors = aspectRatio.size() * minSize.size();
	if (SLPROP(PriorBox, maxSize).size() > 0) {
		SASSERT0(SLPROP(PriorBox, minSize).size() == SLPROP(PriorBox, maxSize).size());
		for (int i = 0; i < SLPROP(PriorBox, maxSize).size(); i++) {
			maxSize.push_back(SLPROP(PriorBox, maxSize)[i]);
			SASSERT(maxSize[i] > minSize[i],
					"maxSize must b greater than minSize.");
			this->numPriors += 1;
		}
	}
	if (SLPROP(PriorBox, variance).size() > 1) {
		// Must and only provide 4 variance.
		SASSERT0(SLPROP(PriorBox, variance).size() == 4);
		for (int i = 0; i < SLPROP(PriorBox, variance).size(); i++) {
			SASSERT0(SLPROP(PriorBox, variance)[i] > 0);
			variance.push_back(SLPROP(PriorBox, variance)[i]);
		}
	} else if (SLPROP(PriorBox, variance).size() == 1) {
		SASSERT0(SLPROP(PriorBox, variance)[0] > 0);
		variance.push_back(SLPROP(PriorBox, variance)[0]);
	} else {
		// set tdefault to 0.1.
		variance.push_back(Dtype(0.1));
	}

	if (SLPROP(PriorBox, imgH) >= 0 || SLPROP(PriorBox, imgW) >= 0) {
		SASSERT(!(SLPROP(PriorBox, imgSize) >= 0),
				"Either imgSize or imgH/imgW should be specified; not both.");
		SASSERT(SLPROP(PriorBox, imgH) > 0, "imgH should be larger than 0.");
		SASSERT(SLPROP(PriorBox, imgW) > 0, "imgW should be larger than 0.");
	} else if (SLPROP(PriorBox, imgSize) >= 0) {
		SASSERT(SLPROP(PriorBox, imgSize) > 0, "imgSize should be larger than 0.");
		SLPROP(PriorBox, imgH) = SLPROP(PriorBox, imgSize);
		SLPROP(PriorBox, imgW) = SLPROP(PriorBox, imgSize);
	} else {
		SLPROP(PriorBox, imgH) = 0;
		SLPROP(PriorBox, imgW) = 0;
	}

	if (SLPROP(PriorBox, stepH) >= Dtype(0) || SLPROP(PriorBox, stepW) >= Dtype(0)) {
		SASSERT(!(SLPROP(PriorBox, step) >= 0),
				"Either step or stepH/stepW should be specified; not both.");
		SASSERT(SLPROP(PriorBox, stepH) > Dtype(0), "stepH should be larger than 0.");
		SASSERT(SLPROP(PriorBox, stepW) > Dtype(0), "stepW should be larger than 0.");
	} else if (SLPROP(PriorBox, step) >= Dtype(0)) {
		SASSERT(SLPROP(PriorBox, step) > Dtype(0), "step should be larger than 0.");
		SLPROP(PriorBox, stepH) = SLPROP(PriorBox, step);
		SLPROP(PriorBox, stepW) = SLPROP(PriorBox, step);
	} else {
		SLPROP(PriorBox, stepH) = Dtype(0);
		SLPROP(PriorBox, stepW) = Dtype(0);
	}

	SLPROP(PriorBox, minSize) = minSize;
	SLPROP(PriorBox, maxSize) = maxSize;
	SLPROP(PriorBox, aspectRatio) = aspectRatio;
	SLPROP(PriorBox, variance) = variance;
}

template <typename Dtype>
PriorBoxLayer<Dtype>::~PriorBoxLayer() {

}


template <typename Dtype>
void PriorBoxLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();

	if (!adjusted)
		return;

	const int layerWidth = this->_inputData[0]->width();
	const int layerHeight = this->_inputData[0]->height();

	/*
	vector<uint32_t> outputShape(4, 1);
	// since all images in a batch has same height and width, we only need to
	// generate one set of priors which can be shared across all images.
	outputShape[0] = 1;
	outputShape[1] = 1;
	// 2 channels. first channel stores the mean of each prior coordinate.
	// second channel stores the variance of each prior coordinate.
	outputShape[2] = 2;
	outputShape[3] = layerWidth * layerHeight * this->numPriors * 4;
	SASSERT0(outputShape[3] > 0);
	*/
	vector<uint32_t> outputShape(4, 1);
	outputShape[0] = 2;
	outputShape[1] = layerWidth * layerHeight * this->numPriors * 4;
	SASSERT0(outputShape[1] > 0);

	this->_outputData[0]->reshape(outputShape);
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::feedforward() {
	reshape();


	/*
	if (SLPROP_BASE(name) == "conv8_2_mbox_priorbox") {
		cout << SLPROP_BASE(name) << " forward()" << endl;
		this->_printOn();
		this->_inputData[0]->print_grad({}, false);
		this->_printOff();
	}
	*/



	const int layerHeight = this->_inputData[0]->height();
	const int layerWidth = this->_inputData[0]->width();
	int imgHeight;
	int imgWidth;
	if (SLPROP(PriorBox, imgH) == 0 || SLPROP(PriorBox, imgW) == 0) {
		imgHeight = this->_inputData[1]->height();
		imgWidth = this->_inputData[1]->width();
	} else {
		imgHeight = SLPROP(PriorBox, imgH);
		imgWidth = SLPROP(PriorBox, imgW);
	}

	Dtype stepH;
	Dtype stepW;
	if (SLPROP(PriorBox, stepH) == 0 || SLPROP(PriorBox, stepW) == 0) {
		stepH = Dtype(imgHeight) / layerHeight;
		stepW = Dtype(imgWidth) / layerWidth;
	} else {
		stepH = SLPROP(PriorBox, stepH);
		stepW = SLPROP(PriorBox, stepW);
	}

	Dtype* outputData = this->_outputData[0]->mutable_host_data();
	int dim = layerHeight * layerWidth * this->numPriors * 4;
	int idx = 0;
	for (int h = 0; h < layerHeight; h++) {
		for (int w = 0; w < layerWidth; w++) {
			Dtype centerX = (w + SLPROP(PriorBox, offset)) * stepW;
			Dtype centerY = (h + SLPROP(PriorBox, offset)) * stepH;
			Dtype boxWidth;
			Dtype boxHeight;
			for (int s = 0; s < SLPROP(PriorBox, minSize).size(); s++) {
				int minSize = SLPROP(PriorBox, minSize)[s];
				// first prior: aspectRatio = 1, size = minSize
				boxWidth = boxHeight = minSize;
				// xmin
				outputData[idx++] = (centerX - boxWidth / Dtype(2)) / imgWidth;
				// ymin
				outputData[idx++] = (centerY - boxHeight / Dtype(2)) / imgHeight;
				// xmax
				outputData[idx++] = (centerX + boxWidth / Dtype(2)) / imgWidth;
				// ymax
				outputData[idx++] = (centerY + boxHeight / Dtype(2)) / imgHeight;

				if (SLPROP(PriorBox, maxSize).size() > 0) {
					SASSERT0(SLPROP(PriorBox, minSize).size() == SLPROP(PriorBox, maxSize).size());
					int maxSize = SLPROP(PriorBox, maxSize)[s];
					// second prior: aspectRatio = 1, size = sqrt(minSize * maxSize)
					boxWidth = boxHeight = sqrt(minSize * maxSize);
					// xmin
					outputData[idx++] = (centerX - boxWidth / Dtype(2)) / imgWidth;
					// ymin
					outputData[idx++] = (centerY - boxHeight / Dtype(2)) / imgHeight;
					// xmax
					outputData[idx++] = (centerX + boxWidth / Dtype(2)) / imgWidth;
					// ymax
					outputData[idx++] = (centerY + boxHeight / Dtype(2)) / imgHeight;
				}

				// rest of priors
				for (int r = 0; r < SLPROP(PriorBox, aspectRatio).size(); r++) {
					Dtype ar = SLPROP(PriorBox, aspectRatio)[r];
					if (fabs(ar - Dtype(1)) < 1e-6) {
						continue;
					}
					boxWidth = minSize * sqrt(ar);
					boxHeight = minSize / sqrt(ar);
					// xmin
					outputData[idx++] = (centerX - boxWidth / Dtype(2)) / imgWidth;
					// ymin
					outputData[idx++] = (centerY - boxHeight / Dtype(2)) / imgHeight;
					// xmax
					outputData[idx++] = (centerX + boxWidth / Dtype(2)) / imgWidth;
					// ymax
					outputData[idx++] = (centerY + boxHeight / Dtype(2)) / imgHeight;
				}
			}
		}
	}
	// clip the prior's coordinate such that it is within [0, 1]
	if (SLPROP(PriorBox, clip)) {
		for (int d = 0; d < dim; d++) {
			outputData[d] = std::min<Dtype>(std::max<Dtype>(outputData[d], Dtype(0)), Dtype(1));
		}
	}
	// set the variance.
	outputData += this->_outputData[0]->offset(1, 0, 0, 0);
	if (SLPROP(PriorBox, variance).size() == 1) {
		soooa_set<Dtype>(dim, Dtype(SLPROP(PriorBox, variance)[0]), outputData);
	} else {
		int count = 0;
		for (int h = 0; h < layerHeight; h++) {
			for (int w = 0; w < layerWidth; w++) {
				for (int i = 0; i < this->numPriors; i++) {
					for (int j = 0; j < 4; j++) {
						outputData[count] = SLPROP(PriorBox, variance)[j];
						count++;
					}
				}
			}
		}
	}
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::backpropagation() {
	/*
	if (SLPROP_BASE(name) == "conv8_2_mbox_priorbox") {
		cout << SLPROP_BASE(name) << " backward()" << endl;
		this->_printOn();
		this->_inputData[0]->print_grad({}, false);
		this->_printOff();

		exit(1);
	}
	*/
	return;
}













/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* PriorBoxLayer<Dtype>::initLayer() {
	PriorBoxLayer* layer = NULL;
	SNEW(layer, PriorBoxLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void PriorBoxLayer<Dtype>::destroyLayer(void* instancePtr) {
    PriorBoxLayer<Dtype>* layer = (PriorBoxLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void PriorBoxLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 2);
	} else {
		SASSERT0(index < 1);
	}

    PriorBoxLayer<Dtype>* layer = (PriorBoxLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool PriorBoxLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    PriorBoxLayer<Dtype>* layer = (PriorBoxLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void PriorBoxLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	PriorBoxLayer<Dtype>* layer = (PriorBoxLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void PriorBoxLayer<Dtype>::backwardTensor(void* instancePtr) {
	PriorBoxLayer<Dtype>* layer = (PriorBoxLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void PriorBoxLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool PriorBoxLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 2)
        return false;

	const int layerWidth = inputShape[0].W;
	const int layerHeight = inputShape[0].H;
    const int numMinSize = SLPROP(PriorBox, minSize).size();
    const int numMaxSize = SLPROP(PriorBox, maxSize).size();
    const int numAspectRatio = SLPROP(PriorBox, aspectRatio).size();
    int numPriors = numMinSize * numAspectRatio;

    if (numMinSize != numMaxSize)
        return false;
    numPriors += numMaxSize;

    TensorShape outputShape1;
	outputShape1.N = 2;
	outputShape1.C = layerWidth * layerHeight * numPriors * 4;
    if (outputShape1.C <= 0)
        return false;
	outputShape1.H = 1;
	outputShape1.W = 1;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t PriorBoxLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class PriorBoxLayer<float>;
